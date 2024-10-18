# Import Libraries
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
import math
from transformers import get_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import os
from PIL import Image
import logging
import shutil
import argparse
from pathlib import Path
import itertools
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionAttnProcessor2_0
)

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

# Import our custom modules
from dataset import load_and_preprocess_eeg_data
from things_dataloader import create_dataset
from temp_encoder.encoder import Encoder
from temp_encoder.enc_config import Config as EncoderConfig
from config import Config
from unet_utils import First_Stage, FrozenImageEmbedder

check_min_version("0.23.0.dev0")

# The participant to omit
parser = argparse.ArgumentParser(description='Process the omitted participant.')
parser.add_argument('--omit_participant', type=int, help='The participant to omit')
args = parser.parse_args()
participants = [i for i in range(1, 7) if i != args.omit_participant]

train_only_attn2_layers = True #If false it trains attn1 and attn2

# Case 0 is train both the unet and the encoder
# Case 1 is train the unet and freeze the encoder
# Case 2 is freeze the unet and train the encoder
case_num = 2

# Dataset 0 is the VC dataset
# Dataset 1 is the Things dataset
# Dataset 2 is all particpants for VC dataset except one
dataset_num = 1

encoder_name ='encoder_final'
encoder_path = f'temp_encoder/{encoder_name}.pth'

#checkpoint = 'model_iter4.pt'
checkpoint = None
switch_dataset = {
    0: "VC_dataset",
    1: "THINGS_dataset",
    2: "VC_dataset_all"
}

cur_date = datetime.now().strftime("%Y-%m-%d")
log_name = f'log2_{cur_date}_omit_{args.omit_participant}.log'
image_generation_path = f"validation_image_{cur_date}__omit_{args.omit_participant}"
model_save_path = f"model_{cur_date}_case_{case_num}_{switch_dataset[dataset_num]}_{encoder_name}_omit_{args.omit_participant}.pt"

# Set up logging
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)  # Capture all logs

# Configure file handler for INFO and DEBUG logs
info_handler = logging.FileHandler(log_name)
info_handler.setLevel(logging.INFO)
info_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
info_handler.setFormatter(info_formatter)
info_handler.addFilter(lambda record: record.levelno <= logging.INFO)  # Only logs <= INFO level

# Configure file handler for WARNING and above logs
warning_handler = logging.FileHandler('warning.log')
warning_handler.setLevel(logging.WARNING)
warning_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
warning_handler.setFormatter(warning_formatter)

# Add handlers to the logger
logger.addHandler(info_handler)
logger.addHandler(warning_handler)


def main():
    # Set up the training parameters
    config = Config()

    # Set up the accelerator
    accelerator = Accelerator()
    set_seed(42)

    logger.info(f'lr: {config.lr}; batch: {config.train_batch_size}; Gradient Acc: {config.gradient_accumulation_steps}; Checkpoint: {checkpoint}; Encoder: {encoder_name}; Dataset: {switch_dataset[dataset_num]}')

    # Load the models
    unet, first_stage, vae, frozen_image_embedder, noise_scheduler, optimizer = load_models(config,accelerator,checkpoint)
    
    # Set up the EEG dataset
    with accelerator.main_process_first():
        if dataset_num == 0:  
            train_dataset, test_dataset, tmp_path1 = load_and_preprocess_eeg_data(eeg_signals_path="data/eeg.pth", split_path="data/splits.pth",participant_number=args.omit_participant)
            logger.info(f"VC Data loaded on {accelerator.state.process_index}")
            train_dataset = torch.utils.data.ConcatDataset([train_dataset,test_dataset])
        elif dataset_num == 1:
            train_dataset, test_dataset = create_dataset("data/sub-04/eeg/sub-04_task-rsvp_eeg.vhdr",'Event/E  1',  "data/sub-04/eeg/sub-04_task-rsvp_events.csv" )
            logger.info(f"Things Data loaded on {accelerator.state.process_index}")
        elif dataset_num == 2:
            all_data_train = []
            all_data_test = []
            for i in participants:
                train_dataset, test_dataset, tmp_path1 = load_and_preprocess_eeg_data(eeg_signals_path="data/eeg.pth", split_path="data/splits.pth",participant_number=i)
                all_data_train.append(train_dataset)
                #all_data_test.append(test_dataset)
                if accelerator.is_main_process:
                    logger.info(f"Participant {i} data loaded")
            ___, test_dataset, __ = load_and_preprocess_eeg_data(eeg_signals_path="data/eeg.pth", split_path="data/splits.pth",participant_number=args.omit_participant)
            train_dataset = torch.utils.data.ConcatDataset(all_data_train)
            #test_dataset = torch.utils.data.ConcatDataset(all_data_test)
            
    train_loader, test_loader = create_EEG_dataset(config.train_batch_size,train_dataset, test_dataset)

    if accelerator.is_main_process:
        logger.info(f"Data is {len(train_loader)} batches long")

    # Prepare the training
    train_loader,progress_bar,global_step,lr_scheduler,max_train_steps,test_loader, unet, first_stage = prepare_training(config,train_loader,accelerator,test_loader,optimizer,unet,first_stage)
    
    # Train the model
    logger.info(f"Rank: {accelerator.state.process_index} waiting for everyone")
    accelerator.wait_for_everyone()
    logger.info(f"Rank: {accelerator.state.process_index}; Training started with {len(train_loader)} batches")

    train(train_loader,accelerator, unet, first_stage,vae,frozen_image_embedder, optimizer, lr_scheduler, config, global_step, progress_bar,args.omit_participant, test_loader,noise_scheduler,max_train_steps)
            
    # Clean up the temporary directory
    if tmp_path1 and Path(tmp_path1).exists():
        shutil.rmtree(tmp_path1)
    else:
        logger.info(f"Temporary path {tmp_path1} does not exist")

    train_dataset, test_dataset, train_loader, test_loader = None, None, None, None
    logger.info(f"Participant {args.omit_participant} training completed on {accelerator.state.process_index}")
    
    
    # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        first_stage = accelerator.unwrap_model(first_stage)
        torch.save({
            'first_stage': first_stage.state_dict(),
            'unet': unet.state_dict(),
        }, model_save_path)

        logger.info("Training completed")

    torch.cuda.empty_cache()

    accelerator.end_training()

def prepare_training(config,train_loader,accelerator,test_loader,optimizer,unet,first_stage):
    # Set up the other training parameters
    num_update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    max_train_steps = config.num_epoch * num_update_steps_per_epoch
    num_warmup_steps = 0.1 * max_train_steps  # 10% of max_train_steps as an example
    num_training_steps = max_train_steps
    global_step = 0
    progress_bar = tqdm(range(0, max_train_steps),initial=global_step,desc="Steps",disable=not accelerator.is_local_main_process)

    # Set up the learning rate scheduler
    lr_scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=num_warmup_steps * accelerator.num_processes,num_training_steps=num_training_steps * accelerator.num_processes)
        
    # Prepare everything for the accelerator so that it can be used for distributed training
    train_loader, lr_scheduler,test_loader,optimizer, first_stage, unet = accelerator.prepare(train_loader, lr_scheduler,test_loader,optimizer, first_stage, unet)

    return train_loader,progress_bar,global_step,lr_scheduler,max_train_steps,test_loader, unet, first_stage
 

def load_models(config,accelerator,weight_dtype=torch.float32):

    #Load the pretrained models
    noise_scheduler = DDPMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet")
    vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae")

    encoder_config=EncoderConfig()
    encoder = Encoder(patch_size=encoder_config.patch_size,
        embed_dim=encoder_config.embed_dim,
        decoder_embed_dim=encoder_config.decoder_embed_dim,
        depth=encoder_config.depth,
        num_heads=encoder_config.num_heads,
        decoder_num_heads=encoder_config.decoder_num_heads,
        mlp_ratio=encoder_config.mlp_ratio,)
    cc = torch.load(encoder_path, map_location=accelerator.device)
    encoder.load_state_dict(cc['model'])  

    frozen_image_embedder = FrozenImageEmbedder()

    cross_attention_params_to_train = []
    state_dict = unet.state_dict()

    if train_only_attn2_layers:
        for name, _ in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if cross_attention_dim is not None:
                layer_name = name.split(".processor")[0]
                param_names = [
                    f"{layer_name}.to_k.weight",
                    f"{layer_name}.to_v.weight",
                    f"{layer_name}.to_q.weight",
                    f"{layer_name}.to_out.0.weight",
                    f"{layer_name}.to_out.0.bias"
                ]
                cross_attention_params_to_train.extend([param for param in param_names if param in state_dict])
    else:
        for name, _ in unet.attn_processors.items():
            layer_name = name.split(".processor")[0]
            param_names = [
                    f"{layer_name}.to_k.weight",
                    f"{layer_name}.to_v.weight",
                    f"{layer_name}.to_q.weight",
                    f"{layer_name}.to_out.0.weight",
                    f"{layer_name}.to_out.0.bias"
                ]
            cross_attention_params_to_train.extend([param for param in param_names if param in state_dict])

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    frozen_image_embedder.requires_grad_(False)

    # Move each model to the appropriate device
    vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device)
    frozen_image_embedder.to(accelerator.device)

    vae.eval() 
    unet.train()

    first_stage = First_Stage(encoder=encoder)
    first_stage.to(accelerator.device)
    first_stage.train()

 

    for param in first_stage.parameters():
        param.requires_grad = True

    for param in first_stage.mapper.parameters():
        param.requires_grad = False

    def disable_grad_for_substrings(model, substrings):
        for name, param in model.named_parameters():
            if any(substring in name for substring in substrings):  
                param.requires_grad = False
                
    substrings_to_freeze = ["decoder", "mask_token", "cls_token"]
    disable_grad_for_substrings(first_stage, substrings_to_freeze)


    for name, param in unet.named_parameters():
        if any(name == p for p in cross_attention_params_to_train):
            param.requires_grad = True
    
    
    if case_num == 1:
        for param in first_stage.parameters():
            param.requires_grad = False
    elif case_num == 2:
        for param in unet.parameters():
            param.requires_grad = False
    
    params_to_train = []
    for name, param in first_stage.named_parameters():
        if param.requires_grad:
            params_to_train.append(name)

    for name, param in unet.named_parameters():
        if param.requires_grad:
            params_to_train.append(name)
    
#    logger.info(f"Parameters to train: {params_to_train}")

    optimizer_cls = torch.optim.AdamW

    if case_num == 0:
        optimizer = optimizer_cls(itertools.chain(first_stage.parameters(), unet.parameters()),lr=config.lr,weight_decay=config.weight_decay,eps=config.eps)
    elif case_num == 1:
        optimizer = optimizer_cls(unet.parameters(),lr=config.lr,weight_decay=config.weight_decay,eps=config.eps)
    elif case_num == 2:
        optimizer = optimizer_cls(first_stage.parameters(),lr=config.lr,weight_decay=config.weight_decay,eps=config.eps)


    return unet,first_stage, vae, frozen_image_embedder, noise_scheduler, optimizer


def create_EEG_dataset(train_batch_size,train_dataset, test_dataset):
  
    def collate_fn(examples):
        examples = [example for example in examples if example is not None]
        pixel_values = torch.stack([example['image_raw']for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["eeg"] for example in examples])
        return {"pixel_values": pixel_values, "eeg": input_ids}

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4,collate_fn=collate_fn, shuffle=True)

    return train_loader, test_loader


def train(train_loader,accelerator, unet, first_stage,vae,frozen_image_embedder, optimizer, lr_scheduler, config, global_step, progress_bar,participant, test_loader,noise_scheduler,max_train_steps,weight_dtype=torch.float32):
    clip_loss = True
    
    # Start of training code:
    for epoch in range(0, config.num_epoch):
        unet.train()
        first_stage.train()
        train_loss = 0.0
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            # Use the accelerator's context manager to accumulate gradients for efficient training
            with accelerator.accumulate(unet), accelerator.accumulate(first_stage):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states, ___ = first_stage(batch["eeg"].to(accelerator.device))

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                if clip_loss:
                    img_embd = frozen_image_embedder(batch['pixel_values'].to(accelerator.device))
                    clip_loss = first_stage.module.get_clip(batch['eeg'], img_embd)
                    loss = loss + clip_loss

                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                
                accelerator.backward(loss)
                
                # If using synchronized gradients, clip gradients to avoid exploding gradients problem
                if accelerator.sync_gradients:
                    if case_num == 0:
                        accelerator.clip_grad_norm_(itertools.chain(first_stage.parameters(), unet.parameters()), config.max_grad_norm)
                    elif case_num == 1:
                        accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                    elif case_num == 2:
                        accelerator.clip_grad_norm_(first_stage.parameters(), config.max_grad_norm)
                
                # Update the model parameters and learning rate
                optimizer.step()
                lr_scheduler.step()
                
                # Reset gradients for the next iteration
                optimizer.zero_grad()

            # Update the progress bar and log the training progress
            if accelerator.sync_gradients: 
                progress_bar.update(1)
                global_step += 1
                epoch_loss += train_loss

                # Reset the training loss for the next iteration
                train_loss = 0.0

            # Update the progress bar with the latest training metrics
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

        if accelerator.is_main_process:
            logger.info(f"Epoch: {epoch} --> Total Loss {epoch_loss}; Avg Batch Loss {epoch_loss/len(train_loader)}")
        # Check if we need to run validation
        if accelerator.is_main_process:
            if epoch % config.validation_epochs == 0:
                log_validation(test_loader,participant,unet,vae,first_stage,epoch,accelerator,noise_scheduler, config,train_loader)


# Validation function
def log_validation(test_loader,participant,unet,vae,first_stage,epoch,accelerator,noise_scheduler, config,train_loader):
        
    # Set the model to evaluation mode to disable dropout and batch normalization effects during inference
        unet.eval()
        first_stage.eval()
   
        folder_path_train = f"val/train/"
        folder_path_test = f"val/test/" 

        
        image_name = str(train_loader.dataset[1]['image_path']).split("/")[-1]
        sample = train_loader.dataset[1]['eeg'].to(accelerator.device)
      
        sample = sample.unsqueeze(0)
        embedding, __ = first_stage(sample)
        generator = torch.manual_seed(42)
                
        # Initialize unconditional embeddings as zeros, which will be used for guided diffusion
        uncond_embeddings = torch.zeros_like(embedding)
        #embedding = embedding.unsqueeze(0)
                
        # Combine unconditional and conditional embeddings for guided diffusion
        eeg_embeddings = torch.cat([uncond_embeddings, embedding]).to(accelerator.device)
                
        # Initialize latent vectors with random noise, shaping them for the image generation process
        latents = torch.randn((config.images_to_produce, unet.module.config.in_channels, config.height // 8, config.width // 8), generator=generator).to(accelerator.device)
                
        # Prepare the latents for the diffusion process by scaling with the initial noise level
        noise_scheduler.set_timesteps(config.num_inference_steps)
        latents = latents * noise_scheduler.init_noise_sigma
                
        # Iteratively denoise the latents through the diffusion process, applying classifier-free guidance
        for t in tqdm(noise_scheduler.timesteps):
            # Duplicate the latents for unconditional and conditional paths
            latent_model_input = torch.cat([latents] * 2)
            
            # Scale the model input for the current timestep
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                    
            with torch.no_grad():
                # Predict the noise component to be subtracted from the latents
                noise_pred = unet(latent_model_input, t, eeg_embeddings).sample
                    
            # Split predictions into unconditional and conditional components
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    
            # Apply classifier-free guidance to enhance the conditional generation's fidelity
            noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
            # Update the latents based on the predicted noise
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
                
        # Decode the latents into images
        latents = 1 / 0.18215 * latents  # Normalize latents before decoding
        with torch.no_grad():
            image = vae.decode(latents).sample  # Decode latents to images
                
        # Process and save the generated images
        image = (image / 2 + 0.5).clamp(0, 1)  # Normalize images
        image=image.detach().cpu().permute(0, 2, 3, 1).numpy()  # Prepare for conversion to PIL format
        images = (image * 255).round().astype("uint8")  # Convert to uint8 format
        pil_images = [Image.fromarray(image) for image in images]  # Convert numpy arrays to PIL images
        image_path = f"{folder_path_train}/{epoch}_{image_generation_path}_{image_name}"  # Define the path for saving the validation image
        pil_images[0].save(image_path)





        image_name = str(test_loader.dataset[1]['image_path']).split("/")[-1]
        sample = test_loader.dataset[1]['eeg'].to(accelerator.device)
        sample = sample.unsqueeze(0)
        embedding, __ = first_stage(sample)
        generator = torch.manual_seed(42)
                
        # Initialize unconditional embeddings as zeros, which will be used for guided diffusion
        uncond_embeddings = torch.zeros_like(embedding)
        #embedding = embedding.unsqueeze(0)
                
        # Combine unconditional and conditional embeddings for guided diffusion
        eeg_embeddings = torch.cat([uncond_embeddings, embedding]).to(accelerator.device)
                
        # Initialize latent vectors with random noise, shaping them for the image generation process
        latents = torch.randn((config.images_to_produce, unet.module.config.in_channels, config.height // 8, config.width // 8), generator=generator).to(accelerator.device)
                
        # Prepare the latents for the diffusion process by scaling with the initial noise level
        noise_scheduler.set_timesteps(config.num_inference_steps)
        latents = latents * noise_scheduler.init_noise_sigma
                
        # Iteratively denoise the latents through the diffusion process, applying classifier-free guidance
        for t in tqdm(noise_scheduler.timesteps):
            # Duplicate the latents for unconditional and conditional paths
            latent_model_input = torch.cat([latents] * 2)
            
            # Scale the model input for the current timestep
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                    
            with torch.no_grad():
                # Predict the noise component to be subtracted from the latents
                noise_pred = unet(latent_model_input, t, eeg_embeddings).sample
                    
            # Split predictions into unconditional and conditional components
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    
            # Apply classifier-free guidance to enhance the conditional generation's fidelity
            noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
            # Update the latents based on the predicted noise
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
                
        # Decode the latents into images
        latents = 1 / 0.18215 * latents  # Normalize latents before decoding
        with torch.no_grad():
            image = vae.decode(latents).sample  # Decode latents to images
                
        # Process and save the generated images
        image = (image / 2 + 0.5).clamp(0, 1)  # Normalize images
        image=image.detach().cpu().permute(0, 2, 3, 1).numpy()  # Prepare for conversion to PIL format
        images = (image * 255).round().astype("uint8")  # Convert to uint8 format
        pil_images = [Image.fromarray(image) for image in images]  # Convert numpy arrays to PIL images
        image_path = f"{folder_path_test}/{epoch}_{image_generation_path}_{image_name}"  # Define the path for saving the validation image
        pil_images[0].save(image_path)  
            
        logger.info(f"Validation images saved for participant {participant} at epoch {epoch}")
         

if __name__ == "__main__":
    main()
