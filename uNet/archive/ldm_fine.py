# Import Libraries
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
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
from dataset_DD import load_and_preprocess_eeg_data as load_and_preprocess_eeg_data_DD

check_min_version("0.23.0.dev0")

# dataset 0 is all data for omited participant
# dataset 1 is 30 images

dataset_num = 1

case_num = 0 #0 is all, 1 is unet only, 2 is encoder only


# The participant to omit
parser = argparse.ArgumentParser(description='Process the omitted participant.')
parser.add_argument('--omit_participant', type=int, help='The participant to omit')
args = parser.parse_args()
participant = args.omit_participant

checkpoint = 'models/model_2024-04-10_case_0_VC_dataset_all_encoder_final_omit_6.pt'

encoder_name ='encoder_final'
encoder_path = f'temp_encoder/{encoder_name}.pth'

cur_date = datetime.now().strftime("%Y-%m-%d")
log_name = f'log5_{cur_date}_Participant_{args.omit_participant}.log'
image_generation_path = f"validation_image_{cur_date}_Participant_{args.omit_participant}"
model_save_path = f"model__Participant_{args.omit_participant}.pt"


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

device = torch.device("cuda")

def main():
    # Set up the training parameters
    config = Config()

    #log the configuration
    logger.info(f"Configuration: {config}")

    # Load the models
    custom_diffusion_layers,unet, first_stage, vae, frozen_image_embedder, noise_scheduler, optimizer = load_models(config,checkpoint)
    
    # Set up the EEG dataset
    if dataset_num == 0:
        train_dataset, test_dataset, tmp_path1 = load_and_preprocess_eeg_data(eeg_signals_path="data/eeg.pth", split_path="data/splits.pth",participant_number=participant)
    else:
        train_dataset, ___ = load_and_preprocess_eeg_data_DD(eeg_signals_path="data/eeg.pth",participant_number=participant)
        ___, test_dataset, ___ = load_and_preprocess_eeg_data(eeg_signals_path="data/eeg.pth", split_path="data/splits.pth",participant_number=participant)
    

    train_loader, test_loader = create_EEG_dataset(config.train_batch_size,train_dataset, test_dataset)

    
    logger.info(f"Data is {len(train_loader)} batches long")

    # Prepare the training
    custom_diffusion_layers,train_loader,progress_bar,global_step,lr_scheduler,max_train_steps,test_loader, unet, first_stage = prepare_training(custom_diffusion_layers,config,train_loader,test_loader,optimizer,unet,first_stage)
    
    # Train the model
    logger.info(f"Training started with {len(train_loader)} batches")

    train(custom_diffusion_layers,train_loader, unet, first_stage,vae,frozen_image_embedder, optimizer, lr_scheduler, config, global_step, progress_bar,participant, test_loader,noise_scheduler,max_train_steps)
            
    # Clean up the temporary directory
    if tmp_path1 and Path(tmp_path1).exists():
        shutil.rmtree(tmp_path1)
    else:
        logger.info(f"Temporary path {tmp_path1} does not exist")

    train_dataset, test_dataset, train_loader, test_loader = None, None, None, None
    
    
    # Save the final model
    torch.save({
            'first_stage': first_stage.state_dict(),
            'unet': unet.state_dict(),
        }, model_save_path)

    logger.info("Training completed")

    torch.cuda.empty_cache()


def prepare_training(custom_diffusion_layers,config,train_loader,test_loader,optimizer,unet,first_stage):
    # Set up the other training parameters
    num_update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = 0.1 * max_train_steps  # 10% of max_train_steps as an example
    num_training_steps = max_train_steps
    global_step = 0
    progress_bar = tqdm(range(0, max_train_steps),initial=global_step,desc="Steps")

    # Set up the learning rate scheduler
    lr_scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)
        
    return custom_diffusion_layers,train_loader,progress_bar,global_step,lr_scheduler,max_train_steps,test_loader, unet, first_stage
 

def load_models(config,weight_dtype=torch.float32):

    #Load the pretrained models
    noise_scheduler = DDPMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet")
    vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae")

    encoder_config=EncoderConfig()
    encoder = Encoder(time_len=encoder_config.data_len,
        patch_size=encoder_config.patch_size,
        embed_dim=encoder_config.embed_dim,
        decoder_embed_dim=encoder_config.decoder_embed_dim,
        depth=encoder_config.depth,
        num_heads=encoder_config.num_heads,
        decoder_num_heads=encoder_config.decoder_num_heads,
        mlp_ratio=encoder_config.mlp_ratio)
    
    frozen_image_embedder = FrozenImageEmbedder()

    first_stage = First_Stage(encoder=encoder)

    cc = torch.load(checkpoint)
    first_stage.load_state_dict(cc['first_stage'], strict=False)
    unet.load_state_dict(cc['unet'])

    cross_attention_params_to_train = []
    state_dict = unet.state_dict()

    
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

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    frozen_image_embedder.requires_grad_(False)
    first_stage.requires_grad_(False)

    # Move each model to the appropriate device
    vae.to(device)
    unet.to(device)
    frozen_image_embedder.to(device)
    first_stage.to(device)

    attention_class = (CustomDiffusionAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else CustomDiffusionAttnProcessor)
    train_kv = True
    train_q_out = True
    custom_diffusion_attn_procs = {}
    st = unet.state_dict()
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        layer_name = name.split(".processor")[0]
        weights = {
                "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
                "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
        }
        if train_q_out:
            weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
            weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
            weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=train_kv,
                    train_q_out=train_q_out,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(unet.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
        else:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=False,
                train_q_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                )
    del st
    unet.set_attn_processor(custom_diffusion_attn_procs)
    custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)

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
        first_stage.requires_grad_(False)
    elif case_num == 2:
        unet.requires_grad_(False)

    vae.eval() 
    unet.train()
    first_stage.train()

    optimizer_cls = torch.optim.AdamW
    if case_num == 0:
        optimizer = optimizer_cls(itertools.chain(unet.parameters(),first_stage.parameters(),custom_diffusion_layers.parameters()),lr=config.learning_rate,weight_decay=config.weight_decay,eps=config.eps)
    elif case_num == 1:
        optimizer = optimizer_cls(unet.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay,eps=config.eps)
    elif case_num == 2:
        optimizer = optimizer_cls(first_stage.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay,eps=config.eps)
    return custom_diffusion_layers,unet, first_stage, vae, frozen_image_embedder, noise_scheduler, optimizer


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


def train(custom_diffusion_layers,train_loader, unet, first_stage,vae,frozen_image_embedder, optimizer, lr_scheduler, config, global_step, progress_bar,participant, test_loader,noise_scheduler,max_train_steps,weight_dtype=torch.float32):
    clip_loss = True
    
    # Start of training code:
    for epoch in range(0, config.num_train_epochs):
        unet.train()
        first_stage.train()
        train_loss = 0.0
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):           
            latents = vae.encode(batch["pixel_values"].to(device, dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states, ___ = first_stage(batch["eeg"].to(device))

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
            target = noise

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            if clip_loss:
                img_embd = frozen_image_embedder(batch['pixel_values'].to(device))
                clip_loss = first_stage.get_clip(batch['eeg'].to(device), img_embd)
                loss = loss + clip_loss

            avg_loss = loss.mean()
            train_loss += avg_loss.item()

            #backpropagation of the loss
            avg_loss.backward()

            # clip the gradients
            if case_num == 0:
                torch.nn.utils.clip_grad_norm_(itertools.chain(unet.parameters(),first_stage.parameters(),custom_diffusion_layers.parameters()), config.max_grad_norm)                
            elif case_num == 1:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
            elif case_num == 2:
                torch.nn.utils.clip_grad_norm_(first_stage.parameters(), config.max_grad_norm)
        
            # Update the model parameters and learning rate
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
                
            # Update the progress bar and log the training progress
            progress_bar.update(1)
            global_step += 1
            epoch_loss += train_loss

            train_loss = 0.0

            # Update the progress bar with the latest training metrics
        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        logger.info(f"Epoch: {epoch} --> Total Loss {epoch_loss}; Avg Batch Loss {epoch_loss/len(train_loader)}")

        # Check if we need to run validation
        if epoch % config.validation_epochs == 0:
            log_validation(test_loader,participant,unet,vae,first_stage,epoch,noise_scheduler, config)


# Validation function
def log_validation(test_loader,participant,unet,vae,first_stage,epoch,noise_scheduler, config):
        
    # Set the model to evaluation mode to disable dropout and batch normalization effects during inference
    unet.eval()
    first_stage.eval()
    try:
        folder_path_train = f"val/test/"

        
        image_name = str(test_loader.dataset[1]['image_path']).split("/")[-1]
        sample = test_loader.dataset[1]['eeg'].to(device)
        sample = sample.unsqueeze(0)
        embedding, __ = first_stage(sample)
        generator = torch.manual_seed(42)
                
        # Initialize unconditional embeddings as zeros, which will be used for guided diffusion
        uncond_embeddings = torch.zeros_like(embedding)
        #embedding = embedding.unsqueeze(0)
                
        # Combine unconditional and conditional embeddings for guided diffusion
        eeg_embeddings = torch.cat([uncond_embeddings, embedding]).to(device)
                
        # Initialize latent vectors with random noise, shaping them for the image generation process
        latents = torch.randn((config.images_to_produce, unet.config.in_channels, config.height // 8, config.width // 8), generator=generator).to(device)
                
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
            
        logger.info(f"Validation images saved for participant {participant} at epoch {epoch}")
    except Exception as e:
        logger.info(f"Error in validation: {e}")
        

if __name__ == "__main__":
    main()
