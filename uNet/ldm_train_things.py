# Import Libraries
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
from PIL import Image
import logging
import argparse
import itertools
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoProcessor
import zipfile

from diffusers.utils import check_min_version


# Import our custom modules
from dataset import load_and_preprocess_eeg_data
from things_dataloader import create_dataset
from temp_encoder.encoder import Encoder
from temp_encoder.enc_config import Config as EncoderConfig
#from encoder import Encoder
#from enc_config import Config as EncoderConfig
from config import Config
from unet_utils import First_Stage, FrozenImageEmbedder, adjust_learning_rate

check_min_version("0.23.0.dev0")

# The participant to omit
#parser = argparse.ArgumentParser(description='Process the omitted participant.')
#parser.add_argument('--omit_participant', type=int, help='The participant to omit', default=6)
#args = parser.parse_args()
participants = [i for i in range(1, 7) if i != 6]
#participants_things = [2, 3, 4, 5, 8, 9]
#participants_things = [ 11, 12, 13, 47, 15, 16]
#participants_things = [17, 19, 21, 25, 30]
#participants_things = [ 31, 32, 34, 35, 38]
participants_things = [ 3, 13] #Run this one for 100 epoch on 16 batch size

train_only_attn2_layers = True #If false it trains attn1 and attn2

# Case 0 is train both the unet and the encoder
# Case 1 is train the unet and freeze the encoder
# Case 2 is freeze the unet and train the encoder
case_num = 0

# Dataset 0 is the VC dataset
# Dataset 1 is the Things dataset
# Dataset 2 is all particpants for VC dataset except one
dataset_num = 1

encoder_name ='encoder_final'
encoder_path = f'temp_encoder/{encoder_name}.pth'
#encoder_path = f'{encoder_name}.pth'


checkpoint = 'model_iter4.pt'

switch_dataset = {
    0: "VC_dataset",
    1: "THINGS_dataset",
    2: "VC_dataset_all"
}

cur_date = datetime.now().strftime("%Y-%m-%d")

if dataset_num == 1:
    log_name = f'log_{cur_date}_{switch_dataset[dataset_num]}.log'
    model_save_path = f"model_{cur_date}_case_{case_num}_{switch_dataset[dataset_num]}_{encoder_name}.pt"
    image_generation_path = f"validation_image_{cur_date}"

else:
    log_name = f'log_{cur_date}_omit_{6}.log'
    model_save_path = f"model_{cur_date}_case_{case_num}_{switch_dataset[dataset_num]}_{encoder_name}_omit_{6}.pt"
    image_generation_path = f"validation_image_{cur_date}__omit_{6}"


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
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')

    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)


    # Set up the EEG dataset
    if dataset_num == 0:  
        train_dataset, test_dataset, tmp_path1 = load_and_preprocess_eeg_data()
    elif dataset_num == 1:
        all_data_train = []
        for i in participants_things:
            if i<10:
                train_dataset,tmp_path = create_dataset(f"data/sub-0{i}/eeg/sub-0{i}_task-rsvp_eeg.vhdr",'Event/E  1',  f"data/sub-0{i}/eeg/sub-0{i}_task-rsvp_events.csv")
            else:
                train_dataset, tmp_path = create_dataset(f"data/sub-{i}/eeg/sub-{i}_task-rsvp_eeg.vhdr",'Event/E  1',  f"data/sub-{i}/eeg/sub-{i}_task-rsvp_events.csv")
            all_data_train.append(train_dataset)
        #test_dataset, tmp_path2 = create_dataset(f"data/sub-{things_test}/eeg/sub-{things_test}_task-rsvp_eeg.vhdr",'Event/E  1',  f"data/sub-{things_test}/eeg/sub-{things_test}_task-rsvp_events.csv")
        test_dataset = train_dataset
        train_dataset = torch.utils.data.ConcatDataset(all_data_train)
    elif dataset_num == 2:
        all_data_train = []
        for i in participants:
            train_dataset, test_dataset, tmp_path1 = load_and_preprocess_eeg_data(eeg_signals_path="data/eeg.pth", split_path="data/splits.pth",participant_number=i)
            all_data_train.append(train_dataset)
        ___, test_dataset, __ = load_and_preprocess_eeg_data(eeg_signals_path="data/eeg.pth", split_path="data/splits.pth",participant_number=6)
        train_dataset = torch.utils.data.ConcatDataset(all_data_train)
            
    if dataset_num == 1:
        with zipfile.ZipFile('data/things_stimuli.zip', 'r') as image_zip:
            image_zip.extractall(tmp_path)  # Extract images

        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

        def collate_fn(examples):
            examples = [example for example in examples if example is not None]
            if not examples:
                return None
            input_ids = torch.stack([example["eeg"] for example in examples])
            imgs = [processor(images=Image.open(example['image_path']).convert('RGB'), return_tensors="pt") for example in examples]
            pixel_values = torch.stack([img['pixel_values'].squeeze(0) for img in imgs])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return {"pixel_values": pixel_values, "eeg": input_ids}  
    else:
        def collate_fn(examples):
            examples = [example for example in examples if example is not None]
            pixel_values = torch.stack([example['image_raw']for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["eeg"] for example in examples])
            return {"pixel_values": pixel_values, "eeg": input_ids}
    
    sampler = DistributedSampler(train_dataset, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, sampler=sampler, shuffle=sampler is None, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.train_batch_size, sampler=sampler, shuffle=sampler is None, pin_memory=True, collate_fn=collate_fn)
    
    # Load the models
    unet, first_stage, vae, frozen_image_embedder, noise_scheduler, optimizer = load_models(config,device,checkpoint)
    if torch.cuda.device_count() > 1:
        unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)
        unet = DistributedDataParallel(unet, device_ids=[config.local_rank], output_device=config.local_rank)
        first_stage = torch.nn.SyncBatchNorm.convert_sync_batchnorm(first_stage)
        first_stage = DistributedDataParallel(first_stage, device_ids=[config.local_rank], output_device=config.local_rank)


    for ep in range(config.num_epoch):
        if torch.cuda.device_count() > 1:
            sampler.set_epoch(ep)  # Shuffle data at every epoch for distributed training

        train_one_epoch(ep, train_loader,device, unet, first_stage,vae,frozen_image_embedder, optimizer, config,noise_scheduler)

        # Val
        if ep % config.validation_epochs == 0 and config.local_rank == 0:
            log_validation(test_loader,unet,vae,first_stage,ep,device,noise_scheduler, config,train_loader)


    # Save the final model
    if config.local_rank == 0:
        torch.save({
            'first_stage': first_stage.state_dict(),
            'unet': unet.state_dict(),
        }, model_save_path)

        logger.info("Training completed")

    torch.cuda.empty_cache()


def train_one_epoch(epoch, train_loader,device, unet, first_stage,vae,frozen_image_embedder, optimizer, config, noise_scheduler):    
    
    unet.train()
    first_stage.train()
    optimizer.zero_grad()
    total_loss = []
    for step, batch in enumerate(train_loader):
        if batch is None:
            continue
        if step % config.gradient_accumulation_steps:
            adjust_learning_rate(optimizer, step / len(train_loader) + epoch, config)
            
        latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float32)).latent_dist.sample()
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

        
        img_embd = frozen_image_embedder(batch['pixel_values'].to(device))
        clip_loss = first_stage.module.get_clip(batch['eeg'].to(device), img_embd)
        #clip_loss = first_stage.get_clip(batch['eeg'], img_embd)
        loss = loss + clip_loss

        loss = loss.mean()
        loss.backward()

        if case_num == 0:
            torch.nn.utils.clip_grad_norm_(itertools.chain(unet.parameters(),first_stage.parameters()), config.max_grad_norm)                
        elif case_num == 1:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
        elif case_num == 2:
                torch.nn.utils.clip_grad_norm_(first_stage.parameters(), config.max_grad_norm)
        optimizer.step()

        
        loss_value = loss.item()
        total_loss.append(loss_value)

#        if device == torch.device('cuda:0'):
 #           if step % config.log_steps ==0:
  #              logger.info(f"Step loss: {np.mean(total_loss)}, LR: {optimizer.param_groups[0]['lr']}")
    if device == torch.device('cuda:0'): 
        logger.info(f'[Epoch {epoch}] Average loss: {np.mean(total_loss)}')
                
                

def load_models(config,device,weight_dtype=torch.float32):

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
    cc = torch.load(encoder_path, map_location=device)
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
    unet.to(device)
    frozen_image_embedder.to(device)
    vae.to(device)
    vae.eval() 
    unet.train()

    first_stage = First_Stage(encoder=encoder)
    first_stage.to(device)
    first_stage.train()

    if checkpoint is not None:
        cc = torch.load(checkpoint,map_location=device)
        first_stage.load_state_dict(cc['first_stage'], strict=False)
        unet.load_state_dict(cc['unet'], strict=False)
        logger.info(f'models loaded from {checkpoint}')

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
    
    logger.info(f"Parameters to train: {params_to_train}")

    optimizer_cls = torch.optim.AdamW

    if case_num == 0:
        optimizer = optimizer_cls(itertools.chain(first_stage.parameters(), unet.parameters()),lr=config.lr,weight_decay=config.weight_decay,eps=config.eps)
    elif case_num == 1:
        optimizer = optimizer_cls(unet.parameters(),lr=config.lr,weight_decay=config.weight_decay,eps=config.eps)
    elif case_num == 2:
        optimizer = optimizer_cls(first_stage.parameters(),lr=config.lr,weight_decay=config.weight_decay,eps=config.eps)


    return unet,first_stage, vae, frozen_image_embedder, noise_scheduler, optimizer
# Validation function
def log_validation(test_loader,unet,vae,first_stage,epoch,device,noise_scheduler, config,train_loader):
        
    # Set the model to evaluation mode to disable dropout and batch normalization effects during inference
    unet.eval()
    first_stage.eval()
    try:
        folder_path_train = f"val/train/"
        folder_path_test = f"val/test/" 

        image_name = str(train_loader.dataset[1]['image_path']).split("/")[-1]
        sample = train_loader.dataset[1]['eeg'].to(device)
        sample = sample.unsqueeze(0)
        embedding, __ = first_stage(sample)
        generator = torch.manual_seed(42)
                
        # Initialize unconditional embeddings as zeros, which will be used for guided diffusion
        uncond_embeddings = torch.zeros_like(embedding)
        #embedding = embedding.unsqueeze(0)
                
        # Combine unconditional and conditional embeddings for guided diffusion
        eeg_embeddings = torch.cat([uncond_embeddings, embedding]).to(device)
                
        # Initialize latent vectors with random noise, shaping them for the image generation process
        latents = torch.randn((config.images_to_produce, unet.module.config.in_channels, config.height // 8, config.width // 8), generator=generator).to(device)
                
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
        latents = torch.randn((config.images_to_produce, unet.module.config.in_channels, config.height // 8, config.width // 8), generator=generator).to(device)
                
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
            
        logger.info(f"Validation images saved at epoch {epoch}")
    except Exception as e:
        logger.info(f"Error in validation: {e}")
        

if __name__ == "__main__":
    config = Config()
    local_rank = int(os.environ['LOCAL_RANK'])
    config.local_rank = local_rank

    main()
