import os 
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from PIL import Image
from temp_encoder.enc_config import Config as EncoderConfig
from temp_encoder.encoder import Encoder
from unet_utils import First_Stage
from dataset import load_and_preprocess_eeg_data
from config import Config
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionAttnProcessor2_0
)
import torch.nn.functional as F
config = Config()

ommitted_participant = 2

model_name = "model_2024-04-10_case_0_VC_dataset_all_encoder_final_omit_6"
model_path = f"models/{model_name}.pt"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

first_stage = First_Stage(encoder=encoder)

cc = torch.load(model_path, map_location=device)
first_stage.load_state_dict(cc['first_stage'])
unet.load_state_dict(cc['unet'])

first_stage.to(device)
unet.to(device)
vae.to(device)

unet.eval()
vae.eval()
first_stage.eval()

folder_path = f"generations/{model_name}"
os.makedirs(folder_path, exist_ok=True)

train_dataset, test_dataset, tmp_path1 = load_and_preprocess_eeg_data(eeg_signals_path="data/eeg.pth", split_path="data/splits.pth",participant_number=4)

def collate_fn(examples):
    examples = [example for example in examples if example is not None]
    pixel_values = torch.stack([example['image_raw']for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["eeg"] for example in examples])
    return {"pixel_values": pixel_values, "eeg": input_ids}

# Create the data loaders
#test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

for i in range(len(train_dataset)):
   try:    

        ground_truth_path = test_loader.dataset[i]['image_path']
        ground_truth_image = Image.open(ground_truth_path)

        image_name = str(test_loader.dataset[i]['image_path']).split("/")[-1]
        sample = test_loader.dataset[i]['eeg'].to(device)
        sample = sample.unsqueeze(0)
        embedding, __ = first_stage(sample)
        print(embedding.shape)
        generator = torch.manual_seed(42)
        uncond_embeddings = torch.zeros_like(embedding)
        eeg_embeddings = torch.cat([uncond_embeddings, embedding]).to(device)
        latents = torch.randn((config.images_to_produce, unet.config.in_channels, config.height // 8, config.width // 8), generator=generator).to(device)
                    
        noise_scheduler.set_timesteps(config.num_inference_steps)
        latents = latents * noise_scheduler.init_noise_sigma
        print(latents.shape)                    
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
        image_path = f"{folder_path}/{image_name}"  # Define the path for saving the validation image
        # Create a folder to save the generated images
        os.makedirs(image_path, exist_ok=True)
        for i, pil_image in enumerate(pil_images):
            pil_image.save(f"{image_path}/AI.png")

        #save the ground truth image
        ground_truth_image.save(f"{image_path}/ground_truth.png") 
        print(image_path)       
       # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
       # # Show generated image
       # axs[0].imshow(pil_images[0])
       # axs[0].set_title(f"AI Generated")
      #  axs[0].axis('off')  # Hide axes ticks
        
        # Show ground truth image
     #   axs[1].imshow(ground_truth_image)
    #    axs[1].set_title(f"Ground Truth")
   #     axs[1].axis('off')  # Hide axes ticks 
  #      plt.tight_layout()
 #       plt.savefig(image_path)
#        plt.close()


   except Exception as e:
        print(e)
        continue    
