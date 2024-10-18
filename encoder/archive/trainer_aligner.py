import math
import sys
import torch
import numpy as np
import utils as ut
import datetime
import os
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import timm.optim.optim_factory as optim_factory
import matplotlib.pyplot as plt
import logging
import zipfile
import tempfile
from transformers import AutoProcessor
from pathlib import Path
from PIL import Image
import pandas as pd

from encoder_aligned_images import MaskedAutoencoder
from utils import GradScalerWithClip as NativeScaler
from utils import save_training_checkpoint
from things_dataloader_align import create_dataset
from config import Config
#from enc_config import Config
from unet_utils import FrozenImageEmbedder

check_path = 'encoder_final.pth'
date = datetime.datetime.now().strftime("%d-%m-%Y")
checkpoint_save_path = f'checkpoints/checkpoint_{date}.pth'


tmp_path = tempfile.mkdtemp()
img_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
frozen_image_embedder = FrozenImageEmbedder()
with zipfile.ZipFile('data/things_stimuli.zip', 'r') as image_zip:
    image_zip.extractall(tmp_path)  # Extract images

# Set up logging
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)  # Capture all logs

# Configure file handler for INFO and DEBUG logs
info_handler = logging.FileHandler(f'log_April02.log')
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

def train_one_epoch(model, data_loader, optimizer, device, epoch, grad_scaler, config=None, model_without_ddp=None):
    """
    Trains the model for one epoch through all batches in the data_loader.

    Args:
    - model: The model to be trained.
    - data_loader: DataLoader providing batches of data.
    - optimizer: Optimizer used for training.
    - device: The device to run the training on.
    - epoch: Current epoch number.
    - grad_scaler: Gradient scaler for mixed precision training.
    - log_writer: Logger for training metrics (optional).
    - config: Configuration object containing training settings.
    - start_time: Timestamp marking the start of training (optional).
    - model_without_ddp: Model without Distributed Data Parallel wrapper (optional).
    
    Returns:
    - Mean correlation coefficient across all batches in the epoch.
    """
    model.train()
    total_loss = []
    accum_iter = config.accum_iter  # Gradient accumulation steps

    for step, batch in enumerate(data_loader):
        if step % accum_iter == 0:
            # Adjust learning rate per iteration, not per epoch
            ut.adjust_learning_rate(optimizer, step / len(data_loader) + epoch, config)

        
        image_path_indexs = batch['image_path_index']
        participant_ids = batch['participant_id']
        imgs_pixel_values=[]
        eeg_data = []
        for i in range(len(image_path_indexs)):
            try:
                participant_id = participant_ids[i]
                index = int(image_path_indexs[i])
                if participant_id < 10:
                        event_path = f'data/sub-0{participant_id}/eeg/sub-0{participant_id}_task-rsvp_events.csv'
                else:
                        event_path = f'data/sub-{participant_id}/eeg/sub-{participant_id}_task-rsvp_events.csv'

                events = pd.read_csv(event_path)
                img_path = events.iloc[index]['stim'].replace('\\', '/')
                full_path = Path(tmp_path) / 'things_stimuli' / img_path
                image = Image.open(full_path).convert('RGB')
                image_raw = img_processor(images=image, return_tensors="pt")
                image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)
                imgs_pixel_values.append(image_raw['pixel_values'])
                eeg_data.append(batch['eeg'][i])
            except Exception as e:
                continue


        imgs_pixel_values = torch.stack(imgs_pixel_values).to(device)
        eeg_data = torch.stack(eeg_data).to(device)
        new_batch = {'imgs_pixel_values': imgs_pixel_values, 'eeg': eeg_data}

        samples = new_batch['eeg'].to(device).float()
        img_embds = frozen_image_embedder(new_batch['imgs_pixel_values'].to(device))


        # Perform forward pass and loss computation
        with torch.cuda.amp.autocast(enabled=True):
            pred, loss  = model(samples, img_embds)

        
        # Scale loss and backpropagate
        grad_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training. Step: {step}, Epoch: {epoch}")
            sys.exit(1)

        total_loss.append(loss_value)


        if device == torch.device('cuda:0'):
            if step % config.log_steps ==0:
                logger.info(f"Step loss: {np.mean(total_loss)}, LR: {optimizer.param_groups[0]['lr']}")
    if device == torch.device('cuda:0'): 
        logger.info(f'[Epoch {epoch}] Average loss: {np.mean(total_loss)}')

    return np.mean(total_loss)


def main(config):

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')

    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)


    model = MaskedAutoencoder(
        time_len=config.data_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        decoder_embed_dim=config.decoder_embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
    )

    if check_path is not None:
        model.load_state_dict(torch.load(check_path, map_location=device)['model'], strict=False)
        logger.info(f"Loaded model from {check_path}")

    params_to_train = []
    model.requires_grad_(True)
    # for name, param in model.named_parameters():
    #     if not name.startswith('decoder.'):
    #         param.requires_grad = False
    #     if name == 'decoder.mask_token':
    #         param.requires_grad = False
    #     if name == 'decoder.cls_token':
    #         param.requires_grad = False
    for name, param in model.named_parameters():
        if 'decoder' in name:
            param.requires_grad = False
        if name =='cls_token' or name == 'mask_token':
            param.requires_grad = False


    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_train.append(name)
    print(params_to_train)
                
    model_without_ddp = model
    model.to(device)
    frozen_image_embedder.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank)

    param_groups = optim_factory.param_groups_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()

    dataset_pretrain = []
    for path in config.par_number:
        if path<10:
            file_path = f'data/sub-0{path}/eeg/sub-0{path}_task-rsvp_eeg.vhdr'
        else:
            file_path = f'data/sub-{path}/eeg/sub-{path}_task-rsvp_eeg.vhdr'
       
        dataset = create_dataset(raw_file_path = file_path, event_description = 'Event/E  1', participant_id = path)
        dataset_pretrain.append(dataset)
        logger.info(f"Loaded dataset for participant {path}")
    dataset_pretrain = torch.utils.data.ConcatDataset(dataset_pretrain)
    sampler = DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)


    dataloader_eeg = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler, shuffle=sampler is None, pin_memory=True, collate_fn=collate_fn)
    loss_list = []
    for ep in range(config.num_epoch):
        if torch.cuda.device_count() > 1:
            sampler.set_epoch(ep)  # Shuffle data at every epoch for distributed training

        loss = train_one_epoch(model, dataloader_eeg, optimizer, device, ep, loss_scaler, config, model_without_ddp)
        loss_list.append(loss)

                # Save checkpoint and plot reconstruction figures periodically
        if ep % 20 == 0 and config.local_rank == 0:
            save_training_checkpoint(config, ep, model_without_ddp, optimizer, loss_scaler, config.save_dir, checkpoint_save_path)
      

if __name__ == '__main__':
    local_rank = int(os.environ['LOCAL_RANK'])
    config = Config()
    config.local_rank = local_rank
    main(config)
