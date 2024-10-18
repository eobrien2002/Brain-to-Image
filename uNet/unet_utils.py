import torch
from torch import nn as nn
from transformers import CLIPVisionModelWithProjection
import math
from torch.cuda.amp import GradScaler 

                  

class Dim_Mapper(nn.Module):
    def __init__(self):
        super().__init__()
             
        self.conv1 = nn.Conv1d(128,1, kernel_size=1)
        
        self.fc1 = nn.Linear(1024,768) 

    def forward(self, x):
        # Apply a convolution operation to the input

        x = self.conv1(x)

        # Remove unnecessary dimension after convolution
#        x = x.squeeze(1)

        # Apply a linear transformation
        x = self.fc1(x)
        return x
    
# The point of having a seperate Dim_Mapper class is so that we can swap out using and not using clip alignmnet 
class First_Stage(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder        
        self.seq_len = encoder.num_patches 
        self.input_dim = 1024
        self.output_dim = 768
#        self.input_dim = 512
        
        # Dimensionality mapper for adjusting feature vector sizes
        self.mapper = Dim_Mapper()
       
        # Unet expects (batch, 1, 768)
        # Encoder outputs (batch, sequence_length, feature_dim) -> (batch, 33, 1024)
        # Aligned Encoder is (batch, 1, 512)

 #       self.conv_block = nn.Sequential(
      #      nn.Conv1d(1,1, kernel_size=1),
     #   )

        self.conv_block = nn.Sequential(
             nn.Conv1d(self.seq_len, self.seq_len // 2, 1, bias=True),
             nn.Conv1d(self.seq_len // 2, 77, 1, bias=True) #Before used to conv to 77
         )


        self.fc = nn.Linear(in_features=self.input_dim , out_features=self.output_dim)
    
    def forward(self, x):
        # Encode the input using the encoder model
        x = self.encoder(x)

        latent = x  # Store the encoder output for potential use
       
       
        x = self.conv_block(x)
        
        x = self.fc(x)

        return x, latent
    
    def get_clip(self, x, image_embeddings):
        # Map the input dimensions to align with the image embeddings
        x = self.encoder(x)
      
        x = self.mapper(x)

        # Calculate the CLIP loss by comparing the cosine similarity between the mapped input and image embeddings
        loss = 1 - nn.functional.cosine_similarity(x, image_embeddings, dim=-1).mean()
        return loss

# Abstract class to enforce the implementation of FrozenImageEmbedder
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenImageEmbedder(AbstractEncoder):
    
    def __init__(self):
        super().__init__()
        # Load the pretrained CLIP model for image processing
        self.transformer = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        self.freeze()  # Freeze the model weights to prevent updates during training

    def freeze(self):
        # Set the model to evaluation mode and disable gradient updates
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        # Process inputs through the CLIP model to obtain image embeddings
        outputs = self.transformer(inputs)
        image_embeds = outputs.image_embeds
        return image_embeds

    def encode(self, inputs):
        # Wrapper for the forward method to align with the AbstractEncoder interface
        return self(inputs)

def adjust_learning_rate(optimizer, current_epoch, config):
    """
    Adjusts the learning rate based on the current epoch, following a schedule that initially increases the learning rate linearly during a warmup phase, then decays it according to a half-cycle cosine formula.

    Args:
    - optimizer: The optimizer for which the learning rate needs to be adjusted.
    - current_epoch: The current epoch number during training.
    - config: A configuration object containing settings for the learning rate schedule, including the initial learning rate (`lr`), the minimum learning rate (`min_lr`), the total number of epochs (`num_epoch`), and the number of warmup epochs (`warmup_epochs`).

    Returns:
    - float: The adjusted learning rate.
    """
    # During the warmup phase, increase the learning rate linearly
    if current_epoch < config.warmup_epochs:
        lr = config.lr * current_epoch / config.warmup_epochs
    else:
        # After warmup, apply a half-cycle cosine decay to the learning rate
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * (
            1. + math.cos(
                math.pi * (current_epoch - config.warmup_epochs) / (config.num_epoch - config.warmup_epochs)
            )
        )

    # Apply the calculated learning rate to all parameter groups in the optimizer
    for param_group in optimizer.param_groups:
        # Adjust the learning rate, considering an optional scaling factor if present
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

    return lr


def get_grad_norm(parameters, norm_type=2.0):
    """
    Calculate the norm of gradients for a list of parameters.
    Args:
    - parameters (iterable): An iterable of Parameters.
    - norm_type (float): Type of norm to use. Defaults to L2 norm.
    Returns:
    - torch.Tensor: The total norm of the gradients.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]  # Convert a single Tensor to a list for consistency
    # Filter out parameters without gradients
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    # Determine device from the first parameter
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        # For infinity norm, find the max abs value among all gradients
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # For L2 norm (or other), calculate norm across all parameters
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class GradScalerWithClip:
    """
    A wrapper around `torch.cuda.amp.GradScaler` that adds gradient clipping.
    """
    def __init__(self):
        self.scaler = GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        """
        Scales the loss, computes gradients, optionally clips them, and updates model parameters.

        Args:
        - loss (Tensor): The loss tensor.
        - optimizer (Optimizer): The optimizer.
        - clip_grad (float, optional): Max norm of the gradients.
        - parameters (iterable, optional): Iterable of parameters to clip.
        - create_graph (bool): Whether to create a computational graph for second order gradients.
        - update_grad (bool): Whether to update model parameters.

        Returns:
        - torch.Tensor or None: The norm of the gradients if `clip_grad` is not None; otherwise None.
        """
        # Backward pass with scaled loss
        self.scaler.scale(loss).backward(create_graph=create_graph)

        norm = None  # Default gradient norm is None
        if update_grad:
            # If gradient clipping is enabled, unscale and clip gradients
            if clip_grad is not None and parameters is not None:
                # Unscale gradients before clipping
                self.scaler.unscale_(optimizer)
                # Clip gradients and get their norm
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self.scaler.unscale_(optimizer)
                # Calculate gradient norm without clipping
                if parameters is not None:
                    norm = get_grad_norm(parameters)

            # Step the optimizer and update the scaler
            self.scaler.step(optimizer)
            self.scaler.update()

        return norm

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict):
        self.scaler.load_state_dict(state_dict)
