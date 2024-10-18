import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
#import temp_encoder.utils as ut
import utils as ut
import torch

class PatchEmbed1D(nn.Module):
    """
    A class for converting 1D data into patch embeddings.
    
    It divides the data into patches and projects each patch into an embedding space.
    
    Parameters:
    - time_len: The length of the time series data.
    - patch_size: The size of each patch.
    - in_chans: Number of input channels (features per time point).
    - embed_dim: The dimensionality of the output embedding space.
    """
    
    def __init__(self, time_len=224, patch_size=1, in_chans=128, embed_dim=256):
        # Initialize the parent class (nn.Module)
        super().__init__()
        
        # Calculate the number of patches by dividing the total length by the patch size
        num_patches = time_len // patch_size
        
        # Initialize attributes
        self.patch_size = patch_size
        self.time_len = time_len
        self.num_patches = num_patches

        # Define a convolutional layer to project the input data into the embedding space
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass of the module.
        
        Parameters:
        - x: The input data of shape (Batch size, Channels, Length of the data)
        
        Returns:
        - The patch embeddings of the input data.
        """
        # Ensure x is in the correct shape: (Batch size, Channels, Data length)
        B, C, V = x.shape
        
        # Project the input data into the embedding space and reshape
        # The 'proj' layer outputs (Batch size, embed_dim, Num patches)
        # We transpose to make it (Batch size, Num patches, embed_dim) for further processing
        x = self.proj(x).transpose(1, 2).contiguous()
        
        return x

class MaskedAutoencoder(nn.Module):
    """
    A Masked Autoencoder for 1D data (e.g., time series), using a transformer-based architecture.
    
    This model is designed to encode 1D input data into a lower-dimensional space and then decode 
    it back to its original dimension, with a focus on reconstructing the original data from 
    partial (masked) inputs. It features a Vision Transformer (ViT) backbone for both encoding and 
    decoding processes.
    
    Parameters:
    - time_len: Length of the input time series.
    - patch_size: Size of each patch into which the input data is divided.
    - embed_dim: Dimensionality of the embedding space for the encoder.
    - in_chans: Number of input channels.
    - Various parameters for configuring the depth and heads of the transformer model, along with other hyperparameters.
    """
    
    def __init__(self, time_len=512, patch_size=4, embed_dim=1024, in_chans=128,
                 depth=24, num_heads=16, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        # Initialize the encoder part of the MAE
        # This involves embedding the input patches and applying transformer blocks to them
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)
        num_patches = int(time_len / patch_size)
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Initialize the decoder part of the MAE
        # It decodes the encoded features back to the original data dimensionality
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        #self.decoder = MaskedAutoDecoder()
        #self.initialize_weights()

        self.conv1 = nn.Conv1d(128, 1, kernel_size=1)
        self.linear1 = nn.Linear(1024, 768)


    
    def dim_mapper(self, x):
        x = self.conv1(x)
        x = x.squeeze(1)
        x = self.linear1(x)
        x = x.unsqueeze(1)
        return x


    def forward_encoder(self, x):
        
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]
    
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)

        return x
    
    def forward_loss(self, x, img_embs):

        mse_loss = nn.MSELoss()
        img_embs = img_embs.unsqueeze(1)

        loss = mse_loss(x, img_embs)

        loss += (1 - nn.functional.cosine_similarity(x, img_embs, dim=-1).mean())

        print(loss.item())

        return loss

    def forward(self, eeg_input, img_embs):
        
        latent = self.forward_encoder(eeg_input)

        pred = self.dim_mapper(latent)

        print(pred.shape)
        
        loss  = self.forward_loss(pred, img_embs)
                        
        return pred, loss 


