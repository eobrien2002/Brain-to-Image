import torch
from torch import nn as nn
from transformers import CLIPVisionModelWithProjection
                     

class Dim_Mapper(nn.Module):
    def __init__(self):
        super().__init__()
             
        self.conv1 = nn.Conv1d(1,1, kernel_size=1)
        
        self.fc1 = nn.Linear(512, 768)  

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
        #self.input_dim = 1024
        self.output_dim = 768
        self.input_dim = 512
        
        # Dimensionality mapper for adjusting feature vector sizes
        self.mapper = Dim_Mapper()
       
        # Unet expects (batch, 1, 768)
        # Encoder outputs (batch, sequence_length, feature_dim) -> (batch, 33, 1024)
        # Aligned Encoder is (batch, 1, 512)

        self.conv_block = nn.Sequential(
            nn.Conv1d(1,1, kernel_size=1),
        )

        # self.conv_block = nn.Sequential(
        #     nn.Conv1d(self.seq_len, self.seq_len // 2, 1, bias=True),
        #     nn.Conv1d(self.seq_len // 2, 77, 1, bias=True)
        # )


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
