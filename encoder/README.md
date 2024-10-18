# Masked Variable AutoEncoder with Vision Transformer (ViT) Backbone for EEG-Based Image Reconstruction

## Introduction
This project implements a **Masked Variable AutoEncoder (VAE)** utilizing a **Vision Transformer (ViT)** backbone to reconstruct images from EEG data. By leveraging advanced transformer architectures and masked autoencoding techniques, the model learns rich representations of EEG signals, enabling high-quality downstream image reconstruction.

## Features
- **Masked Variable AutoEncoder**: Efficiently learns to reconstruct EEG signals by masking parts of the input.
- **Vision Transformer Backbone**: Utilizes transformer-based architectures for robust feature extraction.
- **Patching Mechanism**: Divides EEG data into patches to facilitate transformer processing.
- **Distributed Training**: Supports multi-node and multi-GPU training for scalability.

## Architecture Overview

### Masked Variable AutoEncoder (VAE)
The **Masked Variable AutoEncoder (VAE)** is designed to learn compact and meaningful representations of EEG signals. It operates by randomly masking portions of the input data and training the encoder to reconstruct the original signal from the unmasked parts. This approach encourages the model to capture the underlying structure and dependencies within the EEG data.

#### Key Components:
- **Encoder**: Compresses the input EEG data into a latent representation.
- **Decoder**: Reconstructs the EEG signal from the latent representation.
- **Masking Strategy**: Randomly masks patches of the input to prevent the model from relying on any single part of the data.

### Vision Transformer (ViT) Backbone
The **Vision Transformer (ViT)** backbone serves as the core feature extractor in the autoencoder. Transformers have shown  success in capturing long-range dependencies and complex patterns in data, making them ideal for processing EEG signals.

### Patching Mechanism
Patching is a crucial aspect of the model, enabling the transformer to process EEG data effectively. The input EEG signals are divided into smaller, manageable patches, which are then embedded and fed into the transformer layers.

#### Process:
1. **Patch Division**: The EEG signal is split into non-overlapping segments (patches) along the temporal axis.
2. **Embedding**: Each patch is linearly projected into a high-dimensional embedding space.
3. **Positional Encoding**: Positional information is added to each patch to retain the order of the sequence.
4. **Transformer Processing**: The embedded patches are processed through multiple transformer blocks to capture dependencies and extract features.

This patch-based approach allows the model to handle long EEG sequences by breaking them down into smaller, more manageable pieces, facilitating efficient computation and learning.

## Data Preparation

### Datasets Used:
- **MOABB EEG Dataset**: A diverse collection of EEG data sourced from the MOABB (Mother of All BCI Benchmarks) repository.
- **THINGS EEG Dataset**: EEG recordings from participants performing tasks in the THINGS dataset.
- **VC EEG Dataset**: A specialized dataset used for fine-tuning the model to align with specific amplitude and frequency characteristics.

### Data Loading and Processing
Data is loaded, batched, and preprocessed using dedicated data loaders and processors to ensure compatibility with the training pipeline.

- **`moabb_dataloader.py`**: Handles loading and preprocessing of MOABB EEG data.
- **`dataset.py`**: Processes EEG data from both MOABB and THINGS datasets, performing tasks like resampling, normalization, and epoch extraction.
- **`dataset_finetune_encoder.py`**: Prepares the VC EEG data for fine-tuning, aligning it with the pretrained model's representations.

The configuration for dataset paths and parameters is managed through the `config.py` file.

## Training Process

### Pretraining on MOABB and THINGS EEG Datasets
1. **Data Preparation**: EEG data from the MOABB repository and selected participants from the THINGS EEG dataset are loaded and preprocessed.
2. **Model Initialization**: The Masked VAE with a ViT backbone is initialized with specified hyperparameters.
3. **Masking Strategy**: Random patches of the EEG data are masked to train the model to reconstruct the missing information.
4. **Training Loop**: The model is trained to minimize the reconstruction loss, effectively learning rich representations of the EEG signals.
5. **Distributed Training**: Training is performed on 2 nodes with 4 GPUs per node using Compute Canada’s SLURM scheduler, leveraging distributed data parallelism for efficiency.

#### Key File:
- **`trainer_moabb.py`**: Orchestrates the training process, including data loading, model training, loss computation, and checkpointing.

### Fine-Tuning on VC EEG Data
After pretraining, the model is fine-tuned on the VC EEG dataset to adapt the learned representations to the specific amplitude and frequency characteristics of the downstream task.

#### Steps:
1. **Load Pretrained Weights**: Utilize the weights from the pretrained Masked VAE.
2. **Data Alignment**: Fine-tune the model on the VC EEG data to align the signal characteristics.
3. **Quick Fine-Tuning**: Perform a swift adaptation to ensure the model effectively captures the nuances of the VC EEG dataset.

#### Key File:
- **`dataset_finetune_encoder.py`**: Manages the loading and preprocessing of the VC EEG data for fine-tuning.

## Results
The final encoder demonstrated strong performance in reconstructing EEG signals, as evidenced by the correlation coefficients and reconstruction plots. Detailed reconstruction results can be visualized through the provided plots, showcasing the model's ability to accurately reconstruct masked EEG data.
![reconst-10-04-2024-15-05-58](https://github.com/user-attachments/assets/15bc9577-4bca-462e-97a2-5d6ff47077b0)
![reconst-10-04-2024-15-38-27](https://github.com/user-attachments/assets/81465808-635d-4b04-ba3a-7125531cf60f)
![reconst-10-04-2024-14-33-27](https://github.com/user-attachments/assets/eaa49721-71d4-458b-a783-cc32ebbafff7)





