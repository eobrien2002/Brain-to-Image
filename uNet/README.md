EEG-Conditioned Image Reconstruction with Conditional uNet and Vision Transformer Encoder
Table of Contents
Introduction
Features
Architecture Overview
Conditional uNet
Vision Transformer (ViT) Encoder
Integration of EEG Signals
Data Preparation
Datasets Used
Data Loading and Processing
Training Process
Training Script Options
Training Workflow
Results Generation
Project Structure
Usage
Prerequisites
Installation
Training the Model
Generating Results
Contributing
License
Acknowledgements
Introduction
This project implements an advanced EEG-Conditioned Image Reconstruction system that leverages a Conditional uNet architecture integrated with a Vision Transformer (ViT) Encoder. By embedding EEG signals into the image generation process, the model aims to reconstruct high-fidelity images based on neural activity data. This approach bridges the gap between brain-computer interfaces and image synthesis, enabling innovative applications in neuroscience and artificial intelligence.

Features
Conditional uNet Architecture: Utilizes conditional layers within a uNet framework to incorporate EEG embeddings for guided image reconstruction.
Vision Transformer Encoder: Employs a ViT-based encoder to extract rich representations from EEG signals.
Flexible Training Options: Offers multiple training modes, including training/unfreezing specific model components.
Comprehensive Data Handling: Processes diverse EEG datasets with robust data loaders tailored for different experimental setups.
Distributed Training Support: Facilitates scalable training across multiple GPUs and nodes.
Validation and Result Generation: Includes scripts for generating and validating reconstructed images against ground truth.
Architecture Overview
Conditional uNet
The Conditional uNet serves as the core image generation model in this system. Unlike standard uNet architectures, the conditional uNet incorporates additional information—in this case, EEG signal embeddings—to guide the image reconstruction process. This conditioning allows the model to generate images that are not only coherent but also semantically aligned with the underlying neural activity.

Key Components:

Encoder Path: Extracts hierarchical features from the input latent representations.
Decoder Path: Reconstructs the image from the encoded features, integrating conditional embeddings at various stages.
Skip Connections: Facilitates the flow of information between corresponding layers in the encoder and decoder, enhancing reconstruction fidelity.
Conditional Layers: Integrates EEG embeddings into the decoding process to influence image generation dynamically.
Vision Transformer (ViT) Encoder
The Vision Transformer (ViT) Encoder is responsible for transforming raw EEG signals into meaningful embeddings that inform the image reconstruction process. By leveraging transformer-based architectures, the encoder captures long-range dependencies and intricate patterns within the EEG data, providing a robust foundation for conditional image generation.

Advantages:

Rich Feature Extraction: Captures complex relationships within EEG signals.
Scalability: Efficiently processes large and high-dimensional EEG datasets.
Flexibility: Adapts to various EEG signal modalities and experimental conditions.
Integration of EEG Signals
Integrating EEG signals into the image reconstruction pipeline involves several critical steps:

EEG Embedding: Raw EEG data is processed through the ViT encoder to obtain dense embeddings.
Conditional Injection: These embeddings are fed into the conditional uNet at specific layers, influencing the decoding process.
Guided Reconstruction: The conditional uNet leverages the EEG embeddings to reconstruct images that reflect the neural activity patterns captured in the EEG data.
Loss Optimization: The training process minimizes the reconstruction loss between the generated images and ground truth, ensuring high fidelity and semantic alignment.
This integration allows the model to generate images that are not only accurate in terms of pixel-wise reconstruction but also meaningful in relation to the EEG-derived neural representations.

Data Preparation
Datasets Used
VC EEG Dataset:

Description: A specialized EEG dataset capturing neural responses under specific visual conditions.
Usage: Primary dataset for training and fine-tuning the encoder and conditional uNet.
THINGS EEG Dataset:

Description: EEG recordings from participants engaged in tasks within the THINGS experimental framework.
Usage: Supplementary dataset for enhancing the encoder's generalization capabilities.
Stimuli Dataset:

Description: Contains images used as stimuli during EEG recording sessions.
Usage: Provides ground truth images for training and validation.
Data Loading and Processing
Data handling is managed through dedicated data loaders and processors to ensure seamless integration with the training pipeline.

dataset.py:

Functionality: Processes raw EEG data, performs preprocessing steps like filtering and epoch extraction, and prepares it for training.
Components:
EEGDataProcessor: Handles loading, resampling, normalization, and epoch extraction from raw EEG files.
EEGDataLoader: Converts processed EEG data into PyTorch tensors and manages dataset indexing.
SplitDataset: Facilitates the division of datasets into training and testing subsets based on predefined splits.
things_dataloader.py:

Functionality: Specifically handles EEG data from the THINGS dataset, including channel interpolation and image extraction.
Components:
EEGDataProcessor: Similar to dataset.py, tailored for THINGS-specific data formats.
EEGDataLoader: Manages EEG and corresponding image data, ensuring synchronized loading for conditional training.
moabb_dataloader.py:

Functionality: Manages loading and preprocessing of EEG data from the MOABB repository (if applicable).
Components:
MneRawObject: Handles raw EEG data operations such as resampling and normalization.
EEGDataLoader: Converts MOABB data into usable PyTorch tensors.
generate_results.py:

Functionality: Utilizes trained models to generate and validate reconstructed images based on EEG inputs.
Components:
Latent Sampling: Initializes and iteratively denoises latent vectors through the diffusion process.
Image Decoding: Transforms latent representations back into image space using the VAE.
Training Process
Training Script Options
The training process is orchestrated through the ldm_train_v8.py script, which offers various configurations to accommodate different training scenarios and datasets. Key options include:

Participant Omission:

Description: Excludes a specified participant's data from training to assess model generalization.
Usage: Controlled via the --omit_participant command-line argument.
Training Cases:

Case 0: Train both the conditional uNet and the ViT encoder simultaneously.
Case 1: Train only the conditional uNet while freezing the encoder.
Case 2: Train only the encoder while freezing the uNet.
Dataset Selection:

Dataset 0: VC EEG Dataset.
Dataset 1: THINGS EEG Dataset.
Dataset 2: All participants in the VC EEG Dataset except the omitted one.
Training Workflow
The training workflow involves several interconnected steps, each managed by specific modules and scripts:

Configuration Setup:

File: config.py
Function: Defines training parameters, hyperparameters, dataset paths, and other essential configurations.
Model Initialization:

Files: ldm_train_v8.py, unet_utils.py, encoder.py
Function: Loads pretrained models (uNet, VAE) and initializes the ViT-based encoder. Depending on the training case, specific model components are frozen or set to trainable.
Data Loading:

Files: dataset.py, things_dataloader.py
Function: Loads and preprocesses EEG and image data, ensuring synchronized batching for conditional training.
Training Loop:

File: ldm_train_v8.py
Function: Iterates over training epochs, processing batches of EEG and image data. The conditional uNet is trained to minimize the reconstruction loss, incorporating EEG embeddings to guide image generation.
Loss Computation:

Components: Mean Squared Error (MSE) loss between generated and ground truth images, augmented with CLIP-based loss for semantic alignment.
Function: Ensures both pixel-wise accuracy and semantic fidelity in image reconstruction.
Optimization and Scheduling:

Function: Utilizes AdamW optimizer with learning rate scheduling to facilitate efficient convergence.
Checkpointing and Logging:

Function: Periodically saves model checkpoints and logs training progress, including loss metrics and validation results.
Validation and Result Generation:

Files: generate_results.py
Function: Generates reconstructed images based on validation EEG inputs, comparing them against ground truth images to assess model performance.
Results Generation
The generate_results.py script is responsible for generating and validating reconstructed images using the trained models. It performs the following steps:

Model Loading:

Loads the trained conditional uNet and ViT encoder from saved checkpoints.
Initializes the VAE and noise scheduler for the diffusion process.
Data Preparation:

Loads and preprocesses EEG data for participants.
Aligns EEG embeddings with image generation requirements.
Image Generation:

Initializes latent vectors with random noise.
Iteratively denoises the latents through the diffusion scheduler, conditioned on EEG embeddings.
Decodes the final latents into high-fidelity images using the VAE.
Validation:

Compares generated images against ground truth, saving both for qualitative assessment.
Logs the validation results for further analysis.
