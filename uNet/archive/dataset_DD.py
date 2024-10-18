from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
import torch
from PIL import Image
import zipfile
import tempfile
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from transformers import AutoProcessor
import os
from scipy.interpolate import interp1d
import numpy as np
import torch
from scipy.interpolate import interp1d
from collections import defaultdict


class EEGDataProcessor:
    def __init__(self, eeg_signals_path,participant_number):
        # Initialize the processor with file paths and parameters.
        data_dict = torch.load(eeg_signals_path)
        data = [data_dict['dataset'][i] for i in range(len(data_dict['dataset']) ) if data_dict['dataset'][i]['subject']==participant_number]

        self.labels = data_dict["labels"]
        self.images = data_dict["images"]
        self.image_path = 'data/stimuli.zip'
        self.finetune_path = 'data/finetune'
        # Get all image file names from the finetune folder
        self.image_files = os.listdir(self.finetune_path)

        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
        self.tmp_path = None
        self.zip_extracted = False

        self.data_len = 512


        # loop through all the image names in the data and if its not in finetune delete it from the data
        for i in range(len(data)):
            if f'{self.images[data[i]["image"]]}.JPEG' not in self.image_files:
                data[i] = None
        
        data = [x for x in data if x is not None]

        self.data = data


    def __getitem__(self,i):       
            # Load and standardize raw EEG data.
            eeg = self.data[i]["eeg"].float().t()
            eeg = eeg[20:480,:]
            eeg = np.array(eeg.transpose(0,1))
            x = np.linspace(0, 1, eeg.shape[-1])
            x2 = np.linspace(0, 1, self.data_len)
            f = interp1d(x, eeg)
            eeg = f(x2)
            eeg = torch.from_numpy(eeg).float()

            if not self.zip_extracted:
                self.create_temp_dir_and_extract_images()
            try:
                image_name = self.images[self.data[i]["image"]]
                image_path = Path(self.tmp_path) / Path("imageNet_images") / Path(image_name.split('_')[0]) / Path(image_name+'.JPEG')
                image_raw = Image.open(image_path).convert('RGB') 
                image_raw = self.processor(images=image_raw, return_tensors="pt")
                image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)

            except:
                return None

            return {"eeg": eeg, "image_raw": image_raw['pixel_values'],'image_path':image_path} 
        
     
        
    def create_temp_dir_and_extract_images(self):
        self.tmp_path = tempfile.mkdtemp()
        
        with zipfile.ZipFile('data/stimuli.zip', 'r') as image_zip:
            image_zip.extractall(self.tmp_path)
        self.zip_extracted = True
        
    def clean_up_temp_dir(self):
        # Clean up the temporary directory.
        if self.tmp_path and Path(self.tmp_path).exists():
            shutil.rmtree(self.tmp_path)

    def __len__(self):
        return len(self.data)

def load_and_preprocess_eeg_data(eeg_signals_path,participant_number):
    dataset = EEGDataProcessor(eeg_signals_path, participant_number)

    return dataset, dataset.tmp_path

if __name__ == "__main__":
    
    train_dataset,tmp_path1 = load_and_preprocess_eeg_data('data/eeg.pth', 6)

    print(len(train_dataset))
    
    # Example usage of the dataset
    sample = train_dataset[1]
    print(sample['eeg'].shape)
    plt.plot(sample['eeg'][2].numpy()) # Plot a single EEG channel
    plt.show()

    image_path = sample['image_path']
    image  = Image.open(image_path).convert('RGB') 
    plt.imshow(image)
    plt.show()

    if tmp_path1 and Path(tmp_path1).exists():
            shutil.rmtree(tmp_path1)
