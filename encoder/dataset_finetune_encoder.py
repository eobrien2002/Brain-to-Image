from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import torch

torch.manual_seed(42)

class EEGDataProcessor:
    def __init__(self, eeg_signals_path,participant_number=4):
        # Initialize the processor with file paths and parameters.
        data_dict = torch.load(eeg_signals_path)
        data = [data_dict['dataset'][i] for i in range(len(data_dict['dataset']) ) if data_dict['dataset'][i]['subject']==participant_number]


        self.data_len = 512

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
        eeg = eeg/10
        
        return {"eeg": eeg} 
        

    def __len__(self):
        return len(self.data)


def load_and_preprocess_eeg_data(eeg_signals_path='data/ee.pth'):
    
    dataset = EEGDataProcessor(eeg_signals_path)

    return dataset

if __name__ == "__main__":
    
    dataset = load_and_preprocess_eeg_data(eeg_signals_path='data/eeg.pth')
    dataloader_eeg = DataLoader(dataset, batch_size=16)

    sample = next(iter(dataloader_eeg))
    data = sample['eeg'][0]
    

    # Example usage of the dataset
    print(data.shape)
    plt.plot(data[2].numpy()) # Plot a single EEG channel
    plt.show()





