import mne
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch


# Define a class called EEGDataProcessor for processing EEG data
class EEGDataProcessor:
    def __init__(self, raw_file_path, event_description, resample_factor=3):
        # Constructor method to initialize the object with file path, event description, and resample factor
        self.raw_file_path = raw_file_path  # Store the raw EEG file path
        self.event_description = event_description  # Store the event description
        self.resample_factor = resample_factor  # Store the resample factor

    def load_raw_data(self):
        # Method to load the raw EEG data from the specified file
        self.raw = mne.io.read_raw_brainvision(self.raw_file_path, preload=True)  # Read and preload the data
        self.current_sfreq = self.raw.info["sfreq"]  # Get the current sampling frequency

    def preprocess_raw_data(self):
        # Method to preprocess the loaded raw EEG data
        self.raw.set_eeg_reference('average', projection=True)  # Set EEG reference to average
        self.raw.filter(None, 90., fir_design='firwin')  # Apply a low-pass filter at 90 Hz
        #downsample by a factor of 3 so that there are 129 data points per epoch. This makes it simple as we just remove one to get 128
        #Using 128 is much easier for the autoencoder
        #self.raw.resample(sfreq=int(self.current_sfreq / self.resample_factor))  # Resample the data

    #Here we can see the window around the event
    def extract_epochs(self, tmin=-0.128, tmax=0.384):
        # Method to extract epochs from the preprocessed data
        events, event_dict = mne.events_from_annotations(self.raw, event_id=None, regexp=self.event_description)
        self.epochs = mne.Epochs(self.raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True)

    def process_eeg_data(self):
        # Method to process EEG data by sequentially loading, preprocessing, and extracting epochs
        self.load_raw_data()
        self.preprocess_raw_data()
        self.extract_epochs()


class EEGDataLoader:
    def __init__(self, epochs, batch_size=32, train_split=0.8):
        # Assuming 'epochs' is your MNE Epochs object
        eeg_data = epochs.get_data(copy=False)  # Shape (n_epochs, n_channels, n_times)
        print(eeg_data.shape)
        eeg_data = eeg_data[:, :, :512]

        self.n_channels, self.n_times = eeg_data.shape[1], eeg_data.shape[2]
        eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)

        # Create a TensorDataset to wrap the EEG data tensor
        self.dataset = TensorDataset(eeg_data_tensor)

        # Split the dataset into training and testing subsets
        train_size = int(train_split * len(self.dataset))
        test_size = len(self.dataset) - train_size

        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        # Create DataLoaders for both training and testing datasets
        self.batch_size = batch_size
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


def load_and_preprocess_eeg_data(raw_file_path, event_description):
    eeg_processor = EEGDataProcessor(raw_file_path, event_description)
    eeg_processor.process_eeg_data()
    
    # Create an instance of EEGDataLoader using the processed data
    eeg_loader = EEGDataLoader(eeg_processor.epochs)
    
    # Access train and test DataLoaders
    train_loader = eeg_loader.train_data_loader
    test_loader = eeg_loader.test_data_loader
    n_channels = eeg_loader.n_channels
    n_times = eeg_loader.n_times
    
    return train_loader, test_loader, n_channels, n_times




