import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
# Set seed
torch.manual_seed(42)

class MneRawObject:
    def __init__(self, raw_objects, data_len=512, data_chan=128):
        total_data = []
        for raw in raw_objects:
            raw.resample(1000)
            
            data = raw.get_data()
            total_data.append(data)

        #stack the data along the channels axis
        total_data = np.concatenate(total_data, axis=1)


        total_data = (total_data - np.mean(total_data, axis=0)) / np.std(total_data, axis=0)

        epochs = []
        num_epochs = total_data.shape[1] // data_len

        for i in range(num_epochs):
            start_idx = i * data_len
            end_idx = start_idx + data_len
            epoch = total_data[:, start_idx:end_idx]
            epochs.append(epoch)

        data = np.array(epochs)

        # Initialize an array of zeros with dimensions epochs, 'self.data_chan' and 'self.data_len'
        output_data = np.zeros((data.shape[0], data_chan, data_len))
        print(output_data.shape)

        # Check if 'self.data_chan' is greater than the second to last dimension of 'data'
        if data_chan > data.shape[-2]:
            # If so, duplicate 'data' to fill 'output_data' as much as possible
            for i in range(data_chan // data.shape[-2]):
                output_data[:,i * data.shape[-2]: (i + 1) * data.shape[-2], :] = data
            # Handle the case where 'self.data_chan' is not a multiple of 'data.shape[-2]'
            if data_chan % data.shape[-2] != 0:
                output_data[:,-(data_chan % data.shape[-2]):, :] = data[:,:data_chan % data.shape[-2], :]
        elif data_chan < data.shape[-2]:
            # If 'self.data_chan' is less than the second to last dimension of 'data', select a random segment
            random_start_idx = np.random.randint(0, int(data.shape[-2] - data_chan) + 1)
            output_data = data[:, random_start_idx: random_start_idx + data_chan, :]
        elif data_chan == data.shape[-2]:
            # If 'self.data_chan' matches the second to last dimension of 'data', use 'data' as is
            output_data = data

        self.data = output_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        eeg = self.data[index]
        eeg = torch.tensor(eeg, dtype=torch.float32)
        return {'eeg': eeg}
 
def create_dataset_from_raw(raw_objects):
    
    dataset = MneRawObject(raw_objects)

    return dataset


class EEGDataLoader:
    def __init__(self, path):

        with open(path, 'rb') as file:
            raw_list = pickle.load(file)
        print("raw_list", len(raw_list))
        dataset=create_dataset_from_raw(raw_list)
        self.data = dataset.data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        eeg = self.data[index]
        eeg = torch.tensor(eeg, dtype=torch.float32)
        return {'eeg': eeg/10}

   
def create_dataset(path):
    
    dataset = EEGDataLoader(path)

    return dataset

if __name__ == "__main__":
    print("Testing EEGDataLoader")

    dataset = create_dataset('AlexMI.pickle')

    print(len(dataset))

    print(dataset[0]['eeg'].shape)

    sample = dataset[0]['eeg']

    plt.plot(sample[22,:])
    plt.show()


    
