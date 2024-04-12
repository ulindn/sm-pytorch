# References: 
## https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
## https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Define the dataset
class FashionMNISTDataset(Dataset):
    def __init__(self):

        # Download training data from open dataset
        self.training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=transform)

        # Download test data from open dataset
        self.test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=transform)
        return

    '''
    # not needed for this example
    
    def __len__(self):
        return len(self.training_data), len(self.test_data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y
    
    def split():
        # train = 
        # test = 
        return # train, test
    '''

# Create data loaders
def create_data_loaders(batch_size):
    # Create data loaders
    dsFMNIST = FashionMNISTDataset()
    train_dataloader = torch.utils.data.DataLoader(dataset=dsFMNIST.training_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset=dsFMNIST.test_data, batch_size=batch_size)

    # Print the number of batches
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of test batches: {len(test_dataloader)}")

    # Print shape of training data and test data with X and y
    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    return train_dataloader, test_dataloader
