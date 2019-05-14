"""Dataloader for UrbanSound8K dataset"""
# Standard dist imports

# Third party imports
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Project level imports

# Module level constants

def get_dataloader(csv_file, batch_size, shuffle=True, num_workers=4):
    dataset = UrbanSoundDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True)
    return loader

class UrbanSoundDataset(Dataset):
    """Custom dataset class for UrbanSound8K dataset"""
    def __init__(self):
        pass

if __name__ == '__main__':
    batch_size = 16
    csv_file = '<FILL IN>'
    loader = get_dataloader(csv_file=csv_file,
                            batch_size=batch_size)
    for i, (img, label) in enumerate(loader['train']):
        img = img.numpy()
        lbl = label.numpy()
        print(i, img.shape, img.min(), img.max(), img.dtype)
        print(i, lbl.shape, lbl.min(), lbl.max(), lbl.dtype)
        break