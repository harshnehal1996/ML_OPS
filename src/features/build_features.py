import torch
from torch.utils.data import Dataset

class RandomData(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1000
    
    def __getitem__(self, index):
        x = torch.randn(3, 256, 256)
        y = torch.zeros(11, 256, 256, dtype=torch.long)
        labels = torch.randint(0, 11, (1, 256, 256))
        y.scatter_(0, labels, 1)
        return x, y

def get_train_data():
    return RandomData()
