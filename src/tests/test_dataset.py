import torch
import sys

from torch.utils.data import DataLoader
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from os.path import isfile, join
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path

src_dir = Path(__file__).resolve().parents[1]
model_dir = os.path.join(src_dir, 'data')
sys.path.append(model_dir)
from make_dataset import ImageDataset 
from hydra import initialize, compose

def test_dataset():
    with initialize(version_base=None, config_path='../conf/data/'):
        config = compose(config_name="dataset")
        import __main__
        setattr(__main__, "ImageDataset", ImageDataset)
        print(config)
        project_dir = Path(__file__).resolve().parents[2]
        test_data = torch.load(os.path.join(project_dir, config.dataset.test_path))
        train_data = torch.load(os.path.join(project_dir, config.dataset.train_path))
        N_train = 2975
        N_test = 500
        assert len(train_data) == N_train 
        assert len(test_data) == N_test
