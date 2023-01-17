import torch
import sys

from torch.utils.data import DataLoader
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from os.path import isfile, join
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import pytest
from pathlib import Path

src_dir = Path(__file__).resolve().parents[1]
model_dir = os.path.join(src_dir, 'data')
sys.path.append(model_dir)
# sys.path.append(r"E:\DTU\MLOPs\ML_OPS\src\data")
from make_dataset import ImageDataset
# from make_dataset  
from hydra import initialize, compose


def test_dataset():
    with initialize(version_base=None, config_path='../conf/data/'):
        config = compose(config_name="dataset")
        import __main__
        setattr(__main__, "ImageDataset", ImageDataset)

        test_data = torch.load(config.test_path)
        train_data = torch.load(config.train_path)

        # test_loader = DataLoader(dataset=test_data, batch_size=100, shuffle=True)
        # train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)

        N_train = 2975
        N_test = 500

        # print(len(test_data))
        # print(len(train_data))
        assert len(train_data) == N_train 
        assert len(test_data) == N_test
        assert all(torch.Size([3,255,510]) == x[0].shape for x in train_data)
        assert all(torch.Size([4,255,510]) == x[1].shape for x in train_data)
        assert all(torch.Size([3,255,510]) == x[0].shape for x in test_data)
        assert all(torch.Size([4,255,510]) == x[1].shape for x in test_data)

# test_dataset()
