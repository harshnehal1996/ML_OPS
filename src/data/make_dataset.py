# -*- coding: utf-8 -*-
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

class ImageDataset(Dataset):
    def __init__(self, input_dir, output_dir):
        self.input_images = []
        self.output_images = []
        for folder in os.listdir(input_dir):
            for file in os.listdir(os.path.join(input_dir, folder)):
                if file.endswith(".png"):
                    self.input_images.append(os.path.join(input_dir, folder, file))

        for folder in os.listdir(output_dir):
            for file in os.listdir(os.path.join(output_dir, folder)):
                if file.endswith("color.png"):
                    self.output_images.append(os.path.join(output_dir, folder, file))

        self.transform1 = transforms.ToTensor()
        self.transform2 = transforms.Resize(255)

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_images[idx])
        output_image = Image.open(self.output_images[idx])
        if self.transform1:
            input_image = self.transform1(input_image)
            output_image = self.transform1(output_image)
        if self.transform2:
            input_image = self.transform2(input_image)
            output_image = self.transform2(output_image)
        return input_image, output_image

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    dir_path = dirname(dirname(dirname(abspath(__file__))))
    data_path = dir_path + "/data/raw"
    processed_path = dir_path + "/data/processed"
    input_path_train = data_path + "/images/train"
    output_path_train = data_path + "/gtFine/train"
    train_cities = next(os.walk(input_path_train))[1]
 
    transform = transforms.Compose([transforms.ToTensor])
    train_dataset = ImageDataset(
        input_dir=input_path_train, output_dir=output_path_train
    )
    torch.save(train_dataset, processed_path + "/train_set.pt")

    input_path_test = data_path + "/images/val"
    output_path_test = data_path + "/gtFine/val"
    test_dataset = ImageDataset(input_dir=input_path_test, output_dir=output_path_test)
    torch.save(test_dataset, processed_path + "/test_set.pt")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
