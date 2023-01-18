# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from os.path import isfile, join, dirname, abspath
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image

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
                if file.endswith("labelIds.png"):
                    self.output_images.append(os.path.join(output_dir, folder, file))
        
        self.input_images.sort()
        self.output_images.sort()
        self.transform1 = transforms.ToTensor()
        self.transform2 = None

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
    data_path = dir_path + "/data/Cityspaces"
    processed_path = dir_path + "/data_git/processed"
    input_path_train = data_path + "/images/train"
    output_path_train = data_path + "/gtFine/train"
    train_dataset = ImageDataset(
        input_dir=input_path_train, output_dir=output_path_train
    )
    torch.save(train_dataset, processed_path + "/train_set.pt")

    input_path_test = data_path + "/images/val"
    output_path_test = data_path + "/gtFine/val"
    test_dataset = ImageDataset(input_dir=input_path_test, output_dir=output_path_test)
    torch.save(test_dataset, processed_path + "/test_set.pt")


    # # Code to check whether the dataset was made properly

    # count = 0
    # for in_img, out_img in train_data_loader:
    #     # print(in_img.shape)

    #     # Create a figure with a grid of subplots
    #     fig, axes = plt.subplots(4, 8)

    #     # Iterate over the array of tensor images
    #     for i, tensor_image in enumerate(in_img):
    #         # Convert the tensor image to a numpy array
    #         tensor_image = tensor_image.permute(1, 2, 0)
    #         image = tensor_image.numpy()
    #         # Get the row and column indices for the subplot
    #         row = i // 8
    #         col = i % 8
    #         # Add the image to the subplot
    #         axes[row, col].imshow(image)
    #         # Remove the axis labels
    #         axes[row, col].axis('off')
    #     count+=1
    #     # Show the plot
    #     plt.show()

    #     # Create a figure with a grid of subplots
    #     fig1, axes1 = plt.subplots(4, 8)

    #     # Iterate over the array of tensor images
    #     for i, tensor_image in enumerate(out_img):
    #         # Convert the tensor image to a numpy array
    #         tensor_image = tensor_image.permute(1, 2, 0)
    #         image = tensor_image.numpy()
    #         # Get the row and column indices for the subplot
    #         row = i // 8
    #         col = i % 8
    #         # Add the image to the subplot
    #         axes1[row, col].imshow(image)
    #         # Remove the axis labels
    #         axes1[row, col].axis('off')
    #     # count+=1

    #     # Show the plot
    #     plt.show()

        # if count > 0:
        #     break


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
