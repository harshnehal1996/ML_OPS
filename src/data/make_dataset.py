# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from os.path import isfile, join, dirname, abspath
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

class ImageDataset(Dataset):
    def __init__(self, input_dir, output_dir):
        self.input_images = []
        self.output_images = []
        temp_images = []
        temp_masks = []
        
        for folder in os.listdir(input_dir):
            for file in os.listdir(os.path.join(input_dir, folder)):
                if file.endswith(".png"):
                    path = os.path.join(input_dir, folder, file)
                    image = cv2.imread(path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (512, 256))
                    temp_images.append((path, image))

        for folder in os.listdir(output_dir):
            for file in os.listdir(os.path.join(output_dir, folder)):
                if file.endswith("labelIds.png"):
                    path = os.path.join(output_dir, folder, file)
                    mask_image = cv2.imread(path)[:,:,0]
                    mask_image = cv2.resize(mask_image, (512, 256), interpolation=cv2.INTER_NEAREST)
                    temp_masks.append((path, mask_image))
        
        temp_images.sort(key=lambda x: x[0])
        temp_masks.sort(key=lambda x: x[0])
        assert len(temp_images) == len(temp_masks)

        for i in range(len(temp_images)):
            self.input_images(temp_images[i][1])
            self.output_images(temp_masks[i][1])
    
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        return self.input_images[idx], self.output_images[idx]

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
    # train_cities = next(os.walk(input_path_train))[1]

    # transform = transforms.Compose([transforms.ToTensor])
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
