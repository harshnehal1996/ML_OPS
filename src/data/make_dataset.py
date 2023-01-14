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


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())


class ImageDataset(Dataset):
    def __init__(self, input_dir, output_dir):
        self.input_images = []
        self.output_images = []
        for folder in os.listdir(input_dir):
            for file in os.listdir(os.path.join(input_dir,folder)):
                if(file.endswith('.png')):
                    self.input_images.append(os.path.join(input_dir,folder,file))
        
        for folder in os.listdir(output_dir):
            for file in os.listdir(os.path.join(output_dir,folder)):
                if(file.endswith('color.png')):
                    self.output_images.append(os.path.join(output_dir,folder,file))
            
        # self.output_images = [output_dir+f for f in os.listdir(output_dir) if f.endswith('color.png')]
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

# def main(input_filepath, output_filepath):
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    mypath = r"E:\DTU\MLOPs\ML_OPS\data\Cityspaces\images\train"  #todo: change the path for wsl
    input_path_train = r"E:\DTU\MLOPs\ML_OPS\data\Cityspaces\images\train"
    output_path_train = r"E:\DTU\MLOPs\ML_OPS\data\Cityspaces\gtFine\train"
    train_cities = next(os.walk(mypath))[1]
    # print(test_cities)

    # train_image_folders = [ os.path.join(mypath,x) for x in train_cities ]
    # print(train_image_folders)

    # transform = transforms.Compose([transforms.ToTensor, transforms.Resize(255)])

    # train_images_dataset = datasets.ImageFolder(mypath, transform = transform)

    # dataloader = torch.utils.data.DataLoader(train_images_dataset, batch_size=32, shuffle=True)
    # print(dataloader)
    # print(os.listdir(input_path))
    transform = transforms.Compose([transforms.ToTensor])
    train_dataset = ImageDataset(input_dir=input_path_train, output_dir=output_path_train)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers = 4)
    processed_path = r"E:\DTU\MLOPs\ML_OPS\data\processed"
    torch.save(train_dataset, processed_path + r"/train_set.pt")

    input_path_test = r"E:\DTU\MLOPs\ML_OPS\data\Cityspaces\images\val"
    output_path_test = r"E:\DTU\MLOPs\ML_OPS\data\Cityspaces\gtFine\val"
    test_dataset = ImageDataset(input_dir=input_path_test, output_dir=output_path_test)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True,num_workers = 4)
    processed_path = r"E:\DTU\MLOPs\ML_OPS\data\processed"
    torch.save(test_dataset, processed_path + r"/test_set.pt")



    # Code to check whether the dataset was made properly

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
    #     if count > 2:
    #         break





    



    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
