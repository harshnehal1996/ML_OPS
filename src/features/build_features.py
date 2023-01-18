import matplotlib.pyplot as plt
import numpy as np
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os.path import isfile, join, dirname, abspath
import torch
import cv2
import albumentations as albu
import segmentation_models_pytorch as smp
from PIL import Image
from ..data.make_dataset import ImageDataset
import __main__
setattr(__main__, "ImageDataset", ImageDataset)

# class ImageDataset(Dataset):
#     def __init__(self, input_dir, output_dir):
#         self.input_images = []
#         self.output_images = []
#         for folder in os.listdir(input_dir):
#             for file in os.listdir(os.path.join(input_dir, folder)):
#                 if file.endswith(".png"):
#                     self.input_images.append(os.path.join(input_dir, folder, file))
                    
#         for folder in os.listdir(output_dir):
#             for file in os.listdir(os.path.join(output_dir, folder)):
#                 if file.endswith("labelIds.png"):
#                     self.output_images.append(os.path.join(output_dir, folder, file))


#         # self.output_images = [output_dir+f for f in os.listdir(output_dir) if f.endswith('color.png')]
#         self.input_images.sort()
#         self.output_images.sort()
#         self.transform1 = transforms.ToTensor()
#         self.transform2 = None

#     def __len__(self):
#         return len(self.input_images)

#     def __getitem__(self, idx):
#         input_image = Image.open(self.input_images[idx])
#         output_image = Image.open(self.output_images[idx])
#         if self.transform1:
#             input_image = self.transform1(input_image)
#             output_image = self.transform1(output_image)
#         if self.transform2:
#             input_image = self.transform2(input_image)
#             output_image = self.transform2(output_image)
#         return input_image, output_image


class Dataset(torch.utils.data.Dataset):
    """Read images, apply augmentation and preprocessing transformations.
    
     Args:
         dataset(Dataset
         class_values (list): values of classes to extract from segmentation mask
         augmentation (albumentations.Compose): data transfromation pipeline 
             (e.g. flip, scale, etc.)
         preprocessing (albumentations.Compose): data preprocessing 
             (e.g. noralization, shape manipulation, etc.)
    
     """
    
    def __init__(
            self, 
            dataset, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.dataset = torch.load(dataset)
        self.classes = {'sky' : 23,\
                   'building' : 11,\
                   'pole' : 17,\
                   'road' : 7,\
                   'tree' : 21,\
                   'traffic sign' : 20,\
                   'car' : 26,\
                   'static' : 1,\
                   'vegetation' : 22,\
                   'sidewalk' : 8,\
                   'unlabelled' : 3,\
                   'pavement' : 6
                  }

        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        

        # load image and assign mask in 1 of K format
        
        image = cv2.imread(self.dataset.input_images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 256))
        mask_image = cv2.imread(self.dataset.output_images[i])[:,:,0]
        mask_image = cv2.resize(mask_image, (512, 256), interpolation=cv2.INTER_NEAREST)
        masks = []
        for key in self.classes.keys():
            masks.append((mask_image == self.classes[key]).astype(np.int64))

        mask = np.stack(masks, axis=-1)
        
        # apply preprocessing        
        if self.preprocessing is not None:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        else:
            image = np.transpose(image, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
        
        start = np.random.randint(0,512-256)
        image = torch.FloatTensor(image[:, :, start:start+256])
        mask = torch.FloatTensor(mask[:, :, start:start+256])
        
        return image, mask
        
    def __len__(self):
        return len(self.dataset)
        
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_train_data(ENCODER='', ENCODER_WEIGHTS='', PROJECT_PATH=''):
    """ Preprocessing script that changes created datasets into more
        suited for training purposes
    """
    dir_path = PROJECT_PATH
    data_path = os.path.join(dir_path, "data_git/processed")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    input_path_train = os.path.join(data_path, "train_set.pt")

    train_dataset = Dataset(input_path_train, preprocessing=get_preprocessing(preprocessing_fn))
    
    return train_dataset
    # torch.save(train_dataset, input_path_train)

    # input_path_test = os.path.join(data_path, "test_set.pt")

    # test_dataset = Dataset(input_path_test, preprocessing=get_preprocessing(preprocessing_fn))
    # torch.save(test_dataset, input_path_test)
