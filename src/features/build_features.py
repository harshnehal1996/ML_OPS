<<<<<<< HEAD
import matplotlib.pyplot as plt
import numpy as np
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os.path import isfile, join, dirname, abspath
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import albumentations as albu
import segmentation_models_pytorch as smp
from torchvision import datasets, transforms
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
                if file.endswith("color.png"):
                    self.output_images.append(os.path.join(output_dir, folder, file))


        # self.output_images = [output_dir+f for f in os.listdir(output_dir) if f.endswith('color.png')]
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

class Dataset(Dataset):
    """Read images, apply augmentation and preprocessing transformations.
=======
# import torch
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

# class Dataset(Dataset):
#     """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
>>>>>>> c826de40b17252b9b14b02ffed343ced4699b78b
    
#     Args:
#         dataset(Dataset
#         class_values (list): values of classes to extract from segmentation mask
#         augmentation (albumentations.Compose): data transfromation pipeline 
#             (e.g. flip, scale, etc.)
#         preprocessing (albumentations.Compose): data preprocessing 
#             (e.g. noralization, shape manipulation, etc.)
    
#     """
    
<<<<<<< HEAD
    def __init__(
            self, 
            dataset, 
            classes=None, 
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

>>>>>>> c826de40b17252b9b14b02ffed343ced4699b78b
        
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
    
#     def __getitem__(self, i):
        
<<<<<<< HEAD
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
        
        
        # extract certain classes from mask (e.g. cars)
#         masks = [(mask == v) for v in self.object_to_pixel]
#         mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
>>>>>>> c826de40b17252b9b14b02ffed343ced4699b78b
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
        
<<<<<<< HEAD
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
        return len(self.ids)   
        
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

>>>>>>> c826de40b17252b9b14b02ffed343ced4699b78b


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

def main():
    """ Preprocessing script that changes created datasets into more
        suited for training purposes
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    dir_path = dirname(dirname(dirname(abspath(__file__))))
    data_path = dir_path + "/data/processed"
    viz_path = dir_path + "/visualization"

    ENCODER = 'efficientnet-b3'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    input_path_train = data_path + "/train_set.pt"

    train_dataset = Dataset(input_path_train, preprocessing=get_preprocessing(preprocessing_fn))
    torch.save(train_dataset, data_path + "/train_set_f.pt")

    input_path_test = data_path + "/test_set.pt"

    test_dataset = Dataset(input_path_test, preprocessing=get_preprocessing(preprocessing_fn))
    torch.save(test_dataset, data_path + "/test_set_f.pt")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
