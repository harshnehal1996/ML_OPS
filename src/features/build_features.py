import numpy as np
import os
import torch
import cv2
import albumentations as albu
import segmentation_models_pytorch as smp
from ..data.make_dataset import ImageDataset
import __main__
setattr(__main__, "ImageDataset", ImageDataset)

CLASSES = {'sky' : 23,\
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


def process_image(image, mask, preprocessing_fn, crop=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 256))
    mask_image = cv2.resize(mask_image, (512, 256), interpolation=cv2.INTER_NEAREST)
    masks = []

    global CLASSES
    for key in CLASSES.keys():
        masks.append((mask_image == CLASSES[key]).astype(np.int64))

    mask = np.stack(masks, axis=-1)
    
    # apply preprocessing        
    if preprocessing_fn is not None:
        sample = preprocessing_fn(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']
    else:
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
    
    if crop:
        start = np.random.randint(0,512-256)
        image = image[:, :, start:start+256]
        mask = mask[:, :, start:start+256]
    
    return torch.FloatTensor(image), torch.FloatTensor(mask)


class Dataset(torch.utils.data.Dataset):    
    def __init__(
            self, 
            dataset, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.dataset = torch.load(dataset)        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = cv2.imread(self.dataset.input_images[i])
        mask_image = cv2.imread(self.dataset.output_images[i])[:,:,0]
        return process_image(image, mask_image, self.preprocessing)
        
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

def get_test_data(ENCODER='', ENCODER_WEIGHTS='', PROJECT_PATH=''):
    """ Preprocessing script that changes created datasets into more
        suited for training purposes
    """
    dir_path = PROJECT_PATH
    data_path = os.path.join(dir_path, "data_git/processed")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    input_path_test = os.path.join(data_path, "test_set.pt")

    test_dataset = Dataset(input_path_test, preprocessing=get_preprocessing(preprocessing_fn))
    
    return test_dataset
