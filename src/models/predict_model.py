import os
os.environ["GCLOUD_PROJECT"] = "snappy-byte-374310"

import torch
torch.manual_seed(1)

from .utils import PROJECT_DIR, load_model
import numpy as np
from ..features.build_features import process_image, CLASSES, get_preprocessing
from .model import SegmentationModel
import segmentation_models_pytorch as smp
from hydra import initialize, compose
from io import BytesIO
from google.cloud import storage

storage_client = storage.Client()
def get_data(model_type):
    global storage_client
    bucket = storage_client.bucket("trained_model_pt")
    blob = bucket.blob("best_model_%s.pth" % model_type)
    data = BytesIO(blob.download_as_string())    
    return data

models = {}

max_p = max([CLASSES[k] for k in CLASSES.keys()])
I2P = [0 for _ in range(len(CLASSES))]
for i, k in enumerate(CLASSES.keys()):
    I2P[i] = np.array([128, min(int(float(CLASSES[k]) * 255 / max_p), 255), min(int(float(CLASSES[k]) * 255 / max_p), 255)], dtype=np.uint8)

I2P = [(29,49,193),\
     (49, 184, 193),\
     (28, 219, 53),\
     (219, 28, 41),\
     (8, 252, 8),\
     (196, 0, 255),\
     (255, 255, 255),\
     (226, 226, 111),\
     (12, 17, 170),\
     (226, 111, 111),\
     (0, 0, 0),\
     (155, 155, 155)]

for i in range(len(CLASSES)):
    I2P[i] = np.array(list(I2P[i])[::-1])


def set_model(model_type):
    if model_type not in ['b1_exp', 'b3_exp', 'b1_cycle', 'b3_cycle']:
        raise ValueError('illegal model type provided')
    
    path = os.path.join(PROJECT_DIR, 'models/best_model_%s.pth' % model_type)

    with initialize(version_base=None, config_path='../conf'):
        config = compose(config_name="config", overrides=["training=%s" % model_type])
        
        for k1 in config.keys():
            config = config[k1]
        
        device = torch.device('cpu')
        model = SegmentationModel(config.hyperparameters)
        path = get_data(model_type)
        load_model(model, path, device)

        model.eval()
        preprocessing_fn = smp.encoders.get_preprocessing_fn(config.hyperparameters.encoder, 'imagenet')
        models[model_type] = (model, get_preprocessing(preprocessing_fn))

def predict(image, model_type):
    if not models.__contains__(model_type):
        set_model(model_type)
    
    num_classes = len(CLASSES)
    model, processing_fn = models[model_type]
    fake_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    image, _ = process_image(image, fake_mask, processing_fn, crop=False)

    with torch.no_grad():
        prediction = model(image.unsqueeze(0)).squeeze(0).cpu()
        prediction = torch.argmax(prediction, dim=0).numpy()

    mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    
    for i in range(num_classes):
        mask[prediction == i] = I2P[i]

    return mask
