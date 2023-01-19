import os
import torch
import utils
import numpy as np
from ..features.build_features import process_image, CLASSES, get_preprocessing
from .model import SegmentationModel
import segmentation_models_pytorch as smp
from hydra import initialize, compose

models = {}

max_p = max([CLASSES[k] for k in CLASSES.keys()])
I2P = [0 for _ in range(len(CLASSES))]
for i, k in enumerate(CLASSES.keys()):
    I2P[i] = np.array([128, max(int(float(k) * 255 / max_p), 255), max(int(float(k) * 255 / max_p), 255)], dtype=np.uint8)

def set_model(model_type):
    if model_type not in ['b1_exp', 'b3_exp', 'b1_cycle', 'b3_cycle']:
        raise ValueError('illegal model type provided')
    
    if model_type == "b1_exp":
        arg1 = "mit_b1"
        arg2 = "training/b1=sgd_exp"
    elif model_type == "b1_cycle":
        arg1 = "mit_b1"
        arg2 = "training/b1=sgd_cycle"
    elif model_type == "b2_exp":
        arg1 = "efficientnet-b3"
        arg2 = "training/b3=sgd_exp"
    else:
        arg1 = "efficientnet-b3"
        arg2 = "training/b3=sgd_cycle"
    
    path = os.path.join(utils.PROJECT_DIR, 'outputs/best_model_%s.pt' % model_type)

    with initialize(version_base=None, config_path='../conf'):
        config = compose(config_name="config", overrides=["training=%s" % arg1, arg2])
        
        for k1 in config.keys():
            for k2 in config.keys():
                config = config[k1][k2]
                break
        
        model = SegmentationModel(config.hyperparameters)
        utils.load_model(model, path)

        model.eval()
        preprocessing_fn = smp.encoders.get_preprocessing_fn(config.hyperparameters.encoder, 'imagenet')
        models[model_type] = (model, get_preprocessing(preprocessing_fn))

def predict(image, model_type):
    if not models.__contains__(model_type):
        set_model(model_type)
    
    num_classes = len(CLASSES)
    model, processing_fn = models[model_type]
    fake_mask = np.zeros(image.shape, dtype=np.uint8)
    
    image, _ = process_image(image, fake_mask, processing_fn, crop=False)

    with torch.no_grad():
        prediction = (model(image.unsqueeze(0))[0]).cpu()
        prediction = torch.argmax(prediction, dim=0).numpy()

    mask = np.zeros((*prediction.shape,3), dtype=np.uint8)
    
    for i in range(num_classes):
        mask[prediction == i] = I2P[i]

    return mask





