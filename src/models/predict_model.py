import os
import torch
import utils
import numpy as np
from ..features.build_features import process_image, CLASSES, get_preprocessing
from .model import SegmentationModel
import segmentation_models_pytorch as smp
from hydra import initialize, compose

models = {}

def set_model(model_type):
    if model_type not in ['b1_exp', 'b3_exp', 'b1_cycle', 'b3_cycle']:
        raise ValueError('illegal model type provided')
    
    if model_type == "b1_exp":
        arg1 = "b1"
        arg2 = "b1=sgd_exp"
    elif model_type == "b1_cycle":
        arg1 = "b1"
        arg2 = "b1=sgd_cycle"
    elif model_type == "b2_exp":
        arg1 = "b2"
        arg2 = "b2=sgd_exp"
    else:
        arg1 = "b2"
        arg2 = "b2=sgd_cycle"
    
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
        prediction = model(image.unsqueeze(0))
    
    

    







# @hydra.main(config_path='../conf', config_name='config')
# def predict(config):
#     # Initialize the model
#     model = SegmentationModel(config.decoder.hyperparameters)
#     state_dict = torch.load("models/best_model.pth") # todo: change the model path

#     # load the trained model
#     model.load_state_dict(state_dict=state_dict)
#     model.eval()

#     # load the test dataset
#     test_dataset = torch.load("data/processed/test_set.pt")
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, pin_memory=True)

#     # Parameters
#     args = parse_inputs(config)
#     loss = eval("utils.losses.%s" % args['loss_function'])()
#     metrics = [eval("utils.metrics.%s" % x)() for x in args['metrics']]


#     test_epoch = utils.train.ValidEpoch(
#         model, 
#         loss=loss, 
#         metrics=metrics, 
#         device=args['DEVICE'],
#         verbose=True,
#     )

#     test_logs = test_epoch.run(test_loader)
#     print(test_logs)






