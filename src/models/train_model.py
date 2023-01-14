import os
import sys
import hydra
import torch
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as utils
from ..features.build_features import get_train_data
from .model import SegmentationModel
from .utils import parse_inputs, load_model, save_checkpoint

# load hydra config
@hydra.main(config_path='../conf', config_name='config')
def train(config) -> None:
    # set up the training parameters from the config files
    args = parse_inputs(config)
    
    # initialize the model
    model = SegmentationModel(config.decoder.hyperparameters)
    dataset = get_train_data()
    n = len(dataset)
    tslen = int(n / 4)
    trlen = n - tslen
    train_set, val_set = torch.utils.data.random_split(dataset, [trlen, tslen])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args['batch_size'])
    loss = eval("utils.losses.%s" % args['loss_function'])()
    metrics = [eval("utils.metrics.%s" % x)() for x in args['metrics']]
    optimizer = torch.optim.Adam(model.get_decoder_params(), lr=args['lr'], betas=(0.5, 0.999))

    if args['checkpoint_path'] is not None:
        start_epoch, max_score = load_model(model, optimizer, args['checkpoint_path'])
    else:
        start_epoch = 0
        max_score = 0

    train_epoch = utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=args['DEVICE'],
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=args['DEVICE'],
        verbose=True,
    )
    best_metric = args['best_metric']
    checkpoint_frequency = args['checkpoint_frequency']

    for i in range(start_epoch, args['epochs']):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(trainloader)
        valid_logs = valid_epoch.run(valloader)
        
        if max_score < valid_logs[best_metric]:
            max_score = valid_logs[best_metric]
            torch.save(model, './best_model.pth')
            print('Best Model saved!')
        
        if checkpoint_frequency > 0 and (i+1) % checkpoint_frequency == 0:
            save_checkpoint(model, i, max_score, optimizer)

if __name__ == '__main__':
    train()
