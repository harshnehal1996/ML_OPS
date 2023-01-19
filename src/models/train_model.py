import logging
import hydra
import torch
import wandb
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as utils
from ..features.build_features import get_train_data
from .model import SegmentationModel
from .utils import parse_inputs, load_model, save_checkpoint

# load hydra config
@hydra.main(config_path='../conf', config_name='config')
def train(config) -> None:
    # set up the training parameters from the config files
    config, args = parse_inputs(config)
    log = logging.getLogger(__name__)
    
    # initiate wandb logging
    wandb.init(project='project')

    # initialize the model
    model = SegmentationModel(config)
    dataset = get_train_data(**args['metadata'])
    n = len(dataset)
    tslen = int(n / 4)
    trlen = n - tslen
    train_set, val_set = torch.utils.data.random_split(dataset, [trlen, tslen])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, pin_memory=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args['batch_size'], num_workers=8)
    loss = eval("utils.losses.%s" % args['loss_function'])()
    metrics = [eval("utils.metrics.%s" % x)() for x in args['metrics']]
    optimizer = eval("torch.optim.%s" % args['optimizer']['name'])\
                (model.get_decoder_params(), **args['optimizer']['params'])

    if args['checkpoint_path'] is not None:
        start_epoch, max_score = load_model(model, optimizer, args['checkpoint_path'])
    else:
        start_epoch = 0
        max_score = 0
    
    scheduler = None
    if args['lr_scheduler'] is not None:
        if args['lr_scheduler']['name'] ==  'OneCycleLR':
            args['lr_scheduler']['params']['steps_per_epoch'] = len(trainloader)
        scheduler = eval("torch.optim.lr_scheduler.%s" % args['lr_scheduler']['name'])\
                    (optimizer, **args['lr_scheduler']['params'])
    
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
        log.info('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(trainloader)
        valid_logs = valid_epoch.run(valloader)
        log.info('training log : ')
        log.info(train_logs)
        log.info('validation log ; ')
        log.info(valid_logs)

        if scheduler is not None:
            scheduler.step()
        
        if max_score < valid_logs[best_metric]:
            max_score = valid_logs[best_metric]
            torch.save(model, './best_model_%s.pth' % args['model_type'])
            log.info('Best Model saved!')
        
        if checkpoint_frequency > 0 and (i+1) % checkpoint_frequency == 0:
            save_checkpoint(model, i, max_score, optimizer)
        
        wandb.log(
            {
                "Train Accuracy:": train_logs['accuracy'],
                "Train IOU Score:": train_logs['iou_score'],
                "Validation Accuracy:": valid_logs['accuracy'],
                "Validation IOU Score:": valid_logs['iou_score'],
            }
        )
        
    

if __name__ == '__main__':
    train()
