import os
import torch
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parents[2])

def load_model(model, path):
    model.load_decoder_weights(torch.load(path))

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_decoder_weights(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    score = checkpoint['score']
    return epoch + 1, score

def save_checkpoint(model, epoch, score, optimizer):
    torch.save({
                'epoch': epoch,
                'model' : model.get_decoder_dict(),
                'optimizer' : optimizer.state_dict(),
                'score' : score,
                }, 'checkpoint_%d.pth'%epoch)

def parse_inputs(config):
    def raise_error(string):
        raise ValueError(string)

    def check_metrics(metrics):
        for m in metrics:
            if m not in ['IoU', 'Accuracy', 'loss']:
                raise_error("Illegal data")

    def get_best_metric(metric, loss_fn):
        if metric == 'IoU':
            return 'iou_score'
        if metric == 'Accuracy':
            return 'accuracy'
        if metric == 'loss':
            return loss_fn[:-4].lower() + '_' + 'loss'
        raise_error('metric does not match available choices')
    
    def get_optimizer(dictionary):
        if dictionary['name'] not in ['Adam', 'SGD', 'RMSProp']:
            raise_error("optimizer must be Adam or SGD or RMSProp")
        
        if not dictionary['params'].__contains__('lr'):
            raise_error("learining rate is required")
        
        lr = dictionary['params']['lr']
        if lr <= 0:
            raise_error('invalid learning rate %f' % lr)
        
        return dictionary
    
    def get_scheduler(dictionary):
        if dictionary['name'] == 'OneCycleLR':
            M = dictionary.copy()
            nonlocal epochs
            M['params']['epochs'] = epochs
            return M
        
        return dictionary


    model_type = ""
    print(config)
    for k1 in config.keys():
        for k2 in config[k1].keys():
            config = config[k1][k2]
            model_type = k2 + "_" + config.hyperparameters.id
            break
    
    use_cuda = config.hyperparameters.cuda
    
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    
    batch_size = config.hyperparameters.batch_size
    if batch_size != int(batch_size) or batch_size <= 0:
        raise_error('invalid batch size') 

    checkpoint_path = config.hyperparameters.checkpoint_path
    if checkpoint_path is not None and not os.path.isfile(checkpoint_path):
        raise Exception("file %s not found!" % checkpoint_path) 
    
    epochs = config.hyperparameters.epochs
    if epochs < 0 or epochs != int(epochs):
        raise_error('invalid epochs value')

    loss_function = config.hyperparameters.loss_function
    allowedloss = ['JaccardLoss',\
                   'DiceLoss',\
                   'L1Loss',\
                   'MSELoss',\
                   'CrossEntropyLoss',\
                   'NLLLoss',\
                   'BCELoss',\
                   'BCEWithLogitsLoss']
    
    if loss_function not in allowedloss:
        raise_error('loss function not recognized, use from: ' + ', '.join(allowedloss))

    metrics = config.hyperparameters.metrics.split('\n')[:-1]
    if len(metrics) == 0:
        raise_error('no metric provided')
    check_metrics(metrics)
    
    checkpoint_frequency = config.hyperparameters.checkpoint_frequency
    if checkpoint_frequency != int(checkpoint_frequency):
        raise_error('checkpoint_frequency value is invalid')
    
    best_metric = config.hyperparameters.best_metric
    best_metric = get_best_metric(best_metric, loss_function)
    
    if config.hyperparameters.seed is not None:
        torch.manual_seed(config.hyperparameters.seed)

    if config.hyperparameters.optimizer is None: 
        raise_error("illegal optimizer")
    
    optimizer = get_optimizer(config.hyperparameters.optimizer)
    lr_scheduler = config.hyperparameters.lr_scheduler

    if lr_scheduler is not None:
        if not lr_scheduler.__contains__('name') or not lr_scheduler.__contains__('params'):
            raise_error("illegal lr_scheduler data")
        
        lr_scheduler = get_scheduler(lr_scheduler)

    metadata = {'ENCODER' : config.hyperparameters.encoder,\
                'ENCODER_WEIGHTS' : 'imagenet',\
                'PROJECT_PATH' : PROJECT_DIR}
    
    args = {'epochs' : epochs,\
            'batch_size' : batch_size,\
            'DEVICE' : DEVICE,\
            'loss_function' : loss_function,\
            'metrics' : metrics,\
            'checkpoint_frequency' : checkpoint_frequency,\
            'best_metric' : best_metric,\
            'optimizer' : optimizer,\
            'lr_scheduler' : lr_scheduler,\
            'checkpoint_path' : checkpoint_path,\
            'metadata' : metadata,\
            'model_type' : model_type}
    
    return  config.hyperparameters, args
