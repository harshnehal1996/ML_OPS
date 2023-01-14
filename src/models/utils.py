import os
import torch

def load_model(model, optimizer, checkpoint_path):
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

    config = config.decoder
    
    use_cuda = config.hyperparameters.cuda
    
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    
    batch_size = config.hyperparameters.batch_size
    if batch_size != int(batch_size) or batch_size <= 0:
        raise_error('invalid batch size %d' % batch_size) 

    checkpoint_path = config.hyperparameters.checkpoint_path
    if checkpoint_path is not None and not os.path.isfile(checkpoint_path):
        raise Exception("file %s not found!" % checkpoint_path) 
    
    lr = config.hyperparameters.lr
    if lr <= 0:
        raise_error('invalid learning rate %f' % lr)
    
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
    
    args = {'epochs' : epochs,\
            'batch_size' : batch_size,\
            'lr': lr,\
            'DEVICE' : DEVICE,\
            'loss_function' : loss_function,\
            'metrics' : metrics,\
            'checkpoint_frequency' : checkpoint_frequency,\
            'best_metric' : best_metric,\
            'checkpoint_path' : checkpoint_path}
    
    return  args
    