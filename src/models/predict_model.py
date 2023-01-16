import torch
import hydra
from .model import SegmentationModel
import segmentation_models_pytorch.utils as utils
from .utils import parse_inputs, load_model, save_checkpoint

@hydra.main(config_path='../conf', config_name='config')

def predict(config):
    # Initialize the model
    model = SegmentationModel(config.decoder.hyperparameters)
    state_dict = torch.load("models/best_model.pth") # todo: change the model path

    # load the trained model
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    # load the test dataset
    test_dataset = torch.load("data/processed/test_set.pt")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, pin_memory=True)

    # Parameters
    args = parse_inputs(config)
    loss = eval("utils.losses.%s" % args['loss_function'])()
    metrics = [eval("utils.metrics.%s" % x)() for x in args['metrics']]


    test_epoch = utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=args['DEVICE'],
        verbose=True,
    )

    test_logs = test_epoch.run(test_loader)
    print(test_logs)






