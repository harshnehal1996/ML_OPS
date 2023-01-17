import torch
import segmentation_models_pytorch as smp

# TODO : Automate number of classes and encoder type parameters

class SegmentationModel(torch.nn.Module):
    def __init__(self, config):
        super(SegmentationModel, self).__init__()

        # create two different experiments for each decoder
        if config.decoder == 'Unet':
            self.model = smp.Unet(encoder_name=config.encoder, encoder_weights='imagenet', classes=config.classes, activation=config.activation)
        elif config.decoder == 'FPN':
            self.model = smp.FPN(encoder_name=config.encoder, encoder_weights='imagenet', classes=config.classes, activation=config.activation)
        else:
            raise ValueError("decoder_name = %s not supported." % config.decoder)
        
        # fix weights of the encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, inputs):
        if inputs.ndim != 4:
            raise ValueError("Expected 4D Tensor, but got %dD instead" % inputs.ndim)
        return self.model(inputs)
    
    def load_decoder_weights(self, weights):
        self.model.decoder.load_state_dict(weights)

    def get_decoder_dict(self):
        return self.model.decoder.state_dict()
    
    def get_decoder_params(self):
        return self.model.decoder.parameters()
