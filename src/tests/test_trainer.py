import os
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parents[1]
model_dir = os.path.join(src_dir, 'models')
sys.path.append(model_dir)

from hydra import initialize, compose
from model import SegmentationModel
import torch

def test_model():
    with initialize(version_base=None, config_path='../conf'):
        config = compose(config_name="config")
        print(config)
        for k1 in config.keys():
            for k2 in config[k1].keys():
                for k3 in config[k1][k2].keys():
                    config = config[k1][k2][k3]
                    break
        
        model = SegmentationModel(config.hyperparameters)
        with torch.no_grad():
            inputs = torch.randn(1,3,256,256)
            output = model(inputs)
            assert tuple(output.shape) == (1,config.hyperparameters.classes,256,256)

# def test_error_on_wrong_shape():
#     model = MyAwesomeModel()
#     with pytest.raises(ValueError, match=r"Expected 4D tensor, got [1-9][0-9]*D tensor instead"):
#         model(torch.randn(1,28,29))





