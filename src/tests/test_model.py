import os
import sys
import pytest
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
        
        for k1 in config.keys():
            config = config[k1]
            break
        
        model = SegmentationModel(config.hyperparameters)
        with torch.no_grad():
            inputs = torch.randn(1,3,256,256)
            output = model(inputs)
            assert tuple(output.shape) == (1,config.hyperparameters.classes,256,256)
        
        with pytest.raises(ValueError, match=r"Expected 4D Tensor, but got [0-9]*D instead"):
            model(torch.randn(1, 288, 229))





