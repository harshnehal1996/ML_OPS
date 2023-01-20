import os
import sys
import pytest
from pathlib import Path

src_dir = Path(__file__).resolve().parents[1]
model_dir = os.path.join(src_dir, 'models')
sys.path.append(model_dir)

from hydra import initialize, compose
from utils import parse_inputs

def test_parser():
    with initialize(version_base=None, config_path='../conf'):
        config = compose(config_name="config")
        
        for k1 in config.keys():
            config[k1].hyperparameters.batch_size = 2.3
            
            with pytest.raises(ValueError, match=r"invalid batch size"):
                parse_inputs(config)
            
            config[k1].hyperparameters.batch_size = 32
            config[k1].hyperparameters.epochs = 2.3
            with pytest.raises(ValueError, match=r"invalid epochs value"):
                parse_inputs(config)
            
            config[k1].hyperparameters.epochs = 23
            parse_inputs(config)
            break
