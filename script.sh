#!/bin/bash
cd app
python3 -m src.data.make_dataset
python3 -m src.models.train_model

