MLOps_Project - Semantic Understanding of Urban Street Scenes using Computer Vision
==============================

* Harsh Rai
* Joshua Sebastian
* Navaneeth KP
* Reczulski Michal 

### Project Goal

The goal of the project is to use Computer Vision for semantic segmentation of Cityscapes Dataset to segment into the following classes: 
* person
* road
* building
* car
* motorcyle
* bicycle
* truck
* bus
* traffic light
* traffic sign
* void

### Framework

Since we chose a Computer Vision problem, we plan to use the [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) and [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) framework. To build the model we are going to use latter which is built on top of the Pytorch Image model framework.

### Data

We are using the [Cityscapes GTfine dataset](https://www.kaggle.com/datasets/xiaose/cityscapes) from kaggle. An example of one of the images is as shown:
<p align="center"><img src="reports\figures\cityscape_example.png" alt="city_seg" width="800" height="440"/>

There are 3475 finely annotated images including train and validation. There are 1525 finely annotated test data.

### Deep Learning Model

We plan to use timm-efficientnet-b2 as an encoder as it has 7M parameters which are pretrained and Unet as decoder which has 4 Million parameters required to be trained. The segmentation_head API from segmentation_models.pytorch will help us define the number of channels in the output mask.

TIMM models are scriptable, exportable and also provides the functionality to build data augmentation pipelines.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Checklist
See [CHECKLIST.md](https://github.com/harshnehal1996/ML_OPS/tree/main/reports/README.md)