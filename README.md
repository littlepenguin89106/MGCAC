# MGCAC
\[ACCV2024\] Official implementation of "A Recipe for CAC: Mosaic-based Generalized Loss for Improved Class-Agnostic Counting"

## Installation
We have tested with Python 3.10 and Pytorch 2.4.1, please follow the Pytorch official instructions to build your environment. For other required Python packages, use the `requirements.txt` for installation. 

## Data Preparation

### FSC147
Follow [BMNet](https://github.com/flyinglynx/Bilinear-Matching-Network?tab=readme-ov-file#data-preparation) to setup your data.

### FSC-Mosaic
Download the data from [here](https://drive.google.com/file/d/1EtoCo6TT_gpdGSL0pSkeNvHQEAErS6OU/view?usp=sharing). 

## Evaluating

We use YAML files in the `config` directory to run experiments. Please refer to `default.py` for parameter setup.

### Pretrained Weights
Download the pretrained weights from [here](https://drive.google.com/file/d/17jVlHF3NscQgbTlVQiud2eYut9SCKYfs/view?usp=sharing). Then modify `DIR.runs` and `DIR.exp` in the configuration to set the path to your pretrained weights.

### Evaluating FSC147
Run `python main.py --cfg=config/eval_fsc.yaml`.

### Evaluating FSC-Mosaic
Run `python mosaic.py --cfg=config/eval_mosaic.yaml`.

## Training
Download the pretrained backbone from [here](https://github.com/microsoft/CvT), we use `CvT-21-384x384-IN-22k.pth` checkpoint. Modify `MODEL.BACKBONE.PRETRAINED_PATH` in the configuration.

Run `python main.py --cfg=config/train.yaml`.

# Acknowledgments
Our code is based on the works of [FamNet](https://github.com/cvlab-stonybrook/LearningToCountEverything), [BMNet](https://github.com/flyinglynx/Bilinear-Matching-Network), and [MixFormer](https://github.com/MCG-NJU/MixFormer), and we appreciate their outstanding work. This work was primarily supported by the National Science and Technology Council (NSTC) and Academia Sinica. We also extend our thanks to the National Center for High-performance Computing (NCHC) of the National Applied Research Laboratories (NARLabs) in Taiwan for providing the necessary computational and storage resources.
