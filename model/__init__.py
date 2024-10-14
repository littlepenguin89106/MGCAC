import torch

from model.class_agnostic_counting_model import CACModel

def build_model(cfg):

    model = CACModel(cfg)
    
    return model
    
    
    