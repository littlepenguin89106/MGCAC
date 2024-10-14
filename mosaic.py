import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import os 

from config import cfg 
from loss import get_loss
from dataset.mosaic import build_mosaic_dataset
from engine import evaluate_mosaic,visualize_mosaic
from model import build_model

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def main(args):
    print(args)
    device = torch.device(cfg.TRAIN.device)
    seed = cfg.TRAIN.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(cfg)
    criterion = get_loss(cfg)
    criterion.to(device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
        
    dataset_val = build_mosaic_dataset(cfg)

    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)    
    
    if cfg.VAL.evaluate_only:
        if os.path.isfile(cfg.VAL.resume):
            checkpoint = torch.load(cfg.VAL.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        else:
            print('model state dict not found.')
        if cfg.VAL.visualization:
            visualize_mosaic(model, dataset_val, data_loader_val, device, cfg.DIR.output_dir)
        else:
            evaluate_mosaic(model, data_loader_val, device, cfg.DIR.output_dir)
        return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Class Agnostic Object Counting in PyTorch"
    )
    parser.add_argument(
        "--cfg",
        default="config/bmnet+_fsc147.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    
    cfg.DIR.output_dir = os.path.join(cfg.DIR.runs, cfg.DIR.exp)
    if not os.path.exists(cfg.DIR.output_dir):
        os.mkdir(cfg.DIR.output_dir)    

    cfg.TRAIN.resume = os.path.join(cfg.DIR.output_dir, cfg.TRAIN.resume)
    cfg.VAL.resume = os.path.join(cfg.DIR.output_dir, cfg.VAL.resume)

    with open(os.path.join(cfg.DIR.output_dir, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    main(cfg)