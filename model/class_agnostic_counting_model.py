import torch
from torch import nn
from model.counter import get_counter
from model.mixformer import build_mixformer
from dataset.train import pad_ratio
import torch.nn.functional as F

class CACModel(nn.Module):
    """ Class Agnostic Counting Model"""
    
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_mixformer(cfg)

        self.counter = get_counter(cfg)
        
    def forward(self, samples: torch.Tensor, patches: torch.Tensor, is_train: bool):
        # Stage 1: extract features for query images and exemplars
        patches = patches['patches']
        search = self.backbone(patches,samples)
        density_map = self.counter(search)

        if not is_train:
            return density_map
        else:
            return {'density_map': density_map}

    def forward_ref_boxes(self,query,boxes):
        refs = []
        scale_embedding = []
        nw,nh = query.size()[1:]
        for b in boxes[0]:
            y1, x1 = int(b[1]), int(b[2])  
            y2, x2 = int(b[3]), int(b[4])
            ref = pad_ratio(query[:,y1:y2,x1:x2])
            refs.append(F.interpolate(ref[:,None],size=(128,128),mode='bicubic').squeeze())
            scale = [abs(y2 -y1) / nh, abs(x2 - x1) / nw]
            scale_embedding.append(scale)
        refs = torch.stack(refs)
        scale_embedding = torch.tensor(scale_embedding)
        output = self.forward(query.unsqueeze(0),{'patches':refs.unsqueeze(0),'scale_embedding':scale_embedding.cuda().unsqueeze(0)},False)
        return output
