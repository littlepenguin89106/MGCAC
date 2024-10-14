import torch
import torch.nn as nn
from geomloss import SamplesLoss
import torch.nn.functional as F

def grid(H, W, stride):
    coodx = torch.arange(0, W, step=stride) + stride / 2
    coody = torch.arange(0, H, step=stride) + stride / 2
    y, x = torch.meshgrid( [  coody.type(torch.cuda.FloatTensor) / 1, coodx.type(torch.cuda.FloatTensor) / 1 ] )
    return torch.stack( (x,y), dim=2 ).view(-1,2)

def per_cost(X, Y):
    x_col = X.unsqueeze(-2)
    y_lin = Y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
    C = torch.sqrt(C)
    s = (x_col[:,:,:,-1] + y_lin[:,:,:,-1]) / 2
    s = s * 0.2 + 0.5
    return (torch.exp(C/s) - 1)

def exp_cost(X, Y):
    x_col = X.unsqueeze(-2)
    y_lin = Y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
    C = torch.sqrt(C)
    return (torch.exp(C/scale) - 1.)
    
class GLoss(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.LOSS.cost == 'exp':
            self.cost = exp_cost
        elif cfg.LOSS.cost == 'per':
            self.cost = per_cost
        self.criterion = SamplesLoss(blur=cfg.LOSS.blur, scaling=cfg.LOSS.scaling, debias=False, backend='tensorized', cost=self.cost, reach=cfg.LOSS.reach, p=cfg.LOSS.p)
        global scale
        scale = cfg.LOSS.scale
        
    def forward(self, outputs, density_map, points):
        predict_map = outputs['density_map']

        shape = (1,int(self.cfg.LOSS.crop_size/self.cfg.LOSS.downsample_ratio),int(self.cfg.LOSS.crop_size/self.cfg.LOSS.downsample_ratio))
        cood_grid = grid(predict_map.shape[2], predict_map.shape[3], 1).unsqueeze(0) * self.cfg.LOSS.downsample_ratio + (self.cfg.LOSS.downsample_ratio / 2)
        cood_grid = cood_grid.type(torch.cuda.FloatTensor) / float(self.cfg.LOSS.crop_size)
        i = 0
        emd_loss = 0
        point_loss = 0
        pixel_loss = 0
        entropy = 0
        for p in points:
            p = p.cuda()
            if len(p) < 1:
                gt = torch.zeros((1, shape[1], shape[2])).cuda()
                point_loss += torch.abs(gt.sum() - predict_map[i].sum()) / shape[0]
                pixel_loss += torch.abs(gt.sum() - predict_map[i].sum()) / shape[0]
                emd_loss += torch.abs(gt.sum() - predict_map[i].sum()) / shape[0]
            else:
                gt = torch.ones((1, len(p), 1)).cuda()
                cood_points = p.reshape(1, -1, 2) / float(self.cfg.LOSS.crop_size) 
                A = predict_map[i].reshape(1, -1, 1)
                l, F, G = self.criterion(A, cood_grid, gt, cood_points)

                C = self.cost(cood_grid, cood_points)
                PI = torch.exp((F.repeat(1,1,C.shape[2])+G.permute(0,2,1).repeat(1,C.shape[1],1)-C).detach()/self.cfg.LOSS.blur**self.cfg.LOSS.p)*A*gt.permute(0,2,1)
                entropy += torch.mean((1e-20+PI) * torch.log(1e-20+PI))
                AE = PI
                AE = AE.sum(1).reshape(1,-1,1)
                emd_loss += (torch.mean(l) / shape[0])
                if self.cfg.LOSS.d_point == 'l1':
                    point_loss += torch.sum(torch.abs(PI.sum(1).reshape(1,-1,1)-gt)) / shape[0] 
                else:
                    point_loss += torch.sum((PI.sum(1).reshape(1,-1,1)-gt)**2) / shape[0] 
                if self.cfg.LOSS.d_pixel == 'l1':
                    pixel_loss += torch.sum(torch.abs(PI.sum(2).reshape(1,-1,1).detach()-A)) / shape[0] 
                else:
                    pixel_loss += torch.sum((PI.sum(2).reshape(1,-1,1).detach()-A)**2) / shape[0] 
            i += 1

        counting_loss = emd_loss + self.cfg.LOSS.tau*(pixel_loss + point_loss) + self.cfg.LOSS.blur*entropy

        return counting_loss, 0

class L2Loss(nn.Module):
    def __init__(self, cfg):
        super(L2Loss, self).__init__()
        self.reduction = 'mean'
        self.downsample_ratio = cfg.LOSS.downsample_ratio
        self.kernel = torch.ones(1, 1, cfg.LOSS.downsample_ratio, cfg.LOSS.downsample_ratio).to(cfg.TRAIN.device)

    def forward(self, outputs, density_map, points):
        outputs = outputs['density_map']
        if self.downsample_ratio > 1:
            density_map = F.conv2d(density_map, self.kernel, stride=self.downsample_ratio)
        loss = F.mse_loss(outputs, density_map, reduction=self.reduction)
        return loss, 0

def get_loss(cfg):
    # get counting loss
    
    if cfg.LOSS.name == 'GLoss':
        loss = GLoss(cfg)
    else:
        loss = L2Loss(cfg)
    
    return loss

if __name__ == '__main__':
    pass
    
    