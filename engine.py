"""
Training, evaluation and visualization functions
"""
import math
import os
from typing import Iterable
from PIL import Image
import numpy as np
import torch
import json
from tqdm import tqdm
import torch.nn.functional as F

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    loss_sum = 0
    loss_counting = 0
    loss_contrast = 0

    for idx, sample in enumerate(data_loader):
        img, patches, targets = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        patches['scale_embedding'] = patches['scale_embedding'].to(device)
        density_map = targets['density_map'].to(device)
        pt_map = targets['pt_map']

        outputs = model(img, patches, is_train=True)

        counting_loss, contrast_loss = criterion(outputs, density_map, pt_map)
        loss = counting_loss if isinstance(contrast_loss, int) else counting_loss + contrast_loss
        
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            continue
            
        loss_sum += loss_value
        loss_contrast += contrast_loss if isinstance(contrast_loss, int) else contrast_loss.item()
        loss_counting += counting_loss.item()      

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if (idx + 1) % 10 == 0:
            print('Epoch: %d, %d / %d, loss: %.8f, counting loss: %.8f, contrast loss: %.8f'%(epoch, idx+1,
                                                                                            len(data_loader),
                                                                                            loss_sum / (idx+1),
                                                                                            loss_counting / (idx+1),
                                                                                            loss_contrast / (idx+1)))

    return loss_sum / len(data_loader)

def test_norm(model,img,patches,avg_ratio,thres,scale_factor):
    if avg_ratio[0]*avg_ratio[1] <= thres:
        h,w = img.shape[-2:]
        img = F.interpolate(img,scale_factor=scale_factor,mode='bicubic')
        outputs =[]
        with torch.no_grad():
            for i in range(scale_factor):
                row_out = []
                for j in range(scale_factor):
                    row_out.append(model(img[...,i*h:(i+1)*h,j*w:(j+1)*w],patches,is_train=False))
                row_out = torch.cat(row_out,dim=-1)
                outputs.append(row_out)
        outputs = torch.cat(outputs,dim=-2)
    else:
        with torch.no_grad():
            outputs = model(img, patches, is_train=False)
    return outputs

@torch.no_grad()
def evaluate(cfg,model, data_loader, device, output_dir):
    nae = 0
    rse = 0
    mae = 0
    mse = 0
    model.eval()

    thres = cfg.VAL.test_norm_thres
    scale_factor = cfg.VAL.test_norm_scale_factor
    for idx, sample in enumerate(tqdm(data_loader)):
        img, patches, targets = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        avg_ratio = patches['scale_embedding'][0]
        gtcount = targets['gtcount']

        if thres and scale_factor:
            outputs = test_norm(model,img,patches,avg_ratio,thres,scale_factor)
        else:
            with torch.no_grad():
                outputs = model(img, patches, is_train=False)

        error = torch.abs(outputs.sum() - gtcount.item()).item()
        mae += error
        mse += error ** 2
        nae += error / gtcount.item()
        rse += error ** 2 / gtcount.item()

    mae = mae / len(data_loader)
    mse = mse / len(data_loader)
    mse = mse ** 0.5
    nae = nae / len(data_loader)
    rse = rse / len(data_loader)
    rse = rse ** 0.5
    
    with open(os.path.join(output_dir, 'result.txt'), 'a') as f:
        f.write('MAE %.2f, MSE %.2f, NAE %.2f, RSE %.2f \n'%(mae, mse,nae,rse))
    print('MAE %.2f, MSE %.2f, NAE %.2f, RSE %.2f \n'%(mae, mse,nae,rse))

    return mae, mse, nae, rse

@torch.no_grad()
def evaluate_carpk(model, data_loader, device, output_dir):
    nae = 0
    rse = 0
    mae = 0
    mse = 0
    model.eval()
    for idx, sample in enumerate(data_loader):
        img, refs, target = sample
        img = img.to(device)
        patches = {}
        patches['patches'] = refs.to(device)
        gtcount = target['gtcount']
        with torch.no_grad():
            outputs = model(img, patches, is_train=False)
        error = torch.abs(outputs.sum() - gtcount.item()).item()
        mae += error
        mse += error ** 2
        nae += error / gtcount.item()
        rse += error ** 2 / gtcount.item()
        
    mae = mae / len(data_loader)
    mse = mse / len(data_loader)
    mse = mse ** 0.5
    nae = nae / len(data_loader)
    rse = rse / len(data_loader)
    rse = rse ** 0.5
    
    with open(os.path.join(output_dir, 'result_carpk.txt'), 'a') as f:
        f.write('MAE %.2f, MSE %.2f, NAE %.2f, RSE %.2f \n'%(mae, mse,nae,rse))
    print('MAE %.2f, MSE %.2f, NAE %.2f, RSE %.2f \n'%(mae, mse,nae,rse))

    return mae, mse, nae, rse

@torch.no_grad()
def evaluate_mosaic(model, data_loader, device, output_dir):
    nae = 0
    rse = 0
    mae = 0
    mse = 0
    model.eval()
    for idx, sample in enumerate(tqdm(data_loader)):
        img = sample['img'].to(device)
        patches = {}
        patches['patches'] = sample['refs'].to(device)
        gtcount = len(sample['points'])
        with torch.no_grad():
            outputs = model(img, patches, is_train=False)
        error = torch.abs(outputs.sum() - gtcount).item()
        mae += error
        mse += error ** 2
        nae += error / gtcount
        rse += error ** 2 / gtcount
        
    mae = mae / len(data_loader)
    mse = mse / len(data_loader)
    mse = mse ** 0.5
    nae = nae / len(data_loader)
    rse = rse / len(data_loader)
    rse = rse ** 0.5
    
    with open(os.path.join(output_dir, 'mosaic_result.txt'), 'a') as f:
        f.write('MAE %.2f, MSE %.2f, NAE %.2f, RSE %.2f \n'%(mae, mse,nae,rse))
    print('MAE %.2f, MSE %.2f, NAE %.2f, RSE %.2f \n'%(mae, mse,nae,rse))

    return mae, mse, nae, rse

@torch.no_grad()
def visualize_mosaic(model, dataset, data_loader, device, output_dir):
    import matplotlib.pyplot as plt
    import cv2
    plt.switch_backend('agg')

    cmap = plt.cm.get_cmap('jet')
    visualization_dir = os.path.join(output_dir, 'mosaic_vis')
    if not os.path.exists(visualization_dir):
        os.mkdir(visualization_dir)
    
    model.eval()
    for idx, sample in tqdm(enumerate(data_loader)):
        img = sample['img'].to(device)
        patches = {}
        patches['patches'] = sample['refs'].to(device)
        bboxes = sample['bboxes']
        points = sample['points']
        gtcount = len(points)
        with torch.no_grad():
            outputs = model(img, patches, is_train=False)

        sample_id = dataset.data_list[idx]
        file_name = dataset.annos[sample_id]['img_name']
        file_path = dataset.data_path + 'images/' + file_name
        origin_img = Image.open(file_path).convert("RGB")
        origin_img = np.array(origin_img)
        h, w, _ = origin_img.shape

        cmap = plt.cm.get_cmap('jet')
        density_map = outputs
        density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()
        kernel = np.ones((9,9), np.uint8)
        density_map = cv2.dilate(density_map,kernel,iterations=1)

        density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0
        density_map = density_map[:,:,0:3] * 0.5 + origin_img * 0.5

        for p in points:
            origin_img = cv2.circle(origin_img,(int(p[0]),int(p[1])),3,(255,0,0),3)

        fig = plt.figure(dpi=800)
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        
        ax1.set_title(f'Ground Truth: {gtcount:.2f}')
        ax2.set_title(f'Prediction: {outputs.sum().item():.2f}')
        ax1.imshow(origin_img)
        ax2.imshow(density_map.astype(np.uint8))
        
        save_path = os.path.join(visualization_dir, os.path.basename(f'{sample_id}.png'))
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


@torch.no_grad()
def visualization(cfg, model, dataset, data_loader, device, output_dir):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    cmap = plt.cm.get_cmap('jet')
    visualization_dir = os.path.join(output_dir, 'visualizations')
    if not os.path.exists(visualization_dir):
        os.mkdir(visualization_dir)
    
    mae = 0
    mse = 0
    model.eval()
    for idx, sample in enumerate(data_loader):
        img, patches, targets = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        gtcount = targets['gtcount']
        gt_density = targets['density_map']
        with torch.no_grad():
            outputs = model(img, patches, is_train=False)
        error = torch.abs(outputs.sum() - gtcount.item()).item()
        mae += error
        mse += error ** 2

        # read original image
        file_name = dataset.data_list[idx][0]
        file_path = image_path = dataset.data_dir + 'images_384_VarV2/' + file_name
        origin_img = Image.open(file_path).convert("RGB")
        origin_img = np.array(origin_img)
        h, w, _ = origin_img.shape

        cmap = plt.cm.get_cmap('jet')
        density_map = outputs
        density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()
        gt_density = torch.nn.functional.interpolate(gt_density, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()

        density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0
        density_map = density_map[:,:,0:3] * 0.5 + origin_img * 0.5
        gt_density = cmap(gt_density / (gt_density.max()) + 1e-14) * 255.0
        gt_density = gt_density[:,:,0:3] * 0.5 + origin_img * 0.5

        fig = plt.figure(dpi=800)
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        
        ax1.set_title(str(gtcount.item()))
        ax2.set_title(str(outputs.sum().item()))
        ax1.imshow(gt_density.astype(np.uint8))
        ax2.imshow(density_map.astype(np.uint8))
        
        save_path = os.path.join(visualization_dir, os.path.basename(file_name))
        plt.savefig(save_path)
        plt.close()
        
    mae = mae / len(data_loader)
    mse = mse / len(data_loader)
    mse = mse ** 0.5
    
    with open(os.path.join(output_dir, 'result.txt'), 'a') as f:
        f.write('MAE %.2f, MSE %.2f \n'%(mae, mse))
    print('MAE %.2f, MSE %.2f \n'%(mae, mse))

    return mae, mse
