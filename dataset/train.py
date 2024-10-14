"""
FSC-147 dataset
The exemplar boxes are sampled and resized to the same size
"""
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import torch
import numpy as np
from torchvision.transforms import transforms
import random
from collections import defaultdict
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def get_image_classes(class_file,data_list):
    name_class_dict = dict()
    class_name_dict = defaultdict(list)
    with open(class_file, 'r') as f:
        classes = [line.split('\t') for line in f.readlines()]

    for entry in classes:
        img_name = entry[0]
        class_name = entry[1].strip()
        if img_name in data_list:
            name_class_dict[img_name] = class_name
            class_name_dict[class_name].append(img_name)
    
    return name_class_dict, class_name_dict 

def batch_collate_fn(batch):
    batch = list(zip(*batch))
    batch[0], scale_embedding, batch[2] = batch_padding(batch[0], batch[2])
    patches = torch.stack(batch[1], dim=0)
    batch[1] = {'patches': patches,'scale_embedding':scale_embedding}
    return tuple(batch)

def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def batch_padding(tensor_list, target_dict):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        density_shape = [len(tensor_list)] + [1, max_size[1], max_size[2]]
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        density_map = torch.zeros(density_shape, dtype=dtype, device=device)
        pt_map = torch.zeros(density_shape, dtype=dtype, device=device)
        gtcount = []
        scale_embedding = []
        for idx, package  in enumerate(zip(tensor_list, tensor, density_map, pt_map)):
            img, pad_img, pad_density, pad_pt_map = package
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            pad_density[:, : img.shape[1], : img.shape[2]].copy_(target_dict[idx]['density_map'])
            pad_pt_map[:, : img.shape[1], : img.shape[2]].copy_(target_dict[idx]['pt_map'])
            gtcount.append(target_dict[idx]['gtcount']) 
            scale_embedding.append(target_dict[idx]['scale_embedding'])
        pt_map = pt_map.transpose(-2,-1)
        pt_map = pt_map.squeeze(1)

        scale_embedding = torch.stack(scale_embedding)
        points = []
        for p in pt_map:
            points.append(p.nonzero())
        target = {'density_map': density_map,
                  'pt_map': points,
                  'gtcount': torch.tensor(gtcount)}
    else:
        raise ValueError('not supported')
    return tensor, scale_embedding, target

def random_crop(img, target, crop_size):
    w,h = img.size
    crop_h = crop_size[0]
    crop_w = crop_size[1]
    res_h = h - crop_h
    res_w = w - crop_w
    top = random.randint(0, res_h)
    left = random.randint(0, res_w)
    img = TF.crop(img, top,left,crop_h,crop_w)
    target['pt_map'] = target['pt_map'][top:top+crop_h,left:left+crop_w]
    target['gt_count'] = np.transpose(target['pt_map'].nonzero()).shape[0]
    target['density_map'] = target['density_map'][top:top+crop_h,left:left+crop_w]

    return img,target

class FSC147Dataset(Dataset):
    def __init__(self, data_dir, data_list, scaling, box_number=3, min_size=384, max_size=1584, preload=False, main_transform=None, ref_transform=None):
        self.data_dir = data_dir
        self.data_list = [name.split('\t')[0] for name in open(data_list).read().splitlines()]
        self.scaling = scaling
        self.box_number = box_number
        self.preload = preload
        self.main_transform = main_transform
        self.ref_transform = ref_transform
        self.min_size = min_size
        self.max_size = max_size 
        
        # load annotations for the entire dataset
        annotation_file = os.path.join(self.data_dir, 'annotation_FSC147_384.json')
        image_classes_file = os.path.join(self.data_dir, 'ImageClasses_FSC147.txt')
                
        self.image_classes,self.class_name_dict = get_image_classes(image_classes_file,self.data_list)

        with open(annotation_file) as f:
            self.annotations = json.load(f)

        # store images and generate ground truths
        self.images = {}
        self.targets = {}
        self.patches = {}

    def __len__(self):
        return len(self.data_list)

    def get_sample(self,file_name,crop_size):
        image_path = os.path.join(self.data_dir, 'images_384_VarV2/' + file_name)
        density_path = os.path.join(self.data_dir, 'gt_density_map_adaptive_384_VarV2/' + file_name.replace('jpg', 'npy'))
        
        img_info = self.annotations[file_name]
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        # resize the image
        r = 1.0
        if h > self.max_size or w > self.max_size:
            r = self.max_size / max(h, w)
        if r * h < self.min_size or w*r < self.min_size:
            r = self.min_size / min(h, w)
        nh, nw = int(r*h), int(r*w)
        img = img.resize((nw, nh), resample=Image.BICUBIC)
    
        density_map = np.load(density_path).astype(np.float32)
        pt_map = np.zeros((nh, nw), dtype=np.int32)
        points = (np.array(img_info['points']) * r).astype(np.int32)
        boxes = np.array(img_info['box_examples_coordinates']) * r   
        boxes = boxes[:self.box_number, :, :]
        gtcount = points.shape[0]
        
        # crop patches and data transformation
        target = dict()
        patches = []
        
        if points.shape[0] > 0:     
            points[:,0] = np.clip(points[:,0], 0, nw-1)
            points[:,1] = np.clip(points[:,1], 0, nh-1)
            pt_map[points[:, 1], points[:, 0]] = 1 
            for box in boxes:
                x1, y1 = box[0].astype(np.int32)
                x2, y2 = box[2].astype(np.int32)
                patch = img.crop((x1, y1, x2, y2))
                patches.append(patch)
        
        target['density_map'] = density_map * self.scaling
        target['pt_map'] = pt_map
        target['gtcount'] = gtcount

        if random.random() > 0.5:
            img = TF.hflip(img)
            target['pt_map'] = np.fliplr(target['pt_map']).copy()
            target['density_map'] = np.fliplr(target['density_map']).copy()

        crop_img, crop_target = random_crop(img,target,crop_size)

        return {'img':crop_img,'patches':patches,'target':crop_target}

    def __getitem__(self, idx):
        file_name = self.data_list[idx]

        if file_name in self.images:
            img = self.images[file_name]
            target = self.targets[file_name]
            patches = self.patches[file_name]        
        else:
            aug_flag=False
            if random.random() < 0.4:
                aug_flag = True

            if aug_flag:
                copy_class_list = list(self.class_name_dict.keys())
                target_class = self.image_classes[file_name]
                copy_class_list.remove(target_class)
                sample_classes = random.sample(copy_class_list,3)

                sample_names = []
                for s in sample_classes:
                    sample_names.append(random.sample(self.class_name_dict[s],1))
                
                target_size = 384
                crop_size = 192
                sample1 = self.get_sample(file_name,(crop_size,crop_size))
                sample2 = self.get_sample(sample_names[0][0],(crop_size,crop_size))
                sample3 = self.get_sample(sample_names[1][0],(crop_size,crop_size))
                sample4 = self.get_sample(sample_names[2][0],(crop_size,crop_size))
                img = Image.new('RGB',(target_size,target_size))

                samples = [sample1,sample2,sample3,sample4]
                target_idx = random.randrange(0,4)
                target_sample = samples[target_idx]
                
                img.paste(samples[0]['img'],(0,0))
                img.paste(samples[1]['img'],(crop_size,0))
                img.paste(samples[2]['img'],(0,crop_size))
                img.paste(samples[3]['img'],(crop_size,crop_size))
                target = {}
                target['pt_map'] = np.zeros((target_size,target_size))
                target['density_map'] = np.zeros((target_size,target_size))

                target_target = target_sample['target']
                target['gtcount'] = target_target['gtcount']
                top_idx = target_idx // 2
                left_idx = target_idx % 2
                target['pt_map'][top_idx*crop_size:(top_idx+1)*crop_size,left_idx*crop_size:(left_idx+1)*crop_size] = target_target['pt_map']
                target['density_map'][top_idx*crop_size:(top_idx+1)*crop_size,left_idx*crop_size:(left_idx+1)*crop_size] = target_target['density_map']
                patches = target_sample['patches']
            else:
                crop_size = 384
                sample = self.get_sample(file_name,(crop_size,crop_size))
                img,patches,target = sample['img'],sample['patches'],sample['target']
            
            scale_embedding = []
            tensor_patches = []
            for patch in patches:
                pw,ph = patch.size
                tensor_patches.append(self.ref_transform(patch))
                scale_embedding.append([ph/crop_size,pw/crop_size])

            tensor_patches = torch.stack(tensor_patches)
            scale_embedding = torch.tensor(scale_embedding)
            img, target = self.main_transform(img, target)
            target['scale_embedding'] = scale_embedding
           
            if self.preload:
                self.images.update({file_name: img})
                self.patches.update({file_name: tensor_patches})
                self.targets.update({file_name: target})
            
        return img, tensor_patches, target

def pad_to_constant(inputs, psize):
    h, w = inputs.size()[-2:]
    ph, pw = (psize-h%psize)%psize,(psize-w%psize)%psize

    if ph or pw:
        inputs = F.pad(inputs, [0, pw, 0, ph])
    
    return inputs
    

class MainTransform(object):
    def __init__(self):
        self.img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __call__(self, img, target):
        img = self.img_trans(img)
        density_map = target['density_map']
        pt_map = target['pt_map']
        pt_map = torch.from_numpy(pt_map).unsqueeze(0)
        density_map = torch.from_numpy(density_map).unsqueeze(0)
        
        img = pad_to_constant(img, 32)
        density_map = pad_to_constant(density_map, 32)
        pt_map = pad_to_constant(pt_map, 32)
        target['density_map'] = density_map.float()
        target['pt_map'] = pt_map.float()
        
        return img, target


def pad_ratio(img):
    _, h, w = img.size()
    
    max_size = max(w,h)
    half_pw , half_ph = (max_size - w)//2, (max_size -h)//2

    new_image = F.pad(img,[half_pw,half_pw,half_ph,half_ph])

    return new_image

class RefTransform(object):
    def __init__(self,is_train, exemplar_size):
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.resize = transforms.Resize(exemplar_size)

    def __call__(self, img) :
        img = self.trans(img)
        img = pad_ratio(img)
        img = self.resize(img)

        return img

def build_dataset(cfg, is_train):
    main_transform = MainTransform()
    ref_transform = RefTransform(is_train, cfg.DATASET.exemplar_size)
    if is_train: 
        data_list = cfg.DATASET.list_train
    else:
        if not cfg.VAL.evaluate_only:
            data_list = cfg.DATASET.list_val 
        else:
            data_list = cfg.DATASET.list_test
    
    dataset = FSC147Dataset(data_dir=cfg.DIR.dataset,
                            data_list=data_list,
                            scaling=1.0,
                            box_number=cfg.DATASET.exemplar_number,
                            main_transform=main_transform,
                            ref_transform=ref_transform)
    
    return dataset
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    import matplotlib.pyplot as plt
    main_transform = MainTransform()
    query_transform = RefTransform(is_train=True, exemplar_size=(128, 128))
    
    dataset = FSC147Dataset(data_dir='/home/tvsrt0p1c/data/FSC147/',
                            data_list='/home/tvsrt0p1c/data/FSC147/train.txt',
                            scaling=1.0,
                            main_transform=main_transform,
                            ref_transform=query_transform)
    
    data_loader = DataLoader(dataset, batch_size=1,shuffle=False, collate_fn=batch_collate_fn)
    
    for idx, sample in enumerate(data_loader):
        img, patches, targets = sample
        
        save_image(img[0],'img.png')
        save_image(patches['patches'][0],'patches.png')
        plt.imshow(targets['density_map'][0].squeeze().detach().cpu().numpy())
        print(patches['scale_embedding'][0])
        plt.savefig('density.png')
        
        import ipdb; ipdb.set_trace()
    