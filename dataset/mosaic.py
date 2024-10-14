from torch.utils.data import Dataset
import json
from PIL import Image
import os
import torch
from torchvision import transforms
import torchvision.transforms.functional as TVF
import numpy as np

class MOSAIC_DATASET(Dataset):
    def __init__(self, examplar_number, data_path, source_path,query_transform,ref_transform):
        super().__init__()
        self.examplar_number = examplar_number
        self.data_path = data_path
        self.source_path = source_path
        self.query_transform = query_transform
        self.ref_transform = ref_transform
        
        with open(os.path.join(self.data_path,'mosaic_annotations.json'),'r') as f:
            self.annos = json.load(f)

        with open(os.path.join(self.source_path,'annotation_FSC147_384.json'),'r') as f:
            self.source_annos = json.load(f)
        
        self.data_list = list(self.annos.keys())

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index) :
        
        sample_id = self.data_list[index]
        anno = self.annos[sample_id]
        
        source_name = anno['source_name']
        source_img = Image.open(os.path.join(self.source_path,'images_384_VarV2',source_name))
        source_anno = self.source_annos[source_name]
        boxes = np.array(source_anno['box_examples_coordinates'])
        boxes = boxes[:self.examplar_number, :, :]
        patches = []
        bboxes = []

        for box in boxes:
            x1, y1 = box[0].astype(np.int32)
            x2, y2 = box[2].astype(np.int32)
            patch = source_img.crop((x1, y1, x2, y2))
            bboxes.append((x1, y1, x2, y2))
            patches.append(self.ref_transform(patch))

        img = Image.open(os.path.join(self.data_path,'images',anno['img_name'])).convert('RGB')
        img = self.query_transform(img)

        points = anno['points']
        target_class = anno['class']

        return {'img': img,'refs': torch.stack(patches),'points':points, 'bboxes': bboxes,'target_class':target_class}

def pad_to_constant(inputs, psize):
    h, w = inputs.size()[-2:]
    ph, pw = (psize-h%psize)%psize,(psize-w%psize)%psize

    if ph or pw:
        tmp_pad = [0, pw, 0, ph]
        inputs = torch.nn.functional.pad(inputs, tmp_pad)
    
    return inputs

def pad_ratio(img):
    _, h, w = img.size()
    
    max_size = max(w,h)
    pw , ph = (max_size - w)//2, (max_size -h)//2

    new_image = TVF.pad(img,[pw,ph])

    return new_image

class MainTransform(object):
    def __init__(self):
        self.img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __call__(self, img):
        img = self.img_trans(img)
        
        img = pad_to_constant(img, 32)
        
        return img

class RefTransform(object):
    def __init__(self, exemplar_size):
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

def build_mosaic_dataset(cfg):
    query_transform = MainTransform()
    ref_transform = RefTransform(cfg.DATASET.exemplar_size)
    return MOSAIC_DATASET(cfg.DATASET.exemplar_number,cfg.DIR.dataset,cfg.DIR.source_dataset,query_transform,ref_transform)

if __name__ == "__main__":
    from torchvision.utils import save_image
    from torch.utils.data import DataLoader
    import cv2
    import matplotlib.pyplot as plt

    query_transform = MainTransform()
    ref_transform = RefTransform((128,128))
    data_path = '/home/tvsrt0p1c/data/mosaic_eval_test'
    source_path = '/home/tvsrt0p1c/data/FSC147'
    mosaic_dataset = MOSAIC_DATASET(3,data_path,source_path,query_transform,ref_transform)
    dataloader = DataLoader(mosaic_dataset,batch_size=1,shuffle=False)

    for sample in dataloader:
        save_image(sample['img'],'img.png')
        save_image(sample['refs'].squeeze(),'refs.png')

        plt_img = sample['img'].squeeze().permute(1,2,0).cpu().numpy().copy()
        for p in sample['points']:
            plt_img = cv2.circle(plt_img,(int(p[0]),int(p[1])),2,(255,0,0),2)
        plt.imshow(plt_img)
        plt.savefig('plt_img.png')
        import ipdb;ipdb.set_trace()