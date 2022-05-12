import os, sys, cv2
import torch
import torchvision as tv
import pytorch_lightning as pl
from tqdm import tqdm
from pycocotools.coco import COCO
import numpy as np
import imgaug as ia

from matplotlib import pyplot as plt
import imgaug.augmenters as iaa
from bbaug.augmentations import augmentations
from bbaug.policies import policies


current_file_directory = os.path.dirname(os.path.realpath(__file__))
coco_labels_file = os.path.join( current_file_directory, 'coco-labels.txt')
coco_labels = []
with open(coco_labels_file, 'r') as f:
    lines = f.readlines()
    coco_labels = [line.strip() for line in lines]


class COCODataset(tv.datasets.CocoDetection):
    
    def __init__(self, root, annFile, augment=False, max_image_size=512):
        super().__init__(root=root, annFile=annFile)
        self.augment = augment
        self.max_image_size = max_image_size
        self.to_tensor = tv.transforms.ToTensor()
        self.policy_container = None
        self.cat_ids = [0] + self.coco.getCatIds()
        if self.augment:
            aug_policy = policies.policies_v3()
            self.policy_container  = policies.PolicyContainer(aug_policy)
            
    
    def __getitem__(self, index):
        image, annotations = super().__getitem__(index)
        
        image = np.array(image, dtype=np.uint8)
        
        targets = []
        for obj in annotations:
            box = obj['bbox']
            cat_id = obj['category_id']
            label_ind = self.cat_ids.index( cat_id )
            idx = obj['id']
            x0 = box[0]
            y0 = box[1]
            w = box[2]
            h = box[3]
            x1 = x0 + w
            y1 = y0 + h
            
            targets.append(
                [ x0, y0, x1, y1, label_ind ]
            )
        
        targets = np.array(targets, dtype=np.float32)
        targets = np.reshape( targets, [-1, 5])
        
        if self.augment:
            random_policy = self.policy_container.select_random_policy()
            img_aug, bbs_aug = self.policy_container.apply_augmentation(
                random_policy,
                image,
                targets[:,:4],
                targets[:, 4]
            )
            
            if bbs_aug.size > 0:
                targets = targets[:bbs_aug.shape[0]]
                targets[:, :4] = bbs_aug[:, 1:]
                targets[:,  4] = bbs_aug[:, 0]
                image = img_aug
        
        h, w, _ = image.shape

        targets[:, [0, 2]] /= w
        targets[:, [1, 3]] /= h
        
        targets = np.reshape( targets, [-1, 5] )
        
        image = cv2.resize(image, (self.max_image_size, self.max_image_size))
        image = (image.astype(np.float32) / 255.0)
        
        targets = self.to_tensor(targets)
        image = self.to_tensor(image)
        
        return image, targets

    
def get_dataset(data="train", dtype=torch.float32, max_image_size=256):
    dtype_to_filepaths = {
        'train': {
            'root': '/mnt/e/Datasets/coco/images/train2017',
            'ann': '/mnt/e/Datasets/coco/annotations/instances_train2017.json'
        },
        'val': {
            'root': '/mnt/e/Datasets/coco/images/val2017',
            'ann': '/mnt/e/Datasets/coco/annotations/instances_val2017.json'
        },
    }
    
    filepaths = dtype_to_filepaths[data]

    coco_ds = COCODataset(
        root=filepaths['root'], 
        annFile=filepaths['ann'],
        augment = data == 'train', 
        max_image_size = max_image_size
    )
    
    return coco_ds

def collate_fn(batch):
    
    images = torch.stack([x[0] for x in batch], dim=0)
    max_boxes = max([x[1].shape[1] for x in batch])
    
    targets = []
    num_boxes = []
    for x in batch:
        target = x[1]
        padded_target = torch.nn.functional.pad(
            target, (0, 0, 0, max_boxes - target.shape[1]), 'constant', 0
        )
        targets.append(padded_target)
        num_boxes.append( target.shape[1] )
    targets = torch.cat(targets, dim=0)
    num_boxes = torch.tensor(num_boxes, dtype=torch.int64)
    num_boxes = torch.reshape(num_boxes, [-1, 1])
    
    return images, targets, num_boxes

class COCODataModule(pl.LightningDataModule):
    def __init__(self, image_size = 256):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        self.train_ds = get_dataset(data='train', dtype = torch.float32, max_image_size=self.hparams.image_size)
        self.val_ds = get_dataset(data='val', dtype = torch.float32, max_image_size=self.hparams.image_size)
        
    def train_dataloader(self, batch_size=4):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn = collate_fn
        )
    
    def val_dataloader(self, batch_size=4):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn = collate_fn
        )


if __name__ == "__main__":
    dataModule = COCODataModule()
    dataModule.setup()
    
    train_dl = dataModule.train_dataloader()
    for elem in train_dl:
        for item in elem:
            print(item.shape)
        break