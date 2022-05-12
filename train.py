#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('load_ext', 'autoreload')


# # In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')
import os, sys
import torch
from torch import nn
from torch.nn import functional as F
import torchvision as tv

import pytorch_lightning as pl
import cv2
import numpy as np
from matplotlib import pyplot as plt
from model.detr import DETR
from dataset import COCODataModule, coco_labels

from loss.losses import DETR_Losses

pl.seed_everything(777)


# In[2]:


batch_size = 48
hidden_dim = 512
num_epochs = 100
num_classes = len(coco_labels)
num_queries = 100
learning_rate = 1e-4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16


# In[3]:


# for elem in train_dl:
#     images, targets, num_boxes = elem
#     imgs = images.permute(0, 2, 3, 1).numpy()
#     targs = targets.numpy()
#     n_boxes = num_boxes.numpy()
    
#     for batch_ind in range(imgs.shape[0]):
#         img = (imgs[batch_ind] * 255.).astype(np.uint8)
#         targ = targs[batch_ind]
#         n_box = n_boxes[batch_ind][0]
        
#         valid_targ = targ[:n_box]
        
#         h, w = img.shape[:2]
        
#         image_size = np.array([w, h, w, h])
        
#         for box in valid_targ:
#             label_ind = box[4].astype(np.int32)
#             label = coco_labels[label_ind-1]
#             box = box[:4]
            
#             box = (box * image_size).astype(np.int32)
#             x0, y0, x1, y1 = box
            
#             color = (36, 255, 12)
#             img = cv2.rectangle( img.copy(), (int(x0), int(y0)), (int(x1), int(y1)), color, int(2) )
#             cv2.putText(img, label, (box[0], max(10, box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
#         plt.imshow(img)
#         plt.show()
#     print(imgs.shape, targs.shape, n_boxes.shape)
#     break


# In[8]:


class DETR_ObjectDetector(pl.LightningModule):
    
    def __init__(self, num_classes=num_classes, num_queries = num_queries, hidden_dim=hidden_dim,
                    batch_size=4, lr=1e-4, lr_backbone=1e-5,weight_decay=1e-4, lr_drop=200, image_size=256,
                    clip_max_norm=0.1):
        
        super().__init__()
        self.save_hyperparameters()
        self.detr = DETR(num_classes=self.hparams.num_classes, num_queries = self.hparams.num_queries, hidden_dim=self.hparams.hidden_dim)
        self.detr_loss = DETR_Losses(num_classes = self.hparams.num_classes, ce_weight=1., bbox_weight=5., giou_weight=2., eos_coef=0.1)
    
    def forward(self, images):
        return self.detr(images)
    
    def configure_optimizers(self):
        all_backbone_params = []
        other_params = []
        for name, param in self.detr.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    all_backbone_params.append(param)
                else:
                    other_params.append(param)
        
        param_dicts = [
            {"params": other_params},
            {
                "params": all_backbone_params,
                "lr": self.hparams.lr_backbone,
            },
        ]
        
        optimizer = torch.optim.Adam(param_dicts, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_drop, gamma=0.1)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': lr_scheduler
        }
    
    def training_step(self, batch, batch_idx):
        images, targets, num_boxes = batch
        out = self(images)
        prepped_targets = self.detr_loss.prep_gt(targets, num_boxes)
        loss_dict = self.detr_loss(out, prepped_targets)
        
        total_loss = (loss_dict['class_loss'] * self.detr_loss.hparams.ce_weight) + \
                     (loss_dict['bbox_loss' ] * self.detr_loss.hparams.bbox_weight) + \
                     (loss_dict['giou_loss' ] * self.detr_loss.hparams.giou_weight)
        
        top_k_accuracy_logs = {}
        for i, k in enumerate(self.detr_loss.hparams.topk):
            top_k_accuracy_logs[f"top_{k}_accuracy"] = loss_dict['accuracy'][i]
            
        self.log(
            "train_performance", 
            top_k_accuracy_logs,
            on_step=True,
            on_epoch=True
        )
        
        self.log(
            "train_loss", {
            'class_loss': loss_dict['class_loss'],
            'bbox_loss': loss_dict['bbox_loss' ],
            'giou_loss': loss_dict['giou_loss' ],
            'total_loss': total_loss
            },
            on_step=True,
            on_epoch=True
        )
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets, num_boxes = batch
        out = self(images)
        prepped_targets = self.detr_loss.prep_gt(targets, num_boxes)
        loss_dict = self.detr_loss(out, prepped_targets)
        
        total_loss = (loss_dict['class_loss'] * self.detr_loss.hparams.ce_weight) + \
                     (loss_dict['bbox_loss' ] * self.detr_loss.hparams.bbox_weight) + \
                     (loss_dict['giou_loss' ] * self.detr_loss.hparams.giou_weight)
        
        top_k_accuracy_logs = {}
        for i, k in enumerate(self.detr_loss.hparams.topk):
            top_k_accuracy_logs[f"top_{k}_accuracy"] = loss_dict['accuracy'][i]
            
        self.log(
            "val_performance", 
            top_k_accuracy_logs,
            on_step=True,
            on_epoch=True
        )
        
        self.log(
            "val_loss", {
            'class_loss': loss_dict['class_loss'],
            'bbox_loss': loss_dict['bbox_loss' ],
            'giou_loss': loss_dict['giou_loss' ],
            'total_loss': total_loss
            },
            on_step=True,
            on_epoch=True                 
        )
        
        self.log(
            "total_val_loss", 
            total_loss,
            on_epoch=True
        )
        
        return total_loss
    
    ###############
    # Data stuff! #
    ###############
    
    def prepare_data(self):
        self.coco_data_module = COCODataModule(image_size=self.hparams.image_size)
        
    def setup(self, stage=None):
        self.coco_data_module.setup()

    def train_dataloader(self):
        train_dl = self.coco_data_module.train_dataloader(batch_size=self.hparams.batch_size)
        return train_dl
    
    def val_dataloader(self):
        val_dl = self.coco_data_module.val_dataloader(batch_size=self.hparams.batch_size)
        return val_dl
    
    

detr = DETR_ObjectDetector(num_classes=num_classes, num_queries = num_queries, hidden_dim=hidden_dim)
detr.configure_optimizers()
detr.load_from_checkpoint(checkpoint_path="Final_Checkpoint.ckpt")


# In[ ]:


trainer = pl.Trainer(
    default_root_dir=  os.path.join( os.getcwd(), 'model_artifacts/'),
    gpus = 1,
    max_epochs = 500,
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor='total_val_loss',save_top_k=3, mode='min'),
        pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    ],
    gradient_clip_val = detr.hparams.clip_max_norm,
    gradient_clip_algorithm='norm',
    auto_scale_batch_size="power",
    precision='bf16',
    log_every_n_steps=50,
    benchmark=True,
    accumulate_grad_batches=10,
    enable_progress_bar=True
)
#trainer.logger._log_graph = True
trainer.logger._default_hp_metric = None

#tuner = pl.tuner.tuning.Tuner(trainer)

#new_batch_size = tuner.scale_batch_size(detr)

detr.hparams.batch_size = batch_size


# In[7]:


trainer.fit(detr)
trainer.save_checkpoint("Final_Checkpoint.ckpt")


# In[ ]:




