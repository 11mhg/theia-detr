import torch
from torch import nn
from torch.nn import functional as F
from loss.hungarian_matcher import HungarianMatcher
import pytorch_lightning as pl

import utils


class DETR_Losses(pl.LightningModule):
    
    def __init__(self, num_classes, ce_weight=1., bbox_weight=5., giou_weight=2., eos_coef=0.1, topk=(1, 5, 10)):
        super().__init__()
        self.save_hyperparameters()
        
        self.matcher = HungarianMatcher(class_weight = self.hparams.ce_weight, bbox_weight = self.hparams.bbox_weight, giou_weight = self.hparams.giou_weight)
        
        empty_weight = torch.ones(self.hparams.num_classes+1)
        empty_weight[-1] = self.hparams.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def prep_gt(self, targets, num_boxes):
        return self.matcher.prep_gt(targets, num_boxes)
    
    def get_perm_idx(self, indices, tgt=False):
        
        if not tgt:
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
        else:
            batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
            src_idx = torch.cat([tgt for (_, tgt) in indices])
        
        return batch_idx, src_idx
    
    @torch.no_grad()
    def accuracy(self, pred, target):
        
        maxk = max(self.hparams.topk)
        bs = target.size(0)
    
        _, pred_topk = pred.topk(maxk, 1, True, True)
        pred_topk = pred_topk.t()
        
        correct = pred_topk.eq(target.reshape(1, -1).expand_as(pred_topk))
        
        res = []
        for k in self.hparams.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * (100.0 / bs)).item())
        return res
        
    
    def forward(self, preds, targets):
        
        indices = self.matcher(preds, targets) # [batch_size] of tuples (left ind, right ind)
        
        src_idx = self.get_perm_idx(indices)
        tgt_idx = self.get_perm_idx(indices, tgt=True)
        
        #get number of boxes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.tensor([num_boxes], dtype=torch.float, device=preds['pred_logits'].device)
        
        
        
        # class_loss
        pred_logits = preds['pred_logits']
        
        #loop through and get the matching target class
        target_classes_o = []
        for t, (_, J) in zip(targets, indices):
            target_classes_o.append( t['labels'][J] )
        target_classes_o = torch.cat(target_classes_o)
        target_classes = torch.full( pred_logits.shape[:2], self.hparams.num_classes, dtype=torch.long, device=pred_logits.device )
        
        target_classes[src_idx] = target_classes_o
        
        class_loss = F.cross_entropy( pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        # get accuracy
        accuracy = self.accuracy(pred_logits[src_idx], target_classes_o)        
        
        # bbox_loss
        pred_boxes  = preds['pred_boxes']
        src_pred_boxes = pred_boxes[src_idx]
        
        target_boxes = []
        for t, (_, J) in zip(targets, indices):
            target_boxes.append(
                t['boxes'][J]
            )
        target_boxes = torch.cat(target_boxes, dim=0)
        
        bbox_loss = F.smooth_l1_loss(src_pred_boxes, target_boxes, reduction='none')
        bbox_loss = bbox_loss.sum() / num_boxes
        
        # giou_loss
        
        giou, _ = utils.box_giou(
            utils.box_cxcywh_to_xyxy(src_pred_boxes),
            utils.box_cxcywh_to_xyxy(target_boxes)
        )
        
        giou = torch.diag( giou ).mean()
        giou_loss = 1. - giou
        
        losses = {
            'class_loss': class_loss,
            'bbox_loss': bbox_loss,
            'giou_loss': giou_loss,
            'accuracy': accuracy
        }
        
        return losses