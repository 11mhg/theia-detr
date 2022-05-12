import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
import utils



class HungarianMatcher(nn.Module):
    def __init__(self, class_weight = 1.0, bbox_weight = 1.0, giou_weight = 1.0):
        super().__init__()
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
    
    @torch.no_grad()
    def forward(self, pred, gt):
        """

        Args:
            pred (dict): 
                -   'pred_logits' - [batch_size, num_queries, num_classes+1]
                -   'pred_boxes' - [batch_size, num_queries, 4]
            gt (list): list of dicts len(gt) = batch_size
                -   'labels' - [num_target_boxes] classes
                -   'boxes'  - [num_target_boxes, 4] boxes
        """
        bs, num_queries = pred['pred_logits'].shape[:2]
        
        pred_prob = pred['pred_logits'].flatten(0, 1).softmax(-1)
        pred_bbox = pred['pred_boxes'].flatten(0, 1)
        
        tgt_ids = torch.cat([v['labels'] for v in gt], dim=0)
        tgt_bbox = torch.cat([v['boxes'] for v in gt], dim=0)
        
        class_loss = -pred_prob[:, tgt_ids]
        
        bbox_loss = torch.cdist( pred_bbox, tgt_bbox, p=1 ) # l1 norm
        
        giou, _= utils.box_giou( utils.box_cxcywh_to_xyxy(pred_bbox), utils.box_cxcywh_to_xyxy(tgt_bbox) )
        
        giou_loss = 1. - giou
        
        C = self.bbox_weight * bbox_loss + self.class_weight * class_loss + self.giou_weight * giou_loss
        C = torch.reshape( C, [bs, num_queries, -1] ).detach().cpu()
        
        sizes = [len(v["boxes"]) for v in gt]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    

    @torch.no_grad()
    def prep_gt(self, targets, num_boxes):
        out = []
        for batch_ind in range(targets.shape[0]):
            target = targets[batch_ind]
            n_box = num_boxes[batch_ind, 0]          
            target = target[:n_box]
            
            label = target[:,  4].long()
            boxes = target[:, :4].float()
            
            d = {}
            d['labels'] = label
            d['boxes'] = boxes

            out.append(d)
        
        return out