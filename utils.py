import os, sys
import torch
import torchvision as tv
import cv2
import numpy as np
from matplotlib import pyplot as plt

from dataset import coco_labels



def box_cxcywh_to_xyxy(box):
    """
    Convert bounding box from center-size to xyxy format.
    :param box: bounding box in center-size format
    :return: bounding box in xyxy format
    """
    x_c, y_c, w, h = box[...,0], box[...,1], box[...,2], box[...,3] 
    b = [
        (x_c - (w*0.5)), (y_c - (h*0.5)),
        (x_c + (w*0.5)), (y_c + (h*0.5)),
    ]
    return torch.stack(b, dim=-1)



def box_iou(b1, b2):
    """returns the iou between the set of boxes 1 and boxes 2
    
    assumes that b1 and b2 are in xyxy format

    Args:
        b1 (torch.Tensor): first set of boxes
        b2 (torch.Tensor): second set of boxes
    Returns:
        tuple(torch.Tensor, torch.Tensor): the iou and the union of the boxes in [N, M] format
    """
    
    area1 = (b1[...,2] - b1[...,0]) * (b1[...,3] - b1[...,1])
    area2 = (b2[...,2] - b2[...,0]) * (b2[...,3] - b2[...,1])
    
    lt = torch.max(b1[:, None, :2], b2[:, :2]) # [N,M,2]
    rb = torch.min(b1[:, None, 2:], b2[:, 2:]) # [N,M,2]
    
    wh = (rb - lt).clamp(min=0) # [N,M,2]
    
    inter = wh[...,0] * wh[...,1] # [N,M]
    union = area1[:, None] + area2 - inter # [N,1]
    
    iou = inter / (union + 1e-6)
    return iou, union


def box_giou(b1, b2):
    """returns the giou between the set of boxes 1 and boxes 2
    
    assumes that b1 and b2 are in xyxy format

    Args:
        b1 (torch.Tensor): first set of boxes
        b2 (torch.Tensor): second set of boxes
    Returns:
        tuple(torch.Tensor, torch.Tensor): the giou and iou of the boxes in [N, M] format
    """
    
    
    iou, union = box_iou(b1, b2)
    
    lt = torch.min(b1[:, None, :2], b2[:, :2]) # [N,M,2]
    rb = torch.max(b1[:, None, 2:], b2[:, 2:]) # [N,M,2]
    
    wh = (rb - lt).clamp(min=0) # [N,M,2]
    area = wh[...,0] * wh[...,1] # [N,M]
    
    giou = iou - (area - union) / (area + 1e-6)
    
    return giou, iou