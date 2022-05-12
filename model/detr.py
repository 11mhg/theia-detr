
import torch
from torch import nn
from torch.nn import functional as F
import torchvision as tv
import pytorch_lightning as pl

from model.backbone import PositionalBackBone
from model.transformer import Transformer
from model.mlp import SimpleMLP

import utils


class DETR(nn.Module):
    def __init__(self, num_classes=80, num_queries = 100, hidden_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        
        self.backbone = PositionalBackBone(positional_emb_size=self.hidden_dim//2)
        self.input_project = nn.Conv2d(1536, self.hidden_dim, kernel_size=1, stride=1, padding=0)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        
        self.transformer = Transformer(d_model=self.hidden_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                    dim_feedforward=2048, dropout=0.1, activation=F.relu)
        
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes+1)
        self.box_embed = SimpleMLP(self.hidden_dim, self.hidden_dim, 4, 3, activation=F.relu)
        
        
    def forward(self, images):
        backbone_info = self.backbone( images )
        backbone_info['features'] = self.input_project(backbone_info['features'])
                
        out, memory = self.transformer( backbone_info['features'], self.query_embed.weight, 
                        backbone_info['pos'] )
        
        pred_logits = self.class_embed(out)
        pred_boxes = self.box_embed(out).sigmoid()
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }
    
    @torch.no_grad()
    def postprocess(self, outputs, sizes):
        out_logits, out_boxes = outputs['pred_logits'], outputs['pred_boxes']
        
        class_probs = F.softmax(out_logits, -1)
        class_scores, class_labels = class_probs[..., :-1].max(-1)
        
        boxes = utils.box_cxcywh_to_xyxy(out_boxes)
        
        img_h, img_w = sizes[:, 0], sizes[:, 1]
        
        boxes_scale = torch.stack([img_w, img_h, img_w, img_h], dim=-1)
        boxes = boxes * boxes_scale[:, None, :]
        
        final = []
        for batch_ind in range(sizes.shape[0]):
            s = class_scores[batch_ind]
            l = class_labels[batch_ind]
            b = boxes[batch_ind]
            
            final.append({
                'scores': s,
                'labels': l,
                'boxes': b
            })
            
        return final