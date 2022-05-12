import torch
import torchvision as tv

class LearnedPositionalEmbedding(torch.nn.Module):
    def __init__(self, in_N, out_N=256):
        super().__init__()
        
        self.row_embeddings = torch.nn.Embedding(in_N, out_N)
        self.col_embeddings = torch.nn.Embedding(in_N, out_N)
    
    def forward(self, x):
        bs = x.shape[0]
        h, w = x.shape[-2:]
        
        i = torch.arange(h).to(x.device) # [0, 1, 2, ..., h-1]
        j = torch.arange(w).to(x.device)  # [0, 1, 2, ..., w-1]
        
        x_emb = self.row_embeddings(i) #[ w, out_N ]
        y_emb = self.col_embeddings(j) #[ h, out_N ]
        
        x_emb = x_emb[None, :   , :].repeat( h, 1, 1) # [ h, w, out_N ]
        y_emb = y_emb[:   , None, :].repeat( 1, w, 1) # [ h, w, out_N ]
        
        xy_emb = torch.cat([ x_emb, y_emb ], dim=-1) # [ h, w, 2*out_N ]
        
        pos = xy_emb.permute( 2, 0, 1)[None, :, :, :].repeat(bs, 1, 1, 1) # [ bs, 2*out_N, h, w ]
        
        return pos
    

class PositionalBackBone(torch.nn.Module):
    def __init__(self, positional_emb_size=256, train_backbone = True):
        super().__init__()
        self.positional_emb_size = positional_emb_size
        
        self.backbone = tv.models.efficientnet_b3(pretrained=True)
        

        for _, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(train_backbone)
        
        
        self.embedding_size = self.backbone.classifier[1].in_features
        self.pos_emb = LearnedPositionalEmbedding(in_N=self.embedding_size, out_N = self.positional_emb_size )
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.backbone.features(x)
        pos = self.pos_emb(x).to(dtype=x.dtype)
        return {
            'features': x, 
            'pos': pos
        }