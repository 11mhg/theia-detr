import torch
from torch import nn
from torch.nn import functional as F
import torchvision as tv

class CustomEncoderSequential(nn.Sequential):
    def forward(self, x):
        for module in self._modules.values():
            if type(x) == dict:
                out = module(**x)
                x['x'] = out
        return out
    
class CustomDecoderSequential(nn.Sequential):
    def forward(self, x):        
        for module in self._modules.values():
            out = module(**x)
            x['x'] = out
        return out            


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu):
        
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        
        
        self.l1 = nn.Linear(d_model, dim_feedforward)
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(dim_feedforward, d_model)
        self.d2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        self.activation = activation
    
    def forward(self, x, x_mask=None, x_key_padding_mask=None, pos=None):
        x_q_k = x if pos is None else x+pos
        
        attn, _ = self.self_attn(x_q_k, x_q_k, value=x, attn_mask=x_mask,
                    key_padding_mask=x_key_padding_mask)
        x = x + self.self_attn_dropout( attn )
        x = self.self_attn_norm(x)
        
        mlp = self.l2( self.d1( self.activation( self.l1(x) ) ) )
        x = x + self.d2( mlp )
        x = self.norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu):
        
        super().__init__()
        
        self.self_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_norm_1 = nn.LayerNorm(d_model)
        self.self_attn_dropout_1 = nn.Dropout(dropout)        
        
        self.self_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_norm_2 = nn.LayerNorm(d_model)
        self.self_attn_dropout_2 = nn.Dropout(dropout)

        self.l1 = nn.Linear(d_model, dim_feedforward)
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(dim_feedforward, d_model)
        self.d2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        self.activation = activation

    def forward(self, x, memory, x_mask=None, memory_mask=None, pos = None, query_pos = None):
        x_q_k = x if query_pos is None else x+query_pos
        
        attn_1, _ = self.self_attn_1(x_q_k, x_q_k, value=x, attn_mask=x_mask)
        x = x + self.self_attn_dropout_1(attn_1)
        x = self.self_attn_norm_1(x)
        
        q_with_pos = x if query_pos is None else x+query_pos
        k_with_pos = memory if pos is None else memory+pos
        
        attn_2, _ = self.self_attn_2(q_with_pos, k_with_pos, value=memory, attn_mask=memory_mask)
        x = x + self.self_attn_dropout_2(attn_2)
        x = self.self_attn_norm_2(x)
        
        mlp = self.l2( self.d1( self.activation( self.l1(x) ) ) )
        x = x + self.d2( mlp )
        x = self.norm(x)
        
        return x



        
        
class Transformer(nn.Module):
    
    def __init__(self, d_model = 512, nhead = 8, num_encoder_layers=6, num_decoder_layers=6,
                    dim_feedforward=2048, dropout=0.1, activation=F.relu):
        
        super().__init__()
        
        self.encoder = CustomEncoderSequential( *[
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation) for _ in range(num_encoder_layers) 
        ])
        self.decoder = CustomDecoderSequential( *[
                TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation) for _ in range(num_decoder_layers) 
        ])
        self.decoder_norm = nn.LayerNorm( d_model )
    
    def forward( self, x, query_embed, pos_embed ):
        bs, c, h, w = x.shape
        
        x = x.flatten(2).permute( 2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        
        memory = self.encoder({'x': x, 'pos': pos_embed})
        
        out = self.decoder({'x': torch.zeros_like(query_embed), 'memory':memory, 'pos': pos_embed, 'query_pos': query_embed})
        out = self.decoder_norm(out)
        
        memory = memory.permute(1, 2, 0).view(bs, c, h, w)
        
        return out.transpose(0, 1), memory