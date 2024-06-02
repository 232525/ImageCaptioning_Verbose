"""
M2 Transformer
Encoder
"""
from .transformer.attention import MultiHeadAttention, ScaledDotProductAttentionMemory
from .transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None,
                 attention_module_kwargs={'m': 40}):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                          identity_map_reordering=identity_map_reordering,
                          attention_module=attention_module,
                          attention_module_kwargs=attention_module_kwargs)
             for _ in range(N)]
        )

    def forward(self, input, attention_mask, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask
    
class M2Encoder(MultiLevelEncoder):
    def __init__(self, N, d_in=1536, **kwargs):
        """
        N, d_model=1024, d_k=128, d_v=128, h=8, d_ff=2048, dropout=.1,
        identity_map_reordering=False, attention_module=ScaledDotProductAttentionMemory,
        attention_module_kwargs={'m': 40}
        """
        super(M2Encoder, self).__init__(N, attention_module=ScaledDotProductAttentionMemory, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, att_mask=None, attention_weights=None):
        # 特征投影，先将特征原始维度，投影至模型维度
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)  # [B, L, C]
        # 对图像特征进行增强，具体操作见父类 MultiLevelEncoder 中的操作
        if att_mask is None:
            # [B, L]
            att_mask = torch.ones(out.size()[:2], device='cuda').long()
            
        outs, _ = super(M2Encoder, self).forward(out, att_mask==0, attention_weights=attention_weights)
        
        # att_feats = outs[:, -1, :, :].contiguous()
        # [B, 3, H*W, C]
        
        # 全局特征
        tmp_mask = att_mask.unsqueeze(1).unsqueeze(-1) # [B, 1, L, 1]
        # [B, 3, C]
        gx = torch.sum(outs * tmp_mask, -2) / torch.sum(tmp_mask, -2)
        # [B, C]
        gx = gx.mean(1)
        
        return gx, outs.contiguous()