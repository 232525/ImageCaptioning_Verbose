"""
M2 Transformer
Decoder
"""
from .transformer.attention import MultiHeadAttention, ScaledDotProductAttentionMemory
from .transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class M2Decoder(nn.Module):
    def __init__(
        self, vocab_size, max_len, N_dec, 
        dim=1024, 
        num_heads=8,
        mlp_ratio=4.0,
        dropout=.1,
        w_pf=False
    ):
        super(M2Decoder, self).__init__()
        self.dim = dim
        self.depth = N_dec
        self.vocab_size = vocab_size
        
        # 词汇嵌入矩阵
        self.word_embed = nn.Embedding(self.vocab_size, self.dim)
        # 词汇位置嵌入矩阵
        self.pos_embed  = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(max_len + 1, dim, 0), freeze=True
        )
        
        self.layers = nn.ModuleList([
            DecoderLayer(
                dim=dim, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                w_pf = w_pf
            ) for i in range(self.depth)
        ])
        
        # 输出层
        self.generator = nn.Linear(self.dim, self.vocab_size, bias=True)
        
        self.clear_buffer()
        
    def init_buffer(self, batch_size):
        # self.seq_len用于记录Inference时运行的时间步
        self.seq_len = 0
        for layer in self.layers:
            layer.init_buffer(batch_size)
            
    def clear_buffer(self):
        self.seq_len = None
        for layer in self.layers:
            layer.clear_buffer()
            
    def apply_to_states(self, fn):
        for layer in self.layers:
            layer.apply_to_states(fn)
    
    # TODO: fix
    def precompute(self, encoder_output):
        p_att_feats = []
        for layer in self.layers:
            pass
        return p_att_feats
    
    def forward(self, gx, seq, encoder_output, seq_mask=None):
        """
        seq: 图像描述序列，[B, Seq_len]，推理过程中Seq_len=1
        encoder_output: Encoder输出，[B, L, C]，即增强后图像grid特征
        """
        seq_len = seq.size()[1]
        pos_indx = torch.arange(1, seq_len + 1, device='cuda').view(1, -1)
        if self.seq_len is not None:
            seq_len = self.seq_len + seq_len
            self.seq_len = seq_len
            pos_indx = torch.arange(seq_len, seq_len + 1, device='cuda').view(1, -1)
            
        # 词汇嵌入 + 位置嵌入
        # [B, seq_len, C] for training or [B, 1, C] for inference
        x = self.word_embed(seq) + self.pos_embed(pos_indx)
        
        # TODO:
        # 可以考虑嵌入图像全局特征
        
        for layer in self.layers:
            x = layer(gx, x, encoder_output, seq_mask)
            
        x = self.generator(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.1, w_pf=False):
        super(DecoderLayer, self).__init__()
        self.dim = dim                            # 1536
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.self_attn = M2MultiHeadSelfAttention(dim, num_heads=num_heads)
        # 单词序列与encoder输出注意力
        self.enc_attn  = M2MultiHeadSelfAttention(dim, num_heads=num_heads)
        self.pwff = PositionWiseFeedForward(dim, int(dim*mlp_ratio), drop)

        self.fc_alpha1 = nn.Linear(dim + dim, dim)
        self.fc_alpha2 = nn.Linear(dim + dim, dim)
        self.fc_alpha3 = nn.Linear(dim + dim, dim)
        
        self.w_pf = w_pf
        # 是否使用Pre-Fusion Module
        if self.w_pf:
            self.fuse_layer = nn.Sequential(
                nn.Linear(dim*2, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.fuse_layer_norm = nn.LayerNorm(dim)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)
    
    def init_buffer(self, batch_size):
        self.self_attn.init_buffer(batch_size)
    
    def clear_buffer(self):
        self.self_attn.clear_buffer()
    
    def apply_to_states(self, fn):
        self.self_attn.apply_to_states(fn)
    
    def precompute(self, encoder_output):
        # TODO: fix
        pass
        return None
    
    def forward(self, gx, seq_embed, encoder_output, seq_mask):
        """
        seq_embed: [B, seq_len, C]
        encoder_output: [B, L, C]
        seq_mask: [B, seq_len, seq_len]
        """
        # 预融合操作
        if self.w_pf:
            assert gx is not None, 'gx is None'
            seq_embed_cat = torch.cat([seq_embed, gx.unsqueeze(1).expand_as(seq_embed)], dim=-1)
            seq_embed = self.fuse_layer(seq_embed_cat) + seq_embed
            seq_embed = self.fuse_layer_norm(seq_embed)
        
        # 残差在注意力机制内容实习
        self_att = self.self_attn(seq_embed, seq_embed, seq_embed, seq_mask)

        enc_att1 = self.enc_attn(self_att, encoder_output[:, 0], encoder_output[:, 0])
        enc_att2 = self.enc_attn(self_att, encoder_output[:, 1], encoder_output[:, 1])
        enc_att3 = self.enc_attn(self_att, encoder_output[:, 2], encoder_output[:, 2])

        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))

        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2 + enc_att3 * alpha3) / np.sqrt(3)

        ff = self.pwff(enc_att)
        
        return ff

    
class M2MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, dropout=0.1):
        super(M2MultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
        self.init_weights()
        self.clear_buffer()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.q_linear.bias, 0)
        nn.init.constant_(self.k_linear.bias, 0)
        nn.init.constant_(self.v_linear.bias, 0)
        nn.init.constant_(self.proj.bias, 0)
        
    def init_buffer(self, batch_size):
        # [B, nH, 0, C/nH]
        self.buffer_key = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device='cuda')
        self.buffer_value = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device='cuda')
        
    def clear_buffer(self):
        self.buffer_key = None
        self.buffer_value = None
        
    def apply_to_states(self, fn):
        self.buffer_key = fn(self.buffer_key)
        self.buffer_value = fn(self.buffer_value)
    
    def forward(self, q, k, v, mask=None):
        """
        q: [B, seq_len, C]
        k: [B, seq_len, C] or [B, L, C]
        v: [B, seq_len, C] or [B, L, C]
        mask: None or [1, seq_len, seq_len]
        """
        shortcut = q
        
        B_, N, C = q.size()
        # 线性变换
        q = self.q_linear(q).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 存储buffer，用于inference
        if self.buffer_key is not None and self.buffer_value is not None:
            self.buffer_key = torch.cat([self.buffer_key, k], dim=2)
            self.buffer_value = torch.cat([self.buffer_value, v], dim=2)
            k = self.buffer_key
            v = self.buffer_value
            
        # 注意力核心操作
        # attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # （词汇嵌入本身加入了绝对位置编码）
        # 相对位置编码，TODO：
        # 单词嵌入序列自注意力时可加入
        # 单词嵌入与图像特征注意力机制时？
        
        # 计算注意力权重
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        
        # 残差 + LayerNorm
        out = self.dropout(out)
        out = self.layer_norm(shortcut + out)
        
        return out
    