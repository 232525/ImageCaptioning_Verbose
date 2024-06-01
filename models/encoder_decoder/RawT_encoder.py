import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

# 加入全局特征共同处理
class Encoder(nn.Module):
    def __init__(
        self, 
        dim=512, 
        input_resolution=(12, 12), 
        depth=3, 
        num_heads=8, 
        window_size=12,  # =12 退化为普通MSA结构
        shift_size=6,    # =0  无SW-MSA，仅W-MSA
        mlp_ratio=4,
        drop=0.1,
        use_gx=False
    ):
        super(Encoder, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_gx = use_gx
        
        # 构建 W-MSA / SW-MSA 层
        # 输入特征尺寸为 144 = 12 x 12，如果构建 SW-MSA 层，
        # 则需要将 window_size 设置得更小，比如设置为 6，且shift_size > 0
        # SW-MSA仅在偶数层被构造，W-MSA在奇数层构造
        # 如：W-MSA，SW-MSA，W-MSA，SW-MSA ......
        self.layers = nn.ModuleList([
            EncoderLayer(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                use_gx=use_gx
            ) for i in range(self.depth)
        ])
    
    def forward(self, x, att_mask=None):
        # x: [B, H*W, C]
        # 对于grid特征，att mask为None亦可
        # 全局特征初始化，图像特征均值 [B, C]
        if att_mask is not None:
            gx = (torch.sum(x * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1))
        else:
            gx = x.mean(1)
            
        # 如果使用全局特征，则需要将全局特征gx和grid特征x合并送入后续层处理
        if self.use_gx:
            # [B, L+1, C]
            O = torch.cat([x, gx.unsqueeze(1)], dim=1)  # [B, H*W+1, C]
            # [B, 1, L+1]
            att_mask = torch.cat(
                [att_mask, torch.ones(att_mask.size()[0], 1).cuda()], 
                dim=1
            ).long().unsqueeze(1)
        else:
            O = x
            att_mask = att_mask.unsqueeze(1)
            
        # 核心操作层
        for layer in self.layers:
            O = layer(O, att_mask)
        
        if self.use_gx:
            gx = O[:, -1, :]
            x  = O[:, :-1, :]
        else:
            gx = O.mean(1)
            x = O
        return gx, x
    
class EncoderLayer(nn.Module):
    def __init__(
        self, 
        dim=512, 
        input_resolution=(12, 12), 
        num_heads=8, 
        window_size=12,    # 窗口大小，如果窗口大小和输入一致，则退化为普通MSA
        shift_size=0,      # shift大小，0 OR window_size // 2
        mlp_ratio=4,       # FeedForward 中间层维度变换
        drop=0.1,
        use_gx=False
    ):
        super(EncoderLayer, self).__init__()
        self.dim = dim                            # 1536
        self.num_heads = num_heads                # 8
        self.mlp_ratio = mlp_ratio     # 4
        self.use_gx = use_gx           # False

        # 构造注意力核心操作层
        self.encoder_attn = MSA(
            dim=dim,
            num_heads=num_heads
        )
        
        # dropout同时用于encoder_attn和ff_layer输出
        self.dropout = nn.Dropout(drop) 
        self.layer_norm1 = nn.LayerNorm(dim)
        
        # 构造FeedForward层
        ffn_embed_dim = int(dim * mlp_ratio)
        self.ff_layer = FeedForward(
            embed_dim = dim, 
            ffn_embed_dim = ffn_embed_dim, 
            relu_dropout = drop
        )
        self.layer_norm2 = nn.LayerNorm(dim)
    
    def forward(self, x, att_mask=None):
        # x: query / key / value  [B, L, C] 其中，L = H * W
        # x为grid特征，一个batch内每个样本特征数量一致，注意力计算时无需mask标注
        # att_mask 为 None 即可，不参与计算
        # H, W = self.input_resolution
        B, L, C = x.shape
        short_cut = x
        x = self.encoder_attn(x, att_mask)
        
        # 注意力后的残差
        x = self.dropout(x)
        x = self.layer_norm1(x + short_cut)
        
        # FeedForward及残差
        short_cut = x
        x = self.ff_layer(x)
        # dropout 残差 LayerNorm在此加入
        x = self.dropout(x)
        x = self.layer_norm2(x + short_cut)

        return x
    
# 不包含残差连接和LayerNorm
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = nn.ReLU()    # ReLU / GELU / CELU
        # self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class MSA(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.o_linear = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask):
        B_, N, C = x.size()
        
        q = self.q_linear(x).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # [B, H, L+1, L+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # [B, 1, L+1]
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.o_linear(out)
        return out
    