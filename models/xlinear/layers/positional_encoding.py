# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn

# """
# XTransformer 位置嵌入矩阵类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()   
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # 位置嵌入计算也和Transformer不一致
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(max_len * 2.0) / d_model))
#         div_term = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x_size):
        return self.pe[:, :x_size]
# """

"""
# 位置嵌入矩阵保存为weight，与nn.Embedding保持一致
# 该类可以调用SwinTransformer Decoder保存的参数
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()   
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('weight', pe)

    def forward(self, x_size):
        return self.weight[:x_size].unsqueeze(0)
"""