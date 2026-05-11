import os
import random
import sys
import einops
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from einops.layers.torch import Rearrange, Reduce
from transformers import PretrainedConfig

from FAST_addon import *

class DebugPrint(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"{self.name}: {x.shape}")
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        attn_output, attn_weights = self.attn(inp_x, inp_x, inp_x)
        x = x + attn_output
        x = x + self.linear(self.layer_norm_2(x))
        return x

class V0(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, dim, (1, 5), bias=True)
        self.cnn2 = nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False)
        self.cnn3 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False)
        self.cnn4 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False)

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = F.gelu(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x
    
class V1(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, dim, (1, 9), stride=(1, 2), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False),
            nn.Conv2d(dim, dim, (1, 7), stride=(1, 2), padding=(0, 0), bias=False),
            nn.Conv2d(dim, dim, (1, 7), stride=(1, 2), padding=(0, 0), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.CNN(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

class V2(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, dim, (1, 5), bias=True)
        self.cnn2 = nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False)
        self.cnn3 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False)
        self.cnn4 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False)

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = F.gelu(x)
        x = self.cnn3(x)
        x = F.gelu(x)
        x = self.cnn4(x)
        x = F.gelu(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

class V2_1(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, dim, (1, 5), bias=True)
        self.cnn2 = nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False)
        self.cnn3 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False)
        self.bn3 = nn.BatchNorm2d(dim)
        self.cnn4 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False)
        self.bn4 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = F.gelu(x)
        x = self.cnn3(x)
        x = F.gelu(x)
        x = self.bn3(x)
        x = self.cnn4(x)
        x = F.gelu(x)
        x = self.bn4(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

class V2_2(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, dim, (1, 5), bias=True)
        self.cnn2 = nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False)
        self.cnn3 = nn.Conv2d(dim, dim, (1, 5), padding=0, bias=False)
        self.cnn4 = nn.Conv2d(dim, dim, (1, 5), padding=0, bias=False)

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = F.gelu(x)
        x = self.cnn3(x)
        x = F.gelu(x)
        x = self.cnn4(x)
        x = F.gelu(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

class V2_3(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# 2T + S
class V2_4(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# 3T + S
class V2_5(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# + AvgPool (1, 3)
class V2_6(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False),
            nn.AvgPool2d((1, 3), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# + AvgPool (1, 5)
class V2_7(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False),
            nn.AvgPool2d((1, 5), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x
    
# + AvgPool (1, 5) + Activation
class V2_8(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False),
            nn.GELU(),
            nn.AvgPool2d((1, 5), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# Activation + maxpool (1, 5) **
class V2_8a(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False),
            nn.GELU(),
            nn.MaxPool2d((1, 5), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# Activation + maxpool (1, 5) + bias with cnn2
class V2_8b(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=True),
            nn.GELU(),
            nn.MaxPool2d((1, 5), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# + 2 AvgPool (1, 5)
class V2_9(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False),
            nn.AvgPool2d((1, 5), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.AvgPool2d((1, 5), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x
    
# + 2 AvgPool (1, 5) + Activation
class V2_10(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=False),
            nn.GELU(),
            nn.AvgPool2d((1, 5), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
            nn.AvgPool2d((1, 5), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# Based on V2_8a
class V2_8a_1(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 7), padding=(0, 3), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=True),
            nn.GELU(),
            nn.MaxPool2d((1, 4), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# Based on V2_8a
class V2_8a_2(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 9), padding=(0, 4), bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=True),
            nn.MaxPool2d((1, 4), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# Based on V2_8a
class V2_8a_3(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 5), padding=(0, 2), bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=True),
            nn.MaxPool2d((1, 4), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, (1, 9), padding=(0, 4), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x
    
class V2_8a_4(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 5), padding=(0, 2), bias=True),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=True),
            nn.MaxPool2d((1, 4), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, (1, 9), padding=(0, 4), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# Based on V2_4
class V2_4_1(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 5), padding=(0, 2), bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), bias=False),
            nn.MaxPool2d((1, 4), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=True),
            nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

# Based on V0
class V0_1(nn.Module):
    def __init__(self, channels, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, (1, 3), padding=(0, 1), bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, (channels, 1), padding=0, bias=True),
            nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), bias=False),
            nn.MaxPool2d((1, 4), stride=(1, 2), padding=(0, 1)),
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x

class V3(nn.Module):
    def __init__(self, channels, dim=32, pool1=4, pool2=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim // 4, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),

            nn.Conv2d(dim // 4, dim // 4, (channels, 1), padding=0),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),

            nn.MaxPool2d((1, pool1), stride=(1, pool1)),

            nn.Conv2d(dim // 4, dim // 2, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),

            nn.MaxPool2d((1, pool2), stride=(1, pool2)),

            nn.Conv2d(dim // 2, dim, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, C, T)  -> (B, F)
        x = einops.rearrange(x, 'B C T -> B 1 C T')
        x = self.cnn(x)
        x = einops.reduce(x, 'B F 1 T -> B F', 'mean')
        return x
    

class Head(nn.Module):
    def __init__(self, head, electrodes, zone_dict, feature_dim):
        super().__init__()
        self.index_dict = {}
        Head = globals()[head]
        self.encoders = nn.ModuleDict()
        for area, ch_names in zone_dict.items():
            self.index_dict[area] = torch.tensor([electrodes.index(ch_name) for ch_name in ch_names])
            self.encoders[area] = Head(len(self.index_dict[area]), feature_dim)
        
    def forward(self, x):
        return torch.stack([encoder(x[:, self.index_dict[area]]) for area, encoder in self.encoders.items()], dim=1)

class FAST(nn.Module):
    name='FAST'
    def __init__(self, config):
        super().__init__()
        self.config = config
        electrodes = config.electrodes
        zone_dict = config.zone_dict
        head = config.head
        dim_cnn = config.dim_cnn
        dim_token = config.dim_token
        seq_len = config.seq_len
        window_len = config.window_len
        slide_step = config.slide_step
        n_classes = config.n_classes
        num_heads = config.num_heads
        num_layers = config.num_layers
        dropout = config.dropout

        self.n_tokens = (seq_len - window_len) // slide_step + 1
        # print('** n_tokens:', self.n_tokens, 'from', seq_len, window_len, slide_step)
        # print('** dim_cnn:', dim_cnn, 'dim_token:', dim_token)

        self.head = Head(head, electrodes, zone_dict, dim_cnn)
        self.input_layer = nn.Sequential(nn.Linear(dim_cnn * len(zone_dict), dim_token), nn.GELU())
        self.transformer = nn.Sequential(*[AttentionBlock(dim_token, dim_token*2, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_tokens + 1, dim_token))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_token))
        self.last_layer = nn.Linear(dim_token, n_classes) 
        self.dropout = nn.Dropout(dropout)

    def forward_head(self, x, step_override = None):
        if step_override is not None:
            slide_step = step_override
        else:
            slide_step = self.config.slide_step
        x = x.unfold(-1, self.config.window_len, slide_step)
        B, C, N, T = x.shape
        x = einops.rearrange(x, 'B C N T -> (B N) C T')
        feature = self.head(x)
        feature = einops.rearrange(feature, '(B N) Z F -> B N Z F', B=B)
        return feature
    
    def batched_forward_head(self, x, step, batch_size):
        feature = []
        for mb in torch.split(x, batch_size, dim=0):
            feature.append(self.forward_head(mb, step))
        return torch.cat(feature, dim=0)
    
    def forward_transformer(self, x, mode="default"):
        x = einops.rearrange(x, 'B N Z F -> B N (Z F)')
        x = self.input_layer(x)
        cls_token_expand = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token_expand, x), dim=1)
        x = x + self.pos_embedding[:, :self.n_tokens + 1]
        tokens = self.transformer(x)
        if mode == "embeddings":
            return tokens
        logits = self.last_layer(self.dropout(tokens[:, 0]))
        return logits
    
    def forward(self, x, forward_mode = 'default'):
        if forward_mode == 'default':
            return self.forward_transformer(self.forward_head(x))
        elif forward_mode == 'train_head':
            x = self.forward_head(x)
            x = einops.rearrange(x, 'B N Z F -> B N (Z F)')
            tokens = self.input_layer(x)
            logits = self.last_layer(tokens).mean(dim=1)
            return logits
        elif forward_mode == 'train_transformer':
            with torch.no_grad():
                x = self.forward_head(x)
            return self.forward_transformer(x)
        elif forward_mode == 'embeddings':
            return self.forward_transformer(self.forward_head(x), mode='embeddings')
        else:
            raise NotImplementedError
        
    def forward_get_tokens(self, x):
        x_shape = x.shape
        x = self.forward_head(x)
        x_head_shape = x.shape
        x = einops.rearrange(x, 'B N Z F -> B N (Z F)')
        x = self.input_layer(x)
        cls_token_expand = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token_expand, x), dim=1)
        try:
            x = x + self.pos_embedding[:, :self.n_tokens + 1]
        except Exception as e:
            raise ValueError(f"Error in forward_get_tokens: {e}, original shape: {x_shape}, after head shape: {x_head_shape}")
        tokens = self.transformer(x)
        return tokens
    
    def forward_get_token_cls(self, x):
        x = self.forward_head(x)
        x = einops.rearrange(x, 'B N Z F -> B N (Z F)')
        x = self.input_layer(x)
        cls_token_expand = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token_expand, x), dim=1)
        x = x + self.pos_embedding[:, :self.n_tokens + 1]
        tokens = self.transformer(x)
        return tokens[:, 0]
        
if __name__ == '__main__':
    from DSO_ElectrodeZones import Electrodes, Zones

    cls_models = [
        V2_1,
        V2_2,
        V2_3,
        V2_4,
        V2_5,
        V2_6,
        V2_7,
        V2_8,
        V2_9,
        V2_10
    ]

    x = torch.randn(1, 6, 100)
    for cls_model in cls_models:
        print(cls_model.__name__)
        model = cls_model(6)
        print(model(x).shape)

    # model = V2_6(6)
    # x = torch.randn(1, 6, 100)
    # print(model(x).shape)

    # config = FAST(
    #     electrodes=Electrodes,
    #     zone_dict=Zones,
    #     head='V0',
    #     dim_cnn=32,
    #     dim_token=32,
    #     seq_len=3000,
    #     window_len=250,
    #     slide_step=250,
    #     n_classes=5,
    #     num_layers=4,
    #     num_heads=8,
    #     dropout=0.2,
    # )
    # model = Head_Benchmark(config)
    # x = torch.randn(10, len(Electrodes), config.seq_len)

    # print(model(x)[0].shape)
    # o1, o2 = model(x)
    # print(o1.shape, o2.shape)
    # print(model(x, 60)[0].shape)
    # print(model(x, 120)[0].shape)