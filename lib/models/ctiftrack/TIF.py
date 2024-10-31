# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn, Tensor


from typing import List

import torch
import torch.nn.functional as F




class SelfAttention(torch.nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Linear transformations for queries, keys, and values for each head
        self.W_q = torch.nn.Conv1d(256, 256,kernel_size= 1 ,  bias=False)
        self.W_k = torch.nn.Conv1d(256, 256, kernel_size= 1 , bias=False)
        self.W_v = torch.nn.Conv1d(256, 256, kernel_size= 1 , bias=False)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()

        # Linear transformations for queries, keys, and values
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # Reshape to split into multiple heads
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate scaled dot-product attention
        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.matmul(attention, values).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, emb_dim)
        out = out + x
        return out




class FeatureStorage:
    def __init__(self, max_features=2):
        self.max_features = max_features
        self.features = []
        self.path = ''
        self.selfattention = SelfAttention(768, 8 ).to("cuda")
        self.current_video_id = None
    # @classmethod
    # def get_path(self, path):
    #     folder_name = path.split('/')[-2]
    #     print(self.paths)
    #     self.paths = folder_name


    # def diver_path(self):
    #     print(self.paths)
    #     return self.paths

    def store_feature(self, feature):
        
        self.features.append(feature.clone())
        
        if len(self.features) > self.max_features:
            del self.features[0]
    # @classmethod
    def get_feature(self):
        if len(self.features) :
            def compute_self_attention(sequence):
                attn_output= self.selfattention(sequence).to("cuda")
                return attn_output
            
            attention_weighted_sequences = [compute_self_attention(seq) for seq in self.features]
            # attention_weighted_sequences = [seq for seq in self.features]
            feature_matrix = torch.stack(attention_weighted_sequences, dim=0).mean(dim=0).detach()
            feature_matrix = feature_matrix.to(attention_weighted_sequences[0].device)
            last_feat = self.features[-1]
            
            return feature_matrix , last_feat



featureStorage = FeatureStorage(max_features=2)

class ChannelChuLi(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelChuLi, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x  = x.unsqueeze(3)
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        pool = avg_pool + max_pool

        attention = self.fc(pool)
        attention = self.sigmoid(attention)

        # 缩放操作
        x = x * attention * self.scale.expand_as(x)
        x  = x.squeeze(3)
        return x


class TIF(nn.Module):
    def __init__(self,   dim_in = 768, hidden_dim = 1024, dim_out = 768):
        super().__init__()
        self._build_layers( dim_in, hidden_dim , dim_out)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # self.featureStorage = FeatureStorage()
    def _build_layers(self, dim_in, hidden_dim, dim_out):
        self.save_proj = nn.Linear(dim_in, dim_in)
        self.temporal_attn = nn.MultiheadAttention(dim_in, 8, dropout=0)
        self.temporal_fc1 = nn.Linear(dim_in, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, dim_in)
        self.temporal_norm1 = nn.LayerNorm(dim_in)
        self.temporal_norm2 = nn.LayerNorm(dim_in)
        self.channelChuLi = ChannelChuLi(256)




    def _forward_temporal_attn(self, feature_matrix, current_feat):
        if len(feature_matrix) == 0:
            return feature_matrix
        
        embed = self.save_proj(current_feat)  # (n, 256)
        Channel_feat = self.channelChuLi(feature_matrix)
        if len(embed) > 0:
            prev_embed = self.save_proj(feature_matrix)
            embed2 = self.temporal_attn(
                embed,                  # (num_track, dim) to (1, num_track, dim)
                prev_embed.transpose(0, 1),   # (num_track, mem_len, dim) to (mem_len, num_track, dim)
                prev_embed.transpose(0, 1),
            )[0][0]
            embed = self.temporal_norm1(embed + embed2 )
            embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed * Channel_feat)))
            embed = self.temporal_norm2(embed + embed2)


        return embed


    def forward_temporal_attn(self, feature_matrix, current_feat):
        return self._forward_temporal_attn(feature_matrix, current_feat)

    def forward(self):
        feature_matrix ,current_feat= featureStorage.get_feature()
        
        fious_feat = self.forward_temporal_attn(feature_matrix , current_feat)

        return fious_feat