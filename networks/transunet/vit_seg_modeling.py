# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch.nn.functional as F
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

def window_partition(x, window_size):
    # x: (B, C, H, W)
    # 입력 텐서 x는 배치 크기(B), 채널(C), 높이(H), 너비(W)로 구성
    B, C, H, W = x.shape  # 입력 텐서의 배치, 채널, 높이, 너비를 추출

    # num_win_h: 높이 방향의 윈도우 개수
    # num_win_w: 너비 방향의 윈도우 개수
    num_win_h = H // window_size
    num_win_w = W // window_size

    # 텐서를 윈도우 크기(window_size) 기준으로 재구성
    x = x.view(
        B,  # 배치 크기
        C,  # 채널 크기
        num_win_h,  # 윈도우 개수(높이 방향)
        window_size,  # 윈도우 크기(높이 방향)
        num_win_w,  # 윈도우 개수(너비 방향)
        window_size  # 윈도우 크기(너비 방향)
    )
    # x.shape: (B, C, num_win_h, window_size, num_win_w, window_size)

    # 윈도우를 순서대로 정리하기 위해 차원 재배치
    x = x.permute(
        0,  # 배치 크기는 유지
        2,  # 높이 방향 윈도우 개수
        4,  # 너비 방향 윈도우 개수
        1,  # 채널
        3,  # 윈도우 높이
        5   # 윈도우 너비
    ).contiguous()
    # x.shape: (B, num_win_h, num_win_w, C, window_size, window_size)

    # 최종적으로 모든 윈도우를 배치 차원으로 정리
    x = x.view(
        B * num_win_h * num_win_w,  # 배치 * 윈도우 개수
        C,  # 채널
        window_size,  # 윈도우 높이
        window_size   # 윈도우 너비
    )
    # x.shape: (B*num_win_h*num_win_w, C, window_size, window_size)

    return x  # 윈도우 데이터 반환

def window_unpartition(x, window_size, H, W, B):
    # x: (B*num_win_h*num_win_w, C, window_size, window_size)
    # window_partition의 결과로 생성된 윈도우 데이터
    # H, W: 원본 이미지의 높이와 너비
    # B: 배치 크기
    C = x.size(1)  # 채널 크기 추출

    # 윈도우 개수 계산
    num_win_h = H // window_size  # 높이 방향 윈도우 개수
    num_win_w = W // window_size  # 너비 방향 윈도우 개수

    # 윈도우 데이터를 원래 구조로 재배치
    x = x.view(
        B,  # 배치 크기
        num_win_h,  # 높이 방향 윈도우 개수
        num_win_w,  # 너비 방향 윈도우 개수
        C,  # 채널
        window_size,  # 윈도우 높이
        window_size   # 윈도우 너비
    )
    # x.shape: (B, num_win_h, num_win_w, C, window_size, window_size)

    # 차원을 재배치하여 윈도우를 이미지의 원래 위치로 복원
    x = x.permute(
        0,  # 배치 크기 유지
        3,  # 채널
        1,  # 높이 방향 윈도우 개수
        4,  # 윈도우 높이
        2,  # 너비 방향 윈도우 개수
        5   # 윈도우 너비
    ).contiguous()
    # x.shape: (B, C, num_win_h, window_size, num_win_w, window_size)

    # 원본 이미지 크기로 합치기
    x = x.view(
        B,  # 배치 크기
        C,  # 채널
        H,  # 원본 높이
        W   # 원본 너비
    )
    # x.shape: (B, C, H, W)

    return x  # 복원된 이미지 반환

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class NonLocalBlock_multicross(nn.Module):
    def __init__(self, in_channels, inter_channels=None, num_heads=16, window_size=7, num_global_tokens=1):
        super(NonLocalBlock_multicross, self).__init__()
        self.in_channels = in_channels                                                                 # in_channels: 입력 채널 수 (예: 1024)
        self.inter_channels = inter_channels or in_channels // 2                                       # inter_channels: 중간 채널 수, 없으면 in_channels//2
        self.num_heads = num_heads                                                                     # num_heads: 멀티헤드 어텐션에서 헤드 수
        self.window_size = window_size                                                                 # window_size: 윈도우 한 변의 크기 (7x7이라면 7)
        self.num_global_tokens = num_global_tokens                                                     # num_global_tokens: 글로벌 토큰 개수 (전역 정보를 담는 토큰 수)

        assert self.inter_channels % self.num_heads == 0, "inter_channels should be divisible by num_heads"
        self.head_dim = self.inter_channels // self.num_heads                                          # head_dim: 한 헤드가 담당하는 채널 수

        # Query, Key, Value를 만드는 1x1 Conv
        self.query_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)

        self.global_tokens = nn.Parameter(torch.randn(self.num_global_tokens, self.inter_channels))    # global_tokens: (num_global_tokens, inter_channels) 크기의 학습 가능 파라미터
        
        # 최종 출력 W_z: inter_channels → in_channels 로 되돌리는 Conv+BN
        self.W_z = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1),
            nn.BatchNorm2d(self.in_channels)
        )
        # W_z의 초기 가중치 셋업 (초기값 0으로, Residual 영향 최소화)
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

    def forward(self, x_thisBranch, x_otherBranch):
        # x_thisBranch: 기준 branch(기준 slice)의 Feature Map
        # x_otherBranch: 기준 branch(기준 slice) or 다른 branch(인접 slice)의 Feature Map

        B, C, H, W = x_thisBranch.size()

        # 1. 윈도우 분할 (partition)
        # Feature Map을 window_size x window_size 블록으로 나눈다.
        # window_partition: (B, C, H, W) → (B*window_num, C, window_size, window_size)
        x_this_win = window_partition(x_thisBranch, self.window_size)
        x_other_win = window_partition(x_otherBranch, self.window_size)

        # 2. Query, Key, Value 계산
        # 각각의 window에 대해 Query는 x_other_win에서, Key와 Value는 x_this_win에서 만든다.
        # Query/Key/Value: (B*window_num, inter_channels, window_size, window_size)
        query = self.query_conv(x_other_win)
        key = self.key_conv(x_this_win)
        value = self.value_conv(x_this_win)

        B_win = query.shape[0]                          # B_win: window 별 배치 크기 (=B*window_num)
        N = self.window_size * self.window_size         # 각 window 안의 token 수 (=window_size^2)

        # Query/Key/Value를 Multi-Head Attention에 맞게 (B_win, num_heads, N, head_dim) 형태로 변형
        query = query.view(B_win, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        key = key.view(B_win, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        value = value.view(B_win, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # 3. global token 추가
        # global_tokens: (num_global_tokens, inter_channels) → 각 window에 동일한 token 추가
        # (B_win, num_global_tokens, inter_channels) 형태로 변형
        global_tokens = self.global_tokens.unsqueeze(0).expand(B_win, -1, -1)
        # (B_win, num_heads, num_global_tokens, head_dim) 형태로 변형
        global_tokens = global_tokens.view(B_win, self.num_heads, self.num_global_tokens, self.head_dim)

        # Query, Key, Value에 global token 붙이기
        # 이제 각 window token 앞에 global token들이 추가되어 (num_global_tokens + N)개의 token이 됨
        query = torch.cat([global_tokens, query], dim=2)
        key = torch.cat([global_tokens, key], dim=2)
        value = torch.cat([global_tokens, value], dim=2)

        # 4. 어텐션 계산
        # attention_scores: (B_win, num_heads, (num_global_tokens + N), (num_global_tokens + N))
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_map = attention_weights.clone().detach()

        # 어텐션 가중치 * Value
        out = torch.matmul(attention_weights, value)  # (B_win, num_heads, num_global_tokens + N, head_dim)

        # 5. global token 부분 제거
        # 출력은 원래 window token 부분만 필요하므로 앞 부분에 붙었던 global token N개 제외
        out = out[:, :, self.num_global_tokens:, :]  # (B_win, num_heads, N, head_dim)

        # out을 다시 (B_win, inter_channels, window_size, window_size) 형태로 되돌림
        out = out.permute(0, 1, 3, 2).contiguous().view(B_win, self.inter_channels, self.window_size, self.window_size)

        # 6. window 복원 (unpartition)
        # 다시 (B, C, H, W) 형태로 복원
        x_un = window_unpartition(out, self.window_size, H, W, B)

        # 7. W_z를 거쳐 Residual 추가
        z = self.W_z(x_un) + x_thisBranch

        return z, attention_map

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, num_heads=16, window_size=7, num_global_tokens=1):
        super(NonLocalBlock, self).__init__()
        self.attention_block = NonLocalBlock_multicross(
            in_channels=in_channels,
            inter_channels=inter_channels,
            num_heads=num_heads,
            window_size=window_size,
            num_global_tokens=num_global_tokens
        )

    def forward(self, x_thisBranch, x_otherBranch):
        output, attention_map = self.attention_block(x_thisBranch, x_otherBranch)
        return output, attention_map

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownCross(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DownCross, self).__init__()
        self.downcross_conv = DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x):
        return self.downcross_conv(x)

class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        self.window_size = 7
        self.num_global_tokens = 1

        self.cross_attention_prev = NonLocalBlock(in_channels=1024, inter_channels=512, num_heads=16,
                                                     window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_self = NonLocalBlock(in_channels=1024, inter_channels=512, num_heads=16,
                                                     window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_next = NonLocalBlock(in_channels=1024, inter_channels=512, num_heads=16,
                                                     window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.downcross_three = DownCross(3072, 768, 1024)
        
        # Slice Fusion을 위한 학습 가능한 가중치 (3개 slice: prev, self, next)
        self.slice_fusion_weights = nn.Parameter(torch.tensor([0.25, 1.0, 0.25]))
        
        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
            
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, inputs):
        x_prev, x, x_next = inputs
        
        if self.hybrid:
            x_prev, features_prev = self.hybrid_model(x_prev)
            x, features = self.hybrid_model(x)
            x_next, features_next = self.hybrid_model(x_next)
        else:
            x_prev = self.patch_embeddings(x_prev)
            x = self.patch_embeddings(x)
            x_next = self.patch_embeddings(x_next)
            features = None

        xt1, attn_maps_prev = self.cross_attention_prev(x, x_prev)
        xt2, attn_maps_self = self.cross_attention_self(x, x)
        xt3, attn_maps_next = self.cross_attention_next(x, x_next)
        
        """ # softmax를 통해 세 슬라이스 가중치를 정규화
        weights_fused = F.softmax(self.slice_fusion_weights, dim=0)

        # 각 슬라이스에 가중치 적용 (prev: xt1, self: xt2, next: xt3)
        xt1 *= weights_fused[0]
        xt2 *= weights_fused[1]
        xt3 *= weights_fused[2] """
        
        xt = torch.cat([xt1, xt2, xt3], dim=1)
        x = self.downcross_three(xt)

        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        attention_maps = {
            'prev': attn_maps_prev,
            'self': attn_maps_self,
            'next': attn_maps_next
        }
        
        return embeddings, features, attention_maps

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, inputs):
        embedding_output, features, attention_maps = self.embeddings(inputs)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, features, attention_maps, attn_weights

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super(SegmentationHead, self).__init__(conv2d, upsampling)

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, kernel_size=3, padding=1, use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, vis=False):
        super(VisionTransformer, self).__init__()
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, inputs, return_attn=False):
        x_prev, x, x_next = inputs
        
        if x.size()[1] == 1:
            x_prev = x_prev.repeat(1, 3, 1, 1)
            x = x.repeat(1, 3, 1, 1)
            x_next = x_next.repeat(1, 3, 1, 1)
            
        inputs = (x_prev, x, x_next)
        x, features, attention_maps, attn_weights = self.transformer(inputs)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        
        if return_attn:
            return logits, attention_maps, attn_weights
        return logits

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[:, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


# Configuration dictionary for different Vision Transformer variants
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}