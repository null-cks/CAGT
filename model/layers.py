# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
from .utils import pad_batch, unpad_batch
from .gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES
from entmax import entmax_bisect,entmax15
import torch.nn.functional as F


class Attention(gnn.MessagePassing):

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False,
                 symmetric=False, **kwargs):

        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

        self.batch_first = True

        self._qkv_same_embed_dim = True


        self.alpha = nn.Parameter(torch.ones(num_heads), requires_grad=True)
        self.theta1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.theta2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.epsilon = nn.Parameter(torch.ones(self.num_heads, 8), requires_grad=True)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
                x,
                edge_index,
                subgraph_edge_index=None,
                subgraph_edge_attr=None,
                edge_attr=None,
                ptr=None,
                batch=None,
                return_attn=False,
                dm=None,
                coord=None,
                comm_edge_index=None,
                comm_edge_attr=None
                ):

        # Compute value matrix
        v = self.to_v(x)

        # Compute query and key matrices
        if self.symmetric:
            qk = self.to_qk(x)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x).chunk(2, dim=-1)

        # Compute complete self-attention

        out, attn = self.self_attn(qk, v, ptr, return_attn=return_attn, dm=dm, coord=coord)
        return self.out_proj(out), attn


    def self_attn(self, qk, v, ptr, return_attn=False, dm=None, coord=None):
        """ Self attention which can return the attn """

        # qk, mask = pad_batch(qk, ptr, return_mask=True)
        k, q = map(lambda t: rearrange(t, '(b n) (h d) -> b h n d', n=200, h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale


        distance_weights = [
            self.distance_weight_matrix(dm, self.alpha[i])
            for i in range(self.num_heads)]
        comm_weights = [self.community_weight_matrix(self.epsilon[i]) for i in range(self.num_heads)]


        distance_weights = torch.stack(distance_weights, dim=0)
        comm_weights = torch.stack(comm_weights, dim=0)

        min_dis = distance_weights.min()
        max_dis = distance_weights.max()
        distance_weights = (distance_weights - min_dis) / (max_dis - min_dis)
        min_comm = comm_weights.min()
        max_comm = comm_weights.max()
        comm_weights = (comm_weights - min_comm) / (max_comm - min_comm)

        dots = dots + 0.2 * distance_weights * self.theta1 + 0.2 * comm_weights * self.theta2

        dots = self.attend(dots)

        dots = self.attn_dropout(dots)

        # v = pad_batch(v, ptr)
        v = rearrange(v, '(b n) (h d) -> b h n d', n=200, h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> (b n) (h d)')
        # out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None

    def distance_weight_matrix(self, distances, alpha):
        # 使用指数衰减和高斯调节来计算权重
        # weight_matrix = torch.exp(-alpha * distances) + beta * torch.exp(-gamma * (distances - delta) ** 2)
        # weight_matrix = torch.exp(-alpha * distances) + beta * torch.exp(-distances ** 2)
        # weight_matrix = torch.exp(-alpha * distances)

        weight_matrix = torch.exp(-distances ** 2 / (2 * alpha ** 2))
        weight_matrix = weight_matrix.clone()
        weight_matrix.fill_diagonal_(0)
        return weight_matrix

    def community_weight_matrix(self, epsilon):
        comm_index = [0, 23, 59, 82, 101, 127, 144, 169, 200]

        comm_weight = torch.zeros(comm_index[-1], comm_index[-1]).to(
            next(self.parameters()).device)  # 初始化权重矩阵，确保其大小和距离矩阵相同
        for start, end, eps in zip(comm_index[:-1], comm_index[1:], epsilon):
            comm_weight[start:end, start:end].fill_(eps)  # 使用.fill_() 来填充子矩阵
        comm_weight = comm_weight.clone()
        comm_weight.fill_diagonal_(0)
        return comm_weight

    def community_distance_weight_matrix(self, distances, alpha, epsilon):
        comm_index = [0, 23, 59, 82, 101, 127, 144, 169, 200]

        # 基本权重计算
        weight_matrix = torch.exp(-distances ** 2 / (2 * alpha ** 2))

        comm_weight = torch.zeros_like(weight_matrix)  # 初始化权重矩阵，确保其大小和距离矩阵相同
        for start, end, eps in zip(comm_index[:-1], comm_index[1:], epsilon):
            comm_weight[start:end, start:end].fill_(eps)  # 使用.fill_() 来填充子矩阵


        return weight_matrix + comm_weight




class SpatialExtractor(nn.Module):

    def __init__(self, embed_dim, gnn_type="gcn", batch_norm=True, commgnn=True, **kwargs):
        super().__init__()
        self.commgnn = commgnn
        self.gnn_type = gnn_type

        self.batch_norm = batch_norm
        self.comm_conv = get_simple_gnn_layer(gnn_type, embed_dim, **kwargs)
        self.local_conv = get_simple_gnn_layer(gnn_type, embed_dim, **kwargs)
        self.glob_conv = get_simple_gnn_layer(gnn_type, embed_dim, **kwargs)

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(embed_dim)

        self.comm_index = [0, 23, 59, 82, 101, 127, 144, 169, 200]
        self.fuse_lin = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x, edge_index, subgraph_edge_index, edge_attr=None,
                subgraph_edge_attr=None, comm_edge_index=None,
                comm_edge_attr=None, batch=None):

        if self.gnn_type in EDGE_GNN_TYPES:
            if edge_attr is None:

                x_local = F.relu(self.local_conv(x, subgraph_edge_index))
                x_lg = F.relu(self.glob_conv(x_local, edge_attr))
            else:
                x_local = F.relu(self.local_conv(x, subgraph_edge_index, subgraph_edge_attr))
                x_local = F.dropout(x_local, p=0.2, training=self.training)
                # x_glob = F.relu(self.glob_conv(x_local, edge_index, edge_attr))
                x_local_dense, mask = to_dense_batch(x_local, batch)
                com_mean = []
                com_max = []
                for start, end in zip(self.comm_index, self.comm_index[1:]):
                    com_mean.append(x_local_dense[:, start:end, :].mean(dim=1, keepdim=True))
                    com_max.append(x_local_dense[:, start:end, :].max(dim=1, keepdim=True)[0])
                com_fea_mean = torch.cat(com_mean, dim=1)
                com_fea_max = torch.cat(com_max, dim=1)
                com_fea_mean = com_fea_mean + com_fea_max
                com_fea_mean = rearrange(com_fea_mean, 'b n d -> (b n) d')
                com_fea_mean = F.relu(self.comm_conv(com_fea_mean, comm_edge_index, comm_edge_attr))
                com_fea_mean = F.dropout(com_fea_mean, p=0.2, training=self.training)
                com_fea_mean = rearrange(com_fea_mean, '(b n) d -> b n d', n=8)
                comm = []
                for start, end, i in zip(self.comm_index, self.comm_index[1:], range(len(self.comm_index[1:]))):
                    comm.append(com_fea_mean[:, i, :].unsqueeze(1).repeat(1, end - start, 1))
                x_comm = torch.cat(comm, dim=1)[mask]

                # z = torch.cat([x_local_dense, x_comm], dim=2)
                # x_glob = z[mask]
                # x_glob = torch.cat([x,z[mask]],dim=-1)
                # x_glob = self.fuse_lin(z[mask])

                x_lg = torch.cat([x_local, x_comm], dim=-1)
                x_lg = F.leaky_relu(self.fuse_lin(x_lg))
                x_lg = x + x_lg
        else:
            x_local = F.relu(self.local_conv(x, subgraph_edge_index))
            x_lg = F.relu(self.glob_conv(x_local, edge_index))

        x_struct = x_lg

        if self.batch_norm:
            x_struct = self.bn(x_struct)

        # x_struct = self.out_proj(x_struct)
        # x_struct = F.relu(self.out_proj2(x_struct)) + x
        # x_struct = x_struct
        # x_struct = self.fuse_lin(x_struct)

        return x_struct


class TransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", batch_norm=True, pre_norm=False,
                 **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = Attention(d_model, nhead, dropout=dropout, bias=False,  **kwargs)
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, edge_index,
                subgraph_edge_index=None,
                subgraph_edge_attr=None,
                edge_attr=None, ptr=None, batch=None,
                return_attn=False,
                dm=None,
                coord=None,
                comm_edge_index=None,
                comm_edge_attr=None
                ):

        if self.pre_norm:
            x = self.norm1(x)

        x2, attn = self.self_attn(
            x,
            edge_index,
            edge_attr=edge_attr,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=ptr,
            batch=batch,
            return_attn=return_attn,
            dm=dm,
            coord=coord,
            comm_edge_index=comm_edge_index,
            comm_edge_attr=comm_edge_attr
        )

        x = x + self.dropout1(x2)
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)

        if not self.pre_norm:
            x = self.norm2(x)
        return x
