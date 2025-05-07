import math
import os.path as osp
import pickle

import deepdish as dd
import networkx as nx
import numpy as np
from nilearn import connectome
import pyreadr
import torch
import torch_geometric.utils as utils
from networkx.convert_matrix import from_numpy_array
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops,dense_to_sparse
from torch_scatter import scatter_add
from torch_sparse import coalesce
from tqdm import tqdm
from comm_utils import rearrange_node,keep_top_n_symmetric,compute_pe,get_comm_edge_index,get_community_graph,normalize



def read_comm_data(data_dir, filename, dist_matrix, coordinate, rearranged_indices, edge_num=30, pe_dim=30):
    temp = dd.io.load(osp.join(data_dir, filename))
    att = temp['corr'][()]
    # time_series = temp['time_series'][()]
    att[np.isnan(att)] = 0  # nan->0
    num_nodes = att.shape[0]
    rearrange_att = rearrange_node(att, rearranged_indices)  # 节点重排列

    rearrange_dist_matrix = rearrange_node(dist_matrix, rearranged_indices)

    adj = np.abs(rearrange_att)  # 取绝对值
    row, col = np.diag_indices_from(adj)
    adj[row, col] = 0  # 对角赋值0

    sparse_adj = keep_top_n_symmetric(adj, edge_num)

    G = from_numpy_array(sparse_adj)
    A = nx.to_scipy_sparse_array(G)
    sparse_adj = A.tocoo()

    edge_att = np.zeros((len(sparse_adj.row), 2))

    distance_weights = np.exp(-rearrange_dist_matrix)
    for i in range(len(sparse_adj.row)):
        edge_att[i, 0] = adj[sparse_adj.row[i], sparse_adj.col[i]]
        edge_att[i, 1] = distance_weights[sparse_adj.row[i], sparse_adj.col[i]]
    edge_index = np.stack([sparse_adj.row, sparse_adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)  # 确保没有重复的边

    label = temp['label'][()]
    att_torch = torch.from_numpy(rearrange_att).float()
    att_torch[att_torch == float('inf')] = 0  # inf->0
    y_torch = torch.from_numpy(np.array(label)).long()  # classification
    edge_att = edge_att.float()
    pe = compute_pe(edge_index, num_nodes, pos_enc_dim=pe_dim)

    comm_index = [0, 23, 59, 82, 101, 127, 144, 169, 200]

    subgraph_edge_index, subgraph_edge_attr, subgraph_pe, subgraph_degree = get_comm_edge_index(comm_index,
                                                                                                edge_index,
                                                                                                edge_att,
                                                                                                num_nodes,
                                                                                                pos_enc_dim=pe_dim)
    community_graph = get_community_graph(G,comm_index)
    community_graph = normalize(community_graph)
    comm_graph = torch.from_numpy(community_graph)

    degree = utils.degree(edge_index[0], num_nodes)

    coord = torch.from_numpy(np.array(coordinate)[rearranged_indices, :]).float()  # 坐标重排


    rearrange_dist_matrix = torch.from_numpy(rearrange_dist_matrix).float()
    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att,
                subgraph_edge_index=subgraph_edge_index, subgraph_edge_attr=subgraph_edge_attr, pe=pe, deg=degree,
                subgraph_pe=subgraph_pe, subgraph_deg=subgraph_degree, coord=coord,
                dist_matrix=rearrange_dist_matrix, comm_graph=comm_graph)

    return data

