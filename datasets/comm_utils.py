import csv
import math
import os
import pickle

import numpy as np
import pandas as pd
from pyreadr import pyreadr
import torch
import torch_geometric.utils as utils
from torch_scatter import scatter_add
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


def get_comm_index():
    with open('.../DICE_CPAC200_&_Yeo-7-liberal_res-1x1x1.pkl', 'rb') as handle:
        node_clus_map = pickle.load(handle)
    return list(node_clus_map.keys())


def get_dist_matrix():
    result = pyreadr.read_r('.../craddock200.rda')
    df = result["craddock200"]
    coordinate = df.loc[:, "x.mni":"z.mni"]
    x = df["x.mni"]
    y = df["y.mni"]
    z = df["z.mni"]
    dist_matrix = np.ones((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            X = np.array([x[i], y[i], z[i]])
            Y = np.array([x[j], y[j], z[j]])
            dist_matrix[i, j] = np.sqrt(sum(np.power((X - Y), 2)))
    return dist_matrix, coordinate

def get_community_graph(brain_graph,comm_index):
    # comm_index = comm_index[1:]
    num_communities = len(comm_index)-1
    community_graph = np.zeros((num_communities, num_communities))

    # 基于社区间的连接，构建社区间的连接图
    for i in range(num_communities):
        for j in range(i + 1, num_communities):
            # 计算两个社区之间的连接数作为边权重
            inter_community_edges = 0
            for node_i in range(comm_index[i], comm_index[i+1]):
                for node_j in range(comm_index[j], comm_index[j+1]):
                    if brain_graph.has_edge(node_i, node_j):
                        inter_community_edges += 1

            if inter_community_edges > 0:
                community_graph[i, j] = inter_community_edges
                community_graph[j, i] = inter_community_edges
    return community_graph


def rearrange_node(node_feature, rearranged_indices):

    node_feature_rearranged = node_feature[rearranged_indices, :]
    node_feature_rearranged = node_feature_rearranged[:, rearranged_indices]
    return node_feature_rearranged


def compute_pe(edge_index, num_nodes, pos_enc_dim):
    # pos_enc_dim = pos_enc_dim - 3
    W0 = normalize_adj(edge_index, num_nodes=num_nodes).tocsc()
    W = W0
    vector = torch.zeros((num_nodes, pos_enc_dim))
    vector2 = np.zeros((num_nodes, pos_enc_dim))
    vector[:, 0] = torch.from_numpy(W0.diagonal())
    vector2[:, 0] = W0.diagonal()
    for i in range(pos_enc_dim - 1):
        W = W.dot(W0)
        vector[:, i + 1] = torch.from_numpy(W.diagonal())
        vector2[:, i + 1] = W.diagonal()
    # vector = torch.cat([vector, torch.from_numpy(np.array(coordinate))], dim=-1)

    return vector.float()


# 将节点度倒数当作边权重
def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 device=edge_index.device)
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)

def normalize(matrix):
    min_val = matrix.min()
    max_val = matrix.max()
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix


def keep_top_n_symmetric(matrix, n):
    # 遍历每一行
    for i in range(matrix.shape[0]):
        # 找到当前行的前 n 个最大值的索引
        top_n_indices = np.argsort(matrix[i])[:-n - 1:-1]

        # 保留前 n 个最大值，其他值设为 0
        for j in range(matrix.shape[1]):
            if j not in top_n_indices and matrix[i][j] != 0:
                matrix[i][j] = 0
                matrix[j][i] = 0  # 保持对称性

    return matrix



def get_comm_edge_index(comm_index, edge_index, edge_attr, num_nodes, pos_enc_dim):
    edge_indices = []
    edge_attributes = []
    pe = []
    deg = []
    for start, end in zip(comm_index, comm_index[1:]):
        node_set = torch.IntTensor(list(range(start, end)))
        sub_edge_index, sub_edge_attr = utils.subgraph(node_set, edge_index, edge_attr,
                                                       num_nodes=num_nodes,
                                                       return_edge_mask=False)
        edge_indices.append(sub_edge_index)
        edge_attributes.append(sub_edge_attr)
        if sub_edge_index.numel() > 0:
            sub_edge_index = sub_edge_index-start  # edge_index从0开始
        sub_pe = compute_pe(sub_edge_index, len(node_set), pos_enc_dim)
        pe.append(sub_pe)
        sub_deg = utils.degree(sub_edge_index[0], len(node_set))
        deg.append(sub_deg)
    return torch.cat(edge_indices, dim=1), torch.cat(edge_attributes, dim=0), torch.cat(pe, dim=0), torch.cat(deg, dim=0)



# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score, pheno_fpath):
    scores_dict = {}
    with open(pheno_fpath) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                if score == 'HANDEDNESS_CATEGORY':
                    if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                        scores_dict[row['SUB_ID']] = 'R'
                    elif row[score] == 'Mixed':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    elif row[score] == 'L->R':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    else:
                        scores_dict[row['SUB_ID']] = row[score]
                elif score == 'FIQ' or score == 'PIQ' or score == 'VIQ':
                    if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                        scores_dict[row['SUB_ID']] = 100
                    else:
                        scores_dict[row['SUB_ID']] = float(row[score])

                else:
                    scores_dict[row['SUB_ID']] = row[score]

    return scores_dict
def get_pheno_info_from_scores(scores, subject_list, pheno_fpath):
    """
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_list : list of subject IDs
    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    pheno_ft = pd.DataFrame()
    global_phenos = []

    for i, l in enumerate(scores):
        phenos = []
        label_dict = get_subject_score(subject_list, l, pheno_fpath)

        # quantitative phenotypic scores
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                phenos.append(float(label_dict[subject_list[k]]))
        else:
            for k in range(num_nodes):
                phenos.append(label_dict[subject_list[k]])
        global_phenos.append(phenos)

    for i, l in enumerate(scores):
        pheno_ft.insert(i, l, global_phenos[i], True)

    return pheno_ft

# preprocess phenotypes. Categorical -> ordinal representation
def preprocess_phenotypes(pheno_ft):

    ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2, 3])], remainder='passthrough')

    pheno_ft = ct.fit_transform(pheno_ft)
    pheno_ft = pheno_ft.astype('float32')
    return pheno_ft

# create phenotype feature vector to concatenate with fmri feature vectors
def phenotype_ft_vector(pheno_ft, num_subjects):

    gender = pheno_ft[:, 0]
    site = pheno_ft[:, 1]
    eye = pheno_ft[:, 2]
    hand = pheno_ft[:, 3]
    age = pheno_ft[:, 4]
    fiq = pheno_ft[:, 5]
    viq = pheno_ft[:, 6]
    piq = pheno_ft[:, 7]

    phenotype_ft = np.zeros((num_subjects, 4))
    phenotype_ft_sex = np.zeros((num_subjects, 2))
    phenotype_ft_eye = np.zeros((num_subjects, 2))
    phenotype_ft_hand = np.zeros((num_subjects, 3))
    phenotype_ft_site = np.zeros((num_subjects, 20))

    for i in range(num_subjects):

        phenotype_ft_sex[i, int(gender[i])] = 1
        phenotype_ft[i, 0] = age[i]
        phenotype_ft[i, 1] = fiq[i]
        phenotype_ft[i, 2] = viq[i]
        phenotype_ft[i, 3] = piq[i]
        phenotype_ft_eye[i, int(eye[i])] = 1
        phenotype_ft_hand[i, int(hand[i])] = 1
        phenotype_ft_site[i, int(site[i])] = 1

    phenotype_ft = np.concatenate([phenotype_ft_sex, phenotype_ft_site, phenotype_ft_eye, phenotype_ft_hand, phenotype_ft], axis=1)

    return phenotype_ft

# Get the list of subject IDs
def get_ids(fpath, num_subjects=None):
    """

    return:
        subject_ids    : list of all subject IDs
    """

    subject_ids = np.genfromtxt(os.path.join(fpath, 'subject_ids.txt'), dtype=str)

    if num_subjects is not None:
        subject_ids = subject_ids[:num_subjects]

    return subject_ids

def get_pheno_info(data_folder,pheno_fpath):
    subject_ids = get_ids(data_folder)
    subject_ids.sort()
    pheno_ft = get_pheno_info_from_scores(['SEX', 'SITE_ID', 'EYE_STATUS_AT_SCAN', 'HANDEDNESS_CATEGORY',
                                                  'AGE_AT_SCAN', 'FIQ', 'VIQ', 'PIQ'], subject_ids, pheno_fpath)
    pheno_ft.index = subject_ids
    pheno_ft = preprocess_phenotypes(pheno_ft)
    phenotype_ft = phenotype_ft_vector(pheno_ft, len(subject_ids))
    return phenotype_ft

