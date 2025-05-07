# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import Linear
import torch_geometric.nn as gnn
from torch_geometric.utils import to_dense_batch, dense_to_sparse

from .layers import TransformerEncoderLayer, SpatialExtractor
from einops import repeat, rearrange
from .utils import sinusoidal_position_encoding

from .ptdec import DEC


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index,
            subgraph_edge_index=None,
            subgraph_edge_attr=None,  edge_attr=None,
            ptr=None, batch=None, return_attn=False, dm=None, coord=None, comm_edge_index=None, comm_edge_attr=None):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index,
                edge_attr=edge_attr,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=ptr,
                batch=batch,
                return_attn=return_attn,
                dm=dm,
                comm_edge_index=comm_edge_index,
                comm_edge_attr=comm_edge_attr,
                coord=coord
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, pe=False, pe_dim=0,
                 gnn_type="gcn", se="gnn", use_edge_attr=False, num_edge_features=2,
                 **kwargs):
        super().__init__()

        self.pe = pe
        self.pe_dim = pe_dim
        if pe and pe_dim > 0:
            self.embedding_pe = nn.Linear(pe_dim, d_model)
            # self.embedding_pe = nn.Linear(3, d_model)

        self.embedding = nn.Linear(in_features=in_size, out_features=d_model, bias=True)

        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 8)
            self.embedding_edge = nn.Linear(in_features=num_edge_features, out_features=edge_dim, bias=False)
            self.embedding_comm_edge = nn.Linear(in_features=1, out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm, **kwargs)
        self.ds_spatial_feature_extractor = SpatialExtractor(d_model, gnn_type=gnn_type,
                                                                   num_layers=1, **kwargs)
        self.gt_encoder = GraphTransformerEncoder(encoder_layer, num_layers)


        self.encoder = nn.Sequential(
            nn.Linear(d_model *
                      200, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32,
                      d_model * 200),
        )
        self.dec = DEC(cluster_number=100, hidden_dimension=d_model, encoder=self.encoder,
                       orthogonal=True, freeze_center=False, project_assignment=True)
        self.dim_reduction = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(100 * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_class)
        )
        self.node_rearranged_len = [0, 23, 59, 82, 101, 127, 144, 169, 200]

    def forward(self, data, return_attn=False):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        batch = data.batch
        pe = data.pe
        coord = data.coord
        coord = coord / torch.norm(coord, dim=1, keepdim=True)
        comm_graph = data.comm_graph
        # 扩展 coord 张量在第二维度，以便进行广播
        diff = coord[:200,:].unsqueeze(1) - coord[:200,:].unsqueeze(0)
        # 计算差值的平方，然后沿特定维度求和
        dist_matrix = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=-1))



        if self.se == "commgnn":
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_edge_attr = data.subgraph_edge_attr
            comm_edge_index, comm_edge_attr = dense_to_sparse(comm_graph)
            comm_edge_attr = comm_edge_attr.unsqueeze(1).float()
        else:
            subgraph_edge_index = None
            subgraph_edge_attr = None
            comm_edge_index = None
            comm_edge_attr = None

        output = self.embedding(x)


        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
            if comm_edge_attr is not None:
                comm_edge_attr = self.embedding_comm_edge(comm_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None
            comm_edge_attr = None

        output = self.ds_spatial_feature_extractor(
            x=output,
            edge_index=edge_index,
            edge_attr=edge_attr,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_edge_attr=subgraph_edge_attr,
            comm_edge_index=comm_edge_index,
            comm_edge_attr=comm_edge_attr,
            batch=batch
        )
        if self.pe and pe is not None:
            pe = torch.cat([pe, coord], dim=-1)
            pe = self.embedding_pe(pe)
            output = output + pe
        output = self.gt_encoder(
            output,
            edge_index,
            edge_attr=edge_attr,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            batch=batch,
            return_attn=return_attn,
            dm=dist_matrix,
            comm_edge_index=comm_edge_index,
            comm_edge_attr=comm_edge_attr,
            coord=coord
        )


        #OCRead
        x_dense, mask = to_dense_batch(output, batch)
        x, assignment = self.dec(x_dense)
        x = self.dim_reduction(x)
        x = x.reshape((x.shape[0], -1))

        return self.fc(x)
