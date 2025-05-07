import torch
from torch_geometric.data import InMemoryDataset, Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
from read_data import read_data, read_sigle_data, read_data_aal90, read_sparse_BrainNet,read_sigle_data_kedge,read_sigle_data_kedge2,read_comm_data,read_dynamic_graph,read_comm_data2,read_comm_datats,read_comm_data3
from tqdm import tqdm
from comm_utils import get_dist_matrix,get_comm_index,get_pheno_info


class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, raw_name, transform=None, pre_transform=None):
        self.root = root
        self.raw_name = raw_name
        super(ABIDEDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        # return 'raw_aal90'
        return self.raw_name
    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, self.raw_dir)
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        dm, co = get_dist_matrix()
        ri = get_comm_index()

        for i, f in enumerate(tqdm(self.raw_file_names, desc="Processing data")):

            # data_list.append(read_comm_data(osp.join(self.root, self.raw_dir), f, dm, co, ri, edge_num=20, pe_dim=30, pheno_info=pheno_info[i]))
            data_list.append(read_comm_data2(osp.join(self.root, self.raw_dir), f, dm, co, ri, edge_num=30, pe_dim=30, pheno_info=None))
            # data_list.append(read_comm_data3(osp.join(self.root, self.raw_dir), f, dm, co, ri, edge_num=30, pe_dim=30, pheno_info=None))
            # data_list.append(read_comm_datats(osp.join(self.root, self.raw_dir), f, dm, co, ri, edge_num=30, pe_dim=30, pheno_info=None))


        data_list = [elem for elem in data_list if elem is not None]

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


