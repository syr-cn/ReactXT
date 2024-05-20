import torch
from torch_geometric.data import Dataset
import os
from torch_geometric.data import InMemoryDataset
import random
import json
from .data_utils import reformat_smiles

class ChEBI_dataset(Dataset):
    def __init__(self, root, mode, smi_max_len=128, use_graph=True, disable_graph_cache=False, smiles_type='default'):
        super(ChEBI_dataset, self).__init__(root)
        self.root = root
        self.file_path = os.path.join(root, f'{mode}.txt')
        self.smi_max_len = smi_max_len
        self.tokenizer = None
        self.use_graph = use_graph
        self.smiles_type = smiles_type
        if self.use_graph:
            self.idx_graph_map = torch.load(os.path.join(root, 'cid_graph_map.pt'))
        with open(self.file_path) as f:
            lines = f.readlines()
            self.data = [line.split('\t', maxsplit=2) for line in lines[1:]]
        

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cid, smiles, text = self.data[index]
        smiles = reformat_smiles(smiles, smiles_type=self.smiles_type)
        smiles_prompt = f'[START_I_SMILES]{smiles[:self.smi_max_len]}[END_I_SMILES]. '
        text = text.strip() + '\n'
        if self.use_graph:
            graph_list = [self.idx_graph_map[cid]]

        return index, graph_list, text, smiles_prompt
