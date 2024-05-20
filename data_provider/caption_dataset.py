import torch
from torch_geometric.data import Dataset
import os
from torch_geometric.data import InMemoryDataset
from .data_utils import reformat_smiles
import random
import json

class PubChemDataset(InMemoryDataset):
    def __init__(self, path):
        super(PubChemDataset, self).__init__()
        self.data, self.slices = torch.load(path)
    
    def __getitem__(self, idx):
        return self.get(idx)

class CaptionDataset(Dataset):
    def __init__(self, root, mode, smi_max_len=128, use_graph=True, disable_graph_cache=False, smiles_type='default'):
        super(CaptionDataset, self).__init__(root)
        self.root = root
        self.file_path = os.path.join(root, f'{mode}.pt')
        self.smi_max_len = smi_max_len
        self.tokenizer = None
        self.use_graph = use_graph
        self.smiles_type = smiles_type

        self.data = PubChemDataset(self.file_path)

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        smiles = reformat_smiles(data.smiles, smiles_type=self.smiles_type)
        smiles_prompt = f'[START_I_SMILES]{smiles[:self.smi_max_len]}[END_I_SMILES]. '

        text_list = []
        count = 0
        for line in data.text.split('\n'):
            count += 1
            text_list.append(line.strip())
            if count > 100:
                break
        text = ' '.join(text_list) + '\n'
        graph_list = [data] if self.use_graph else []

        return index, graph_list, text, smiles_prompt

class PretrainCaptionDataset(Dataset):
    def __init__(self, root, smi_max_len=128, use_graph=True, disable_graph_cache=False):
        super(PretrainCaptionDataset, self).__init__(root)
        self.pre_train_data = CaptionDataset(
            root,
            'pretrain',
            smi_max_len=smi_max_len,
            use_graph=use_graph,
        )
        self.train_data = CaptionDataset(
            root,
            'train',
            smi_max_len=smi_max_len,
            use_graph=use_graph,
        )

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.pre_train_data) + len(self.train_data)

    def __getitem__(self, index):
        if index < len(self.pre_train_data):
            index, graph_list, text, smiles_prompt =  self.pre_train_data[index]
        else:
            index, graph_list, text, smiles_prompt = self.train_data[index - len(self.pre_train_data)]
        graph_item = graph_list[0]
        if hasattr(graph_item, 'iupac'):
            del graph_item.iupac
        if hasattr(graph_item, 'cid'):
            del graph_item.cid
        del graph_item.text
        del graph_item.smiles
        
        return graph_item, text, smiles_prompt