import torch
from torch_geometric.data import Dataset
import os
from .context_gen import Reaction_Cluster
import json
from .data_utils import smiles2data, reformat_smiles
from collections import defaultdict
import random
from data_provider.caption_dataset import PretrainCaptionDataset
from data_provider.synthesis_dataset import SynthesisDataset

def format_float_from_string(s):
    try:
        float_value = float(s)
        return f'{float_value:.2f}'
    except ValueError:
        return s

class MoleculeAbstract(Dataset):
    def __init__(self, 
            root,
            rxn_num=1000,
            rxn_batch_size=4,
            smi_max_len=128,
            prompt=None,
            disable_graph_cache=False,
            disable_graphs=False,
            context_style='weighted_rxn',
            use_caption_dataset=False,
            caption_batch_num=10000,
            synthesis_datasetpath=None,
            synthesis_batch_num=10000,
            reverse_ratio=0.5,
            enable_abstract=True,
            enable_property=True,
            smiles_type='default',
            mode='train'
        ):
        super(MoleculeAbstract, self).__init__(root)
        self.root = root
        self.rxn_num = rxn_num
        self.rxn_batch_size = rxn_batch_size
        self.smi_max_len = smi_max_len
        self.context_style = context_style
        self.tokenizer = None
        self.disable_graph_cache = disable_graph_cache
        self.disable_graphs = disable_graphs
        self.use_caption_dataset = use_caption_dataset
        self.smiles_type = smiles_type
        if use_caption_dataset:
            self.caption_dataset = PretrainCaptionDataset(
                os.path.join(root, '../caption_data'),
                smi_max_len=smi_max_len,
                use_graph=not self.disable_graphs,
                disable_graph_cache=disable_graph_cache,
                smiles_type=smiles_type,
            )
            self.caption_batch_num = caption_batch_num
        self.use_synthesis_dataset = bool(synthesis_datasetpath)
        if self.use_synthesis_dataset:
            self.synthesis_dataset = SynthesisDataset(
                synthesis_datasetpath,
                'train',
                smi_max_len,
                roundrobin_train=True,
                use_graph=not disable_graphs,
                disable_graph_cache=disable_graph_cache,
                smiles_type='default',
            )
            self.synthesis_batch_num = synthesis_batch_num
        if not self.disable_graphs:
            self.mol_graph_map = torch.load(os.path.join(self.root, 'mol_graph_map.pt'))
        reaction_filename = 'reactions/reactions_test.json' if (mode=='test') else 'reactions/reactions.json'
        if smiles_type=='r_smiles':
            reaction_filename = 'reactions/reactions_wRSMILES.json'
        self.cluster = Reaction_Cluster(self.root, reaction_filename=reaction_filename, reverse_ratio=reverse_ratio)
        self.reload_data_list()
        self.abstract_max_len = 10240
        self.property_max_len = 10240
        self.enable_abstract = enable_abstract
        self.enable_property = enable_property

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        data_len = len(self.data_list)
        if self.use_caption_dataset:
            data_len += len(self.caption_index_list)
        if self.use_synthesis_dataset:
            data_len += len(self.synthesis_index_list)
        return data_len
    
    def reload_data_list(self):
        k = self.rxn_batch_size
        if self.context_style == 'weighted_rxn':
            self.data_list = self.cluster(self.rxn_num, k=k)
        elif self.context_style == 'uniform_rxn':
            self.data_list = self.cluster.generate_batch_uniform_rxn(self.rxn_num, k=k)
        elif self.context_style == 'uniform_mol':
            self.data_list = self.cluster.generate_batch_uniform_mol(self.rxn_num, k=k)
        elif self.context_style == 'single_mol':
            self.data_list = self.cluster.generate_batch_single(self.rxn_num)
        elif self.context_style == 'hybrid':
            self.data_list = self.cluster(self.rxn_num//2, k=k)
            self.data_list += self.cluster.generate_batch_uniform_mol(self.rxn_num//2, k=k)
        else:
            raise NotImplementedError
        if self.use_caption_dataset:
            assert self.caption_batch_num*k <= len(self.caption_dataset)
            caption_index_list = random.sample(range(len(self.caption_dataset)), self.caption_batch_num*k)
            self.caption_index_list = [caption_index_list[i*k:(i+1)*k] for i in range(self.caption_batch_num)]
        else:
            self.caption_index_list = []
        if self.use_synthesis_dataset:
            if self.synthesis_dataset.roundrobin_train:
                self.synthesis_dataset.reload_data()
            assert self.synthesis_batch_num <= len(self.synthesis_dataset)
            self.synthesis_index_list = random.sample(range(len(self.synthesis_dataset)), self.synthesis_batch_num)
        else:
            self.synthesis_index_list = []

    def make_prompt(self, mol_batch, smi_max_len=128):
        mol_prompt_list, text_prompt_list = [], []
        last_role = None
        for mol_dict in mol_batch:
            smiles = mol_dict['canon_smiles']
            if self.smiles_type=='r_smiles':
                if 'r_smiles' in mol_dict:
                    smiles = mol_dict['r_smiles']
                # else:
                #     smiles = reformat_smiles(smiles, smiles_type='restricted')
            else:
                smiles = reformat_smiles(smiles, smiles_type=self.smiles_type)
            mol_prompt = f'[START_SMILES]{smiles[:smi_max_len]}[END_SMILES]. '
            if 'role' in mol_dict:
                role = {
                    'REACTANT': 'Reactant',
                    'CATALYST': 'Catalyst',
                    'SOLVENT': 'Solvent',
                    'PRODUCT': 'Product',
                }[mol_dict['role']]
                if last_role != role:
                    mol_prompt = f'{role}: {mol_prompt}'
                    last_role = role
            text_prompt = self.make_abstract(mol_dict)
            mol_prompt_list.append(mol_prompt)
            text_prompt_list.append(text_prompt)
        return mol_prompt_list, text_prompt_list

    def make_abstract(self, mol_dict):
        prompt = ''
        if self.enable_abstract and 'abstract' in mol_dict:
            abstract_string = mol_dict['abstract'][:self.abstract_max_len]
            prompt += f'[Abstract] {abstract_string} '

        if self.enable_property:
            property_string = ''
            property_dict = mol_dict['property'] if 'property' in mol_dict else {}
            for property_key in ['Experimental Properties', 'Computed Properties']:
                if not property_key in property_dict:
                    continue
                for key, value in property_dict[property_key].items():
                    if isinstance(value, float):
                        key_value_string = f'{key}: {value:.2f}; '
                    elif isinstance(value, str):
                        float_value = format_float_from_string(value)
                        key_value_string = f'{key}: {float_value}; '
                    else:
                        key_value_string = f'{key}: {value}; '
                    if len(property_string+key_value_string) > self.property_max_len:
                        break
                    property_string += key_value_string
            if property_string:
                property_string = property_string[:self.property_max_len]
                prompt += f'[Properties] {property_string}. '
        return prompt

    def get_caption_data(self, index):
        caption_index = self.caption_index_list[index]
        graph_list, mol_prompt_list, text_prompt_list = [], [], []
        for idx in caption_index:
            graph_item, text, smiles_prompt = self.caption_dataset[idx]
            graph_list.append(graph_item)
            mol_prompt_list.append(smiles_prompt)
            text_prompt_list.append(text)

        return graph_list, mol_prompt_list, text_prompt_list
    
    def get_synthesis_data(self, index):
        synthesis_index = self.synthesis_index_list[index]
        _, graph_list, output_text, input_text = self.synthesis_dataset[synthesis_index]
        return graph_list, [input_text], [output_text]

    def __getitem__(self, index):
        if index < len(self.data_list):
            mol_batch = self.data_list[index]
        elif index < len(self.data_list)+len(self.caption_index_list):
            assert self.use_caption_dataset
            return self.get_caption_data(index-len(self.data_list))
        else:
            assert self.use_synthesis_dataset
            return self.get_synthesis_data(index-(len(self.data_list)+len(self.caption_index_list)))

        graph_list = []
        for mol_dict in mol_batch:
            smiles = mol_dict['canon_smiles']
            if self.disable_graphs:
                graph_item = None
            else:
                if self.disable_graph_cache:
                    graph_item = smiles2data(smiles)
                else:
                    assert smiles in self.mol_graph_map
                    graph_item = self.mol_graph_map[smiles]
            graph_list.append(graph_item)
        mol_prompt_list, text_prompt_list = self.make_prompt(mol_batch, smi_max_len=self.smi_max_len)

        return graph_list, mol_prompt_list, text_prompt_list
