import torch
from torch_geometric.data import Dataset
import os
import random
import json
from .data_utils import smiles2data, reformat_smiles

class ActionDataset(Dataset):
    def __init__(self, root, mode, smi_max_len, use_graph=True, disable_graph_cache=False, predict_rxn_condition=False, smiles_type='default'):
        super(ActionDataset, self).__init__(root)
        self.root = root
        self.smi_max_len = smi_max_len
        self.tokenizer = None
        self.use_graph = use_graph
        self.disable_graph_cache = disable_graph_cache
        self.predict_rxn_condition = predict_rxn_condition
        self.smiles_type = smiles_type

        with open(os.path.join(self.root, f'{mode}.json'), encoding='utf-8') as f:
            self.data_list = json.load(f)
        if self.use_graph:
            self.mol_graph_map = torch.load(os.path.join(self.root, 'mol_graph_map.pt'))
        # self.data_list = self.data_list[:100]

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.data_list)

    def make_prompt(self, param_dict, smi_max_len=128, predict_rxn_condition=False):
        action_sequence = param_dict['actions']
        smiles_list = []
        prompt = ''
        prompt += 'Reactants: '
        smiles_wrapper = lambda x: reformat_smiles(x, smiles_type=self.smiles_type)[:smi_max_len]
        for smi in param_dict['REACTANT']:
            prompt += f'{param_dict["extracted_molecules"][smi]}: [START_SMILES]{smiles_wrapper(smi)}[END_SMILES] '
            smiles_list.append(smi)
            
        prompt += 'Product: '
        for smi in param_dict['PRODUCT']:
            prompt += f'{param_dict["extracted_molecules"][smi]}: [START_SMILES]{smiles_wrapper(smi)}[END_SMILES] '
            smiles_list.append(smi)

        if param_dict['CATALYST']:
            prompt += 'Catalysts: '
            for smi in param_dict['CATALYST']:
                if smi in param_dict["extracted_molecules"]:
                    prompt += f'{param_dict["extracted_molecules"][smi]}: [START_SMILES]{smiles_wrapper(smi)}[END_SMILES] '
                else:
                    prompt += f'[START_SMILES]{smiles_wrapper(smi)}[END_SMILES] '
                smiles_list.append(smi)

        if param_dict['SOLVENT']:
            prompt += 'Solvents: '
            for smi in param_dict['SOLVENT']:
                if smi in param_dict["extracted_molecules"]:
                    prompt += f'{param_dict["extracted_molecules"][smi]}: [START_SMILES]{smiles_wrapper(smi)}[END_SMILES] '
                else:
                    prompt += f'[START_SMILES]{smiles_wrapper(smi)}[END_SMILES] '
                smiles_list.append(smi)

        if predict_rxn_condition:
            for value, token in param_dict['extracted_duration'].items():
                action_sequence = action_sequence.replace(token, value)
            for value, token in param_dict['extracted_temperature'].items():
                action_sequence = action_sequence.replace(token, value)
        else:
            prompt += 'Temperatures: '
            for value, token in param_dict['extracted_temperature'].items():
                prompt += f'{token}: {value} '

            prompt += 'Durations: '
            for value, token in param_dict['extracted_duration'].items():
                prompt += f'{token}: {value} '

        prompt += 'Action Squence: '
        return prompt, smiles_list, action_sequence

    def __getitem__(self, index):
        rxn_dict = self.data_list[index]
        rxn_id = rxn_dict['index']
        input_text, smiles_list, output_text = self.make_prompt(rxn_dict, self.smi_max_len, self.predict_rxn_condition)
        output_text = output_text.strip() + '\n'

        graph_list = []
        if self.use_graph:
            for smiles in smiles_list:
                if self.disable_graph_cache:
                    graph_item = smiles2data(smiles)
                else:
                    assert smiles in self.mol_graph_map
                    graph_item = self.mol_graph_map[smiles]
                graph_list.append(graph_item)
        return rxn_id, graph_list, output_text, input_text

