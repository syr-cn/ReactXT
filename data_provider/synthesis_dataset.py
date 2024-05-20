import torch
from torch_geometric.data import Dataset
import os
import random
import json
from .data_utils import smiles2data, escape_custom_split_sequence, reformat_smiles, generate_rsmiles

class SynthesisDataset(Dataset):
    def __init__(self,
            root,
            mode,
            smi_max_len=128,
            use_graph=True,
            disable_graph_cache=False,
            smiles_type='default',
            roundrobin_train=False,
            test_subset=-1
        ):
        super(SynthesisDataset, self).__init__(root)
        self.root = root
        if 'PtoR' in root:
            self.task = 'retro'
        elif 'pretrain' in root:
            self.task = 'pretrain'
        elif 'RtoP' in root:
            self.task = 'forward'
        else:
            raise NotImplementedError(f'Invalid task: {root}')
        if mode=='valid':
            mode='val'
        self.mode = mode
        self.smi_max_len = smi_max_len
        self.tokenizer = None
        self.use_graph = use_graph
        self.disable_graph_cache = disable_graph_cache
        self.smiles_type = smiles_type
        self.roundrobin_train = roundrobin_train
        with open(os.path.join(root, 'mol_graphid_map.json')) as f:
            self.mol_idx_map = json.load(f)
        if self.use_graph:
            self.idx_graph_map = torch.load(os.path.join(root, 'idx_graph_map.pt'))

        if self.roundrobin_train and mode=='train':
            self.reload_counter=-2
            self.reload_data()
        else:
            with open(os.path.join(root, mode, f'src-{mode}.txt')) as f:
                self.input_list = f.readlines()
            with open(os.path.join(root, mode, f'tgt-{mode}.txt')) as f:
                self.output_list = f.readlines()
            assert len(self.input_list) == len(self.output_list)
            self.renew_r_smiles()
            self.input_list = [smi.strip().replace(' ','') for smi in self.input_list]
            self.output_list = [smi.strip().replace(' ','') for smi in self.output_list]
        if test_subset>0 and mode=='test':
            assert test_subset<=len(self.input_list)
            self.input_list = self.input_list[:test_subset]
            self.input_list = self.input_list[:test_subset]

    def reload_data(self):
        if not self.roundrobin_train:
            return
        self.reload_counter = (self.reload_counter+1)%10
        if hasattr(self, 'input_list'):
            del self.input_list
        if hasattr(self, 'output_list'):
            del self.output_list
        with open(os.path.join(self.root, f'train/src-train_{self.reload_counter}.txt')) as f:
            self.input_list = f.readlines()
        with open(os.path.join(self.root, f'train/tgt-train_{self.reload_counter}.txt')) as f:
            self.output_list = f.readlines()
        assert len(self.input_list) == len(self.output_list)
        self.renew_r_smiles()
        self.input_list = [smi.strip().replace(' ','') for smi in self.input_list]
        self.output_list = [smi.strip().replace(' ','') for smi in self.output_list]
        input_list, output_list = [], []
        for input_smiles, output_smiles in zip(self.input_list, self.output_list):
            if input_smiles.count('.') != output_smiles.count('.'):
                continue
            input_list.append(input_smiles)
            output_list.append(output_smiles)
        print(f'Reloaded data from {self.root}/train/src-train_{self.reload_counter}.txt, filtered len={len(self.input_list)}', flush=True)
        self.input_list = input_list
        self.output_list = output_list
    
    def renew_r_smiles(self):
        if self.smiles_type == 'r_smiles' and self.mode == 'train':
            # only renew r_smiles for training set
            if not hasattr(self, 'input_list_mapped'):
                # here we back up the original input_list and output_list
                self.input_list_mapped = self.input_list
                self.output_list_mapped = self.output_list
            self.output_list, self.input_list = generate_rsmiles(self.output_list_mapped, self.input_list_mapped)
            self.input_list = [smi.strip().replace(' ','') for smi in self.input_list]
            self.output_list = [smi.strip().replace(' ','') for smi in self.output_list]

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.input_list)

    def make_prompt(self, input_smiles, output_smiles, smi_max_len=512):
        FORWARD_PROMPT = 'Question: Given the following reactant molecules: {}, what are the expected products? Answer: The product molecules are '
        FORWARD_CATALYST_PROMPT = '{}, and the following catalyst molecules: {}'
        RETRO_PROMPT = 'Question: Given the following product molecules: {}, what are the reactants that produce them? Answer: The reactant molecules are '
        # RETRO_PROMPT = 'Predict the reaction that produces the following product: {} '
        PRETRAIN_PROMPT = 'Reconstruct the masked molecule: {}. Answer: '
        smiles_wrapper = lambda x: reformat_smiles(x, smiles_type=self.smiles_type)[:smi_max_len]
        if self.task=='retro':
            assert '<separated>' not in input_smiles
            smiles_list = input_smiles.split('.')
            in_prompt = '; '.join([f'[START_SMILES]{smiles_wrapper(smi)}[END_SMILES]' for smi in smiles_list])
            input_prompt = RETRO_PROMPT.format(in_prompt)
        elif self.task=='forward':
            if '<separated>' in input_smiles:
                reactant_smiles, reagent_smiles = input_smiles.split('<separated>')
                reactant_smiles = reactant_smiles.split('.')
                reagent_smiles = reagent_smiles.split('.')
                reactant_prompt = '; '.join([f'[START_SMILES]{smiles_wrapper(smi)}[END_SMILES]' for smi in reactant_smiles])
                reagent_prompt = '; '.join([f'[START_SMILES]{smiles_wrapper(smi)}[END_SMILES]' for smi in reagent_smiles])
                smiles_list = reactant_smiles+reagent_smiles
                input_prompt = FORWARD_CATALYST_PROMPT.format(reactant_prompt, reagent_prompt)
            else:
                smiles_list = input_smiles.split('.')
                reactant_prompt = '; '.join([f'[START_SMILES]{smiles_wrapper(smi)}[END_SMILES]' for smi in smiles_list])
                input_prompt = reactant_prompt
            input_prompt = FORWARD_PROMPT.format(input_prompt)
        elif self.task=='pretrain':
            in_prompt = '; '.join([f'[START_SMILES]{smiles_wrapper(smi)}[END_SMILES]' for smi in input_smiles.split('.')])
            input_prompt = PRETRAIN_PROMPT.format(in_prompt)
            smiles_list = output_smiles.split('.')
        # output_smiles = ' '.join([f'[START_SMILES]{smi[:smi_max_len]}[END_SMILES]' for smi in output_smiles.split('.')])
        output_smiles = f'[START_SMILES]{output_smiles}[END_SMILES]'
        output_smiles = escape_custom_split_sequence(output_smiles)

        return input_prompt, smiles_list, output_smiles

    def __getitem__(self, index):
        input_smiles = self.input_list[index]
        output_smiles = self.output_list[index]
        input_text, smiles_list, output_text = self.make_prompt(input_smiles, output_smiles, smi_max_len=self.smi_max_len)
        output_text = output_text.strip()+'\n'

        graph_list = []
        if self.use_graph:
            for smiles in smiles_list:
                if self.disable_graph_cache:
                    graph_item = smiles2data(smiles)
                else:
                    assert smiles in self.mol_idx_map
                    idx = self.mol_idx_map[smiles]
                    assert idx in self.idx_graph_map
                    graph_item = self.idx_graph_map[idx]
                graph_list.append(graph_item)

        return index, graph_list, output_text, input_text