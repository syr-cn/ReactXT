# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from data_provider.reaction_action_dataset import ActionDataset
from data_provider.synthesis_dataset import SynthesisDataset
from data_provider.caption_dataset import CaptionDataset
from data_provider.chebi_dataset import ChEBI_dataset
import re

# we split individual characters inside special tokens like [START_DNA]
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"

def smiles_handler(text, mol_ph, is_gal=True):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)
    if is_gal:
        text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
        text = escape_custom_split_sequence(text)
        return text, smiles_list
    else:
        text = CUSTOM_SEQ_RE.sub(r'\3%s' % (mol_ph), text)
        return text, smiles_list

def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

class TrainCollater:
    def __init__(self, tokenizer, text_max_len, rxn_max_len, mol_ph, mol_token_id, is_gal=True, use_graph=True, use_qa_pair=True):
        self.rxn_max_len = rxn_max_len
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.is_gal = is_gal
        self.use_graph = use_graph
        self.use_qa_pair = use_qa_pair

    def __call__(self, batch):
        return self.collate_qa(batch) if self.use_qa_pair else self.collate(batch)

    def collate(self, batch):
        rxn_ids, graphs, texts, smiles_prompt = zip(*batch)
        if graphs:
            graphs = self.collater(graphs)
        
        ## deal with prompt
        if self.use_graph:
            smiles_prompt = [smiles_handler(p, self.mol_ph, self.is_gal)[0] for p in smiles_prompt]
        else:
            smiles_prompt = [escape_custom_split_sequence(p) for p in smiles_prompt]

        self.tokenizer.padding_side = 'left'
        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)

        is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
        smiles_prompt_tokens['is_mol_token'] = is_mol_token

        self.tokenizer.padding_side = 'right'
        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        return rxn_ids, graphs, smiles_prompt_tokens, text_tokens

    def collate_qa(self, batch):
        rxn_ids, graphs, texts, input_prompt = zip(*batch)
        graphs = [graph for graph_batch in graphs for graph in graph_batch]
        if graphs:
            graphs = self.collater(graphs)
        
        ## deal with prompt
        if self.use_graph:
            input_prompt = [smiles_handler(p, self.mol_ph, self.is_gal)[0] for p in input_prompt]
        else:
            input_prompt = [escape_custom_split_sequence(p) for p in input_prompt]

        self.tokenizer.padding_side = 'right'
        qa_pair = [[q, a] for q, a in zip(input_prompt, texts)]
        qa_batch = self.tokenizer(qa_pair,
                                    truncation=True,
                                    padding='longest',
                                    add_special_tokens=True,
                                    max_length=self.rxn_max_len + self.text_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True,
                                    return_token_type_ids=True)
        is_mol_token = qa_batch.input_ids == self.mol_token_id
        qa_batch['is_mol_token'] = is_mol_token
        return rxn_ids, graphs, qa_batch
    
class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, rxn_max_len, mol_ph, mol_token_id, is_gal=True):
        self.text_max_len = text_max_len
        self.rxn_max_len = rxn_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.is_gal = is_gal
        
    def __call__(self, batch):
        rxn_ids, graphs, texts, input_prompt = zip(*batch)
        inputs = input_prompt
        graphs = [graph for graph_batch in graphs for graph in graph_batch]
        if graphs:
            graphs = self.collater(graphs)
        input_prompt = [smiles_handler(p, self.mol_ph, self.is_gal)[0] for p in input_prompt]

        ## deal with prompt
        self.tokenizer.padding_side = 'left'
        input_prompt_tokens = self.tokenizer(input_prompt, 
                                              truncation=True, 
                                              padding='longest', 
                                              add_special_tokens=True,
                                              max_length=self.rxn_max_len, 
                                              return_tensors='pt', 
                                              return_attention_mask=True)
        
        is_mol_token = input_prompt_tokens.input_ids == self.mol_token_id
        input_prompt_tokens['is_mol_token'] = is_mol_token
        return rxn_ids, graphs, input_prompt_tokens, texts, inputs

class TuneDM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        smi_max_len: int = 128,
        rxn_max_len: int = 128,
        tokenizer=None,
        downstream_task='action',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.rxn_max_len = rxn_max_len
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        DownstreamDataset = {
            'action': ActionDataset,
            'synthesis': SynthesisDataset,
            'caption': CaptionDataset,
            'chebi': ChEBI_dataset,
        }[downstream_task]
        ds_args = {
            'use_graph': not args.disable_graphs,
            'disable_graph_cache': args.disable_graph_cache,
            'smiles_type': args.smiles_type,
        }
        if downstream_task == 'action':
            ds_args['predict_rxn_condition'] = args.predict_rxn_condition
        if downstream_task == 'synthesis':
            ds_args['roundrobin_train'] = args.roundrobin_train
            ds_args['test_subset'] = args.test_subset
        self.train_dataset = DownstreamDataset(root, 'train', smi_max_len, **ds_args)
        self.val_dataset = DownstreamDataset(root, 'valid', smi_max_len, **ds_args)
        self.test_dataset = DownstreamDataset(root, 'test', smi_max_len, **ds_args)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        self.is_gal = args.opt_model.find('galactica') >= 0
        self.use_graph = not args.disable_graphs
        self.is_t5 = args.opt_model.find('t5') >= 0
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.args.roundrobin_train:
            self.train_dataset.reload_data()
        if hasattr(self.train_dataset, 'renew_r_smiles'):
            self.train_dataset.renew_r_smiles()
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(
                tokenizer=self.tokenizer, 
                text_max_len=self.text_max_len, 
                rxn_max_len=self.rxn_max_len, 
                mol_ph=self.mol_ph_token, 
                mol_token_id=self.mol_token_id, 
                is_gal=self.is_gal,
                use_graph=self.use_graph,
                use_qa_pair=not self.is_t5,
            ),
        )
        return loader

    def val_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(
                tokenizer=self.tokenizer, 
                text_max_len=self.text_max_len, 
                rxn_max_len=self.rxn_max_len, 
                mol_ph=self.mol_ph_token, 
                mol_token_id=self.mol_token_id, 
                is_gal=self.is_gal
            ),
        )
        return [test_loader]
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(
                tokenizer=self.tokenizer, 
                text_max_len=self.text_max_len, 
                rxn_max_len=self.rxn_max_len, 
                mol_ph=self.mol_ph_token, 
                mol_token_id=self.mol_token_id, 
                is_gal=self.is_gal
            ),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(
                tokenizer=self.tokenizer, 
                text_max_len=self.text_max_len, 
                rxn_max_len=self.rxn_max_len, 
                mol_ph=self.mol_ph_token, 
                mol_token_id=self.mol_token_id, 
                is_gal=self.is_gal
            ),
        )
        return loader
    