# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from data_provider.molecule_abstract_dataset import MoleculeAbstract
import re
from transformers import BatchEncoding

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


def tokenize_and_merge_batched_qa_pairs(tokenizer, qa_pairs_list, max_length):
    tokenized_batches = {
        'input_ids': [],
        'attention_mask': []
    }
    for qa_pairs in qa_pairs_list:
        max_length_per_qa = max_length // len(qa_pairs)
        batch_input_ids = []
        batch_attention_mask = []
        for qa in qa_pairs:
            # here qa should be string
            tokens = tokenizer(qa,
                            truncation=True,
                            padding=False,
                            add_special_tokens=False,
                            max_length=max_length_per_qa,
                            return_tensors='pt',
                            return_attention_mask=True)
            batch_input_ids.extend(tokens['input_ids'].squeeze().tolist())
            batch_attention_mask.extend(tokens['attention_mask'].squeeze().tolist())

        # Pad the batch to max_length
        padding_length = max_length - len(batch_input_ids)
        batch_input_ids.extend([tokenizer.pad_token_id] * padding_length)
        batch_attention_mask.extend([0] * padding_length)

        tokenized_batches['input_ids'].append(torch.tensor(batch_input_ids).unsqueeze(0))
        tokenized_batches['attention_mask'].append(torch.tensor(batch_attention_mask).unsqueeze(0))

    tokenized_batches['input_ids'] = torch.cat(tokenized_batches['input_ids'], dim=0)
    tokenized_batches['attention_mask'] = torch.cat(tokenized_batches['attention_mask'], dim=0)

    tokenized_batch = BatchEncoding(data=tokenized_batches, tensor_type='pt')
    return tokenized_batch

class TrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, is_gal=True, disable_graphs=False):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.is_gal = is_gal
        self.disable_graphs = disable_graphs

    def __call__(self, batch):
        graphs, mol_prompt, text_prompt = zip(*batch)
        if not self.disable_graphs:
            graphs = [graph for graph_batch in graphs for graph in graph_batch]
            graphs = self.collater(graphs)

        qa_pairs = []
        for mol_batch, text_batch in zip(mol_prompt, text_prompt):
            qa_list = []
            for mol_prompt, text_prompt in zip(mol_batch, text_batch):
                smiles_prompt = smiles_handler(mol_prompt, self.mol_ph, self.is_gal)[0]
                qa_list.append(f'{smiles_prompt} {text_prompt}')
            qa_pairs.append(qa_list)

        self.tokenizer.padding_side = 'right'
        qa_batch = tokenize_and_merge_batched_qa_pairs(self.tokenizer, qa_pairs, self.text_max_len)

        is_mol_token = qa_batch.input_ids == self.mol_token_id
        qa_batch['is_mol_token'] = is_mol_token

        return graphs, qa_batch

class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, is_gal=True, disable_graphs=False, last_only=False):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.is_gal = is_gal
        self.disable_graphs = disable_graphs
        self.last_only = last_only

    def __call__(self, batch):
        graphs, mol_prompt, text_prompt = zip(*batch)
        rxn_ids = [0 for i in range(len(mol_prompt))]
        if self.last_only:
            mol_prompt = [[mol_batch[-1]] for mol_batch in mol_prompt]
            text_prompt = [[text_batch[-1]] for text_batch in text_prompt]
            graphs = [[graph_batch[-1]] for graph_batch in graphs]
        if not self.disable_graphs:
            graphs = [graph for graph_batch in graphs for graph in graph_batch]
            graphs = self.collater(graphs)

        input_text, output_text = [], []
        for mol_batch, text_batch in zip(mol_prompt, text_prompt):
            qa_list = []
            for mol_prompt, text_prompt in list(zip(mol_batch, text_batch))[:-1]:
                smiles_prompt = smiles_handler(mol_prompt, self.mol_ph, self.is_gal)[0]
                qa_list.append(f'{smiles_prompt} {text_prompt}')
            qa_list.append(f'{smiles_handler(mol_batch[-1], self.mol_ph, self.is_gal)[0]} ')
            output_text.append(text_batch[-1])
            input_text.append(qa_list)

        self.tokenizer.padding_side = 'right'
        input_batch = tokenize_and_merge_batched_qa_pairs(self.tokenizer, input_text, self.text_max_len)

        is_mol_token = input_batch.input_ids == self.mol_token_id
        input_batch['is_mol_token'] = is_mol_token
        
        return rxn_ids, graphs, input_batch, output_text, input_text


class PretrainDM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        rxn_max_len: int = 128,
        smi_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.rxn_max_len = rxn_max_len
        self.pretrain_dataset = MoleculeAbstract(
            root,
            rxn_num=args.pretrain_rxn_num,
            rxn_batch_size=args.rxn_batch_size,
            smi_max_len=smi_max_len,
            disable_graph_cache=args.disable_graph_cache,
            context_style=args.context_style,
            disable_graphs=args.disable_graphs,
            use_caption_dataset=args.pretrain_use_caption,
            caption_batch_num=args.caption_batch_num,
            synthesis_datasetpath=args.pretrain_synthesis_path,
            synthesis_batch_num=args.synthesis_batch_num,
            reverse_ratio=args.reverse_ratio,
            enable_abstract=not args.disable_abstract,
            enable_property=not args.disable_property,
            smiles_type=args.smiles_type,
        )
        self.test_dataset = MoleculeAbstract(
            root,
            rxn_num=args.pretrain_rxn_num,
            rxn_batch_size=args.rxn_batch_size,
            smi_max_len=smi_max_len,
            disable_graph_cache=args.disable_graph_cache,
            context_style=args.context_style,
            disable_graphs=args.disable_graphs,
            use_caption_dataset=args.pretrain_use_caption,
            caption_batch_num=args.caption_batch_num,
            reverse_ratio=args.reverse_ratio,
            enable_abstract=not args.disable_abstract,
            enable_property=not args.disable_property,
            smiles_type=args.smiles_type,
            mode='test',
        )
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        self.is_gal = args.opt_model.find('galactica') >= 0
        self.disable_graphs = args.disable_graphs
        self.last_only = args.pretrain_eval_last_only

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        self.pretrain_dataset.reload_data_list()
        loader = DataLoader(
            self.pretrain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(
                tokenizer=self.tokenizer,
                text_max_len=self.text_max_len,
                mol_ph=self.mol_ph_token,
                mol_token_id=self.mol_token_id,
                is_gal=self.is_gal,
                disable_graphs=self.disable_graphs,
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
                mol_ph=self.mol_ph_token, 
                mol_token_id=self.mol_token_id, 
                is_gal=self.is_gal,
                disable_graphs=self.disable_graphs,
                last_only=self.last_only,
            ),
        )
        return [test_loader]

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/action_data')
        parser.add_argument('--context_style', type=str, default='weighted_rxn', choices=['weighted_rxn', 'uniform_rxn', 'uniform_mol', 'single_mol', 'hybrid'])
        parser.add_argument('--rxn_max_len', type=int, default=512)
        parser.add_argument('--text_max_len', type=int, default=512)
        parser.add_argument('--smi_max_len', type=int, default=128)
        parser.add_argument('--pretrain_rxn_num', type=int, default=50000)
        parser.add_argument('--reverse_ratio', type=float, default=0.5, help='ratio of reversed reactions (retro reactions)')
        parser.add_argument('--disable_abstract', action='store_true', default=False)
        parser.add_argument('--disable_property', action='store_true', default=False)
        parser.add_argument('--pretrain_use_caption', action='store_true', default=False)
        parser.add_argument('--caption_batch_num', type=int, default=5000)
        parser.add_argument('--pretrain_synthesis_path', type=str, default=None)
        parser.add_argument('--synthesis_batch_num', type=int, default=5000)
        parser.add_argument('--rxn_batch_size', type=int, default=4)
        parser.add_argument('--roundrobin_train', action='store_true', default=False)
        parser.add_argument('--test_subset', type=int, default=-1)
        parser.add_argument('--pretrain_eval_last_only', default=False, action='store_true')
        parser.add_argument('--prompt', type=str, default=None)
        return parent_parser
