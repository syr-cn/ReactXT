import torch
from torch_geometric.data import Data
from ogb.utils import smiles2graph
from rdkit import Chem
import random
import os
import json
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from .r_smiles import multi_process
import multiprocessing

def reformat_smiles(smiles, smiles_type='default'):
    if not smiles:
        return None
    if smiles_type == 'default':
        return smiles
    elif smiles_type=='canonical':
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    elif smiles_type=='restricted':
        mol = Chem.MolFromSmiles(smiles)
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    elif smiles_type=='unrestricted':
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    elif smiles_type=='r_smiles':
        # the implementation of root-aligned smiles is in r_smiles.py
        return smiles
    else:
        raise NotImplementedError(f"smiles_type {smiles_type} not implemented")

def json_read(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def json_write(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def format_float_from_string(s):
    try:
        float_value = float(s)
        return f'{float_value:.2f}'
    except ValueError:
        return s

def make_abstract(mol_dict, abstract_max_len=256, property_max_len=256):
    prompt = ''
    if 'abstract' in mol_dict:
        abstract_string = mol_dict['abstract'][:abstract_max_len]
        prompt += f'[Abstract] {abstract_string} '

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
            if len(property_string+key_value_string) > property_max_len:
                break
            property_string += key_value_string
    if property_string:
        property_string = property_string[:property_max_len]
        prompt += f'[Properties] {property_string}. '
    return prompt

def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

import re
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")


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

def generate_rsmiles(reactants, products, augmentation=20):
    """
        reactants: list of N, reactant smiles
        products: list of N, product smiles
        augmentation: int, number of augmentations
        
        return: list of N x augmentation
    """
    data = [{
        'reactant': r.strip().replace(' ', ''),
        'product': p.strip().replace(' ', ''),
        'augmentation': augmentation,
        'root_aligned': True,
    } for r, p in zip(reactants, products)]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(func=multi_process,iterable=data)
    product_smiles = [smi for r in results for smi in r['src_data']]
    reactant_smiles = [smi for r in results for smi in r['tgt_data']]
    return reactant_smiles, product_smiles