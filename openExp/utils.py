import pandas as pd
import os
import re
import json
import rdkit
from rdkit import Chem
from rdkit.Chem import CanonSmiles
from rdkit.Chem import MolFromSmiles, MolToSmiles
from collections import defaultdict
import Levenshtein as lev
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

from paragraph2actions.postprocessing.filter_postprocessor import FilterPostprocessor
from paragraph2actions.postprocessing.noaction_postprocessor import NoActionPostprocessor
from paragraph2actions.postprocessing.postprocessor_combiner import PostprocessorCombiner
from paragraph2actions.postprocessing.wait_postprocessor import WaitPostprocessor
from paragraph2actions.postprocessing.drysolution_postprocessor import DrysolutionPostprocessor
from paragraph2actions.postprocessing.duplicate_actions_postprocessor import DuplicateActionsPostprocessor
from paragraph2actions.postprocessing.purify_postprocessor import RemovePurifyPostprocessor
from paragraph2actions.postprocessing.same_temperature_postprocessor import SameTemperaturePostprocessor
from paragraph2actions.postprocessing.initial_makesolution_postprocessor import InitialMakesolutionPostprocessor
from paragraph2actions.readable_converter import ReadableConverter
from paragraph2actions.actions import *

def remove_duplicates_preserve_order(seq):
    return [x for i, x in enumerate(seq) if seq.index(x) == i]

def normed_dis(smi1, smi2):
    if smi1 is None or smi2 is None:
        return 1
    return 2*lev.distance(smi1, smi2)/(len(smi1)+len(smi2))

def remove_nonsense_words(name):
    invalid_phrase={
        'solution',
        'saturated', 'sat', 'satd',
        'aqueous','aq', 'aq.',
        'gas', 
    }
    if r'\u' in name:
        name = name.encode('utf-8').decode('unicode_escape')
    name = name.split(' ')
    name = [i for i in name if i.lower() not in invalid_phrase]
    name = ' '.join(name)
    return name

def smiles_split(string, separator='.'):
    string = str(string)
    mols = []
    for smi in string.split(separator):
        mol = MolFromSmiles(smi)
        if mol is None:
            continue  # Skip invalid SMILES strings
        mols.append(mol)

    parts = []
    current_part = []
    charge_count = 0

    for mol in mols:
        charge = Chem.GetFormalCharge(mol)
        if charge==0:
            if current_part:
                smiles = '.'.join([MolToSmiles(m) for m in current_part])
                smiles = CanonSmiles(smiles)
                parts.append(smiles)
                current_part = []
                charge_count = 0
            parts.append(MolToSmiles(mol))
        else:
            charge_count += charge
            current_part.append(mol)
            if charge_count == 0:
                smiles = '.'.join([MolToSmiles(m) for m in current_part])
                smiles = CanonSmiles(smiles)
                parts.append(smiles)
                current_part = []
                charge_count = 0
    if current_part:
        smiles = '.'.join([MolToSmiles(m) for m in current_part])
        smiles = CanonSmiles(smiles)
        parts.append(smiles)

    return parts

def canon_file(f_name):
    map_dict = json_to_object(f_name)
    corr_dict, err_dict = {}, {}
    for k in map_dict:
        try:
            smiles = map_dict[k]
            corr_dict[k] = Chem.CanonSmiles(smiles)
        except:
            err_dict[k] = map_dict[k]
    return corr_dict, err_dict

def canon_dir(dir_path):
    fail_dict = {}
    for f_name in os.listdir(dir_path):
        corr_dict, err_dict = canon_file(os.path.join(dir_path, f_name))
        object_to_json(corr_dict, os.path.join(dir_path, f_name))
        fail_dict.update(err_dict)
    object_to_json(fail_dict, os.path.join(dir_path, 'fail.json'))

def canon_csv(f_name):
    def smiles_process(smi):
        if not isinstance(smi, str):
            return ''
        try:
            if '.' not in smi:
                return Chem.CanonSmiles(smi)
            return '.'.join([Chem.CanonSmiles(s) for s in smiles_split(smi, separator='.')])
        except:
            return ''

    df = pd.read_csv(f_name)
    df['REACTANT'] = df['REACTANT'].apply(smiles_process)
    df['PRODUCT'] = df['PRODUCT'].apply(smiles_process)
    df['CATALYST'] = df['CATALYST'].apply(smiles_process)
    df['SOLVENT'] = df['SOLVENT'].apply(smiles_process)
    df.to_csv(f_name, index=False)

def create_common_name_map():
    freq_dict_path = 'rules/iupac_names_freq.json'
    name_map_dir = 'rules/name_map'
    allowed_dir = 'rules/allowed'
    
    freq_dict = json_to_object(freq_dict_path)
    name_map_dict = make_namemap_rules(name_map_dir)
    allowed_dict = make_namemap_rules(allowed_dir)
    smiles_name_map_dict = {}
    common_name_map = {}
    
    def get_key_from_value(dictionary, target_value):
        for key, value in dictionary.items():
            if value == target_value:
                return remove_nonsense_words(key)
        assert False
    
    for key in freq_dict:
        value = name_map_dict[key]
        if value is None:
            common_name_map[key] = remove_nonsense_words(key)
            continue
        if value in allowed_dict.values():
            k = get_key_from_value(allowed_dict, value)
            smiles_name_map_dict[value] = k
        if value not in smiles_name_map_dict: # get first occurrance
            smiles_name_map_dict[value] = remove_nonsense_words(key)
        common_name_map[key] = smiles_name_map_dict[value] # map to first occurrance of the material.
    object_to_json(common_name_map, 'rules/common_name_map.json')

def json_to_object(path):
    if os.path.getsize(path) == 0:
        return None

    with open(path, encoding='utf-8') as f:
        result = json.load(f)
    return result

def object_to_json(obj, path:str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def make_namemap_rules(dir_path):
    result = dict()
    for file_name in os.listdir(dir_path):
        if not file_name.endswith('.json'):
            continue
        updating_dict = json_to_object(os.path.join(dir_path, file_name))
        if isinstance(updating_dict, dict):
            for key, value in updating_dict.items():
                if value is not None:
                    result[key] = value
        elif isinstance(updating_dict, list):
            for key in updating_dict:
                if key not in result:
                    result[key] = value
        else:
            raise NotImplementedError
    result = defaultdict(lambda:None, result)
    return result

def action_is_empty(action):
    attrs = [i for i in dir(action) if not (i.startswith('_') or i=='action_name')]
    for att in attrs:
        if getattr(action, att) is None:
            continue
        if getattr(action.__class__(), att, None) == getattr(action, att):
            continue
        return False
    return True