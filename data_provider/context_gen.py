import random
import os
import numpy as np
import argparse
import json
from collections import defaultdict
from matplotlib import pyplot as plt
from collections import Counter
from .data_utils import json_read

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

class Reaction_Cluster:
    def __init__(self, root, reaction_filename, reverse_ratio=0.5):
        self.root = root
        self.reaction_data = json_read(os.path.join(self.root, reaction_filename))
        self.property_data = json_read(os.path.join(self.root, 'Abstract_property.json'))
        self.mol_property_map = {d['canon_smiles']: d for d in self.property_data}
        self.reverse_ratio = reverse_ratio
        self.rxn_mols_attr = defaultdict(lambda:{
            'freq': 0,
            'occurrence': 0,
            'in_caption': False,
        })

        self._read_reaction_mols() # add `valid_mols` in each rxn_dict
        self.mol_counter = Counter(mol for rxn_dict in self.reaction_data for mol in rxn_dict['valid_mols'])
        self._calculate_Pr() # calculate P(r), add `weight` in each rxn_dict
        self._calculate_Pir() # calculate P(i|r), add `mol_weight` in each rxn_dict

    def _read_reaction_mols(self):
        self.valid_rxn_indices = []
        for rxn_id, rxn_dict in enumerate(self.reaction_data):
            mol_role_map = {}
            for key in ['REACTANT', 'CATALYST', 'SOLVENT', 'PRODUCT']:
                for m in rxn_dict[key]:
                    if m in mol_role_map:
                        continue
                    if m in self.mol_property_map:
                        mol_role_map[m] = key
            valid_mols = []
            for mol in mol_role_map:
                assert mol in self.mol_property_map # this is garanteed by the above if statement
                if 'abstract' not in self.mol_property_map[mol]:
                    continue
                valid_mols.append(mol) # here the molecules should be in the R, C, S, P order.
            if len(valid_mols) > 0:
                self.valid_rxn_indices.append(rxn_id)
            rxn_dict['valid_mols'] = valid_mols
            rxn_dict['mol_role_map'] = mol_role_map

    def _calculate_Pr(self):
        total_weights = 0
        for rxn_dict in self.reaction_data:
            rxn_weight = sum([1/self.mol_counter[mol] for mol in rxn_dict['valid_mols']])
            rxn_dict['weight'] = rxn_weight
            total_weights += rxn_weight
        for rxn_dict in self.reaction_data:
            rxn_dict['weight'] = rxn_dict['weight'] / total_weights

    def _calculate_Pir(self):
        for rxn_dict in self.reaction_data:
            mol_weight = {}
            for mol in rxn_dict['valid_mols']:
                mol_weight[mol] = 1/self.mol_counter[mol]
            total_weight = sum(mol_weight.values())
            rxn_dict['mol_weight'] = {m:w/total_weight for m, w in mol_weight.items()}

    def choose_mol(self, valid_mols, k=4, weights=None):
        if k>=len(valid_mols):
            sampled_indices = list(range(len(valid_mols)))
        else:
            sampled_indices = np.random.choice(len(valid_mols), k, replace=False, p=weights)
            sampled_indices = list(sampled_indices)
        sampled_indices = sorted(sampled_indices)
        if random.random() < self.reverse_ratio: # reverse the indices with reverse_ratio chance.
            sampled_indices.reverse()
        sampled_mols = [valid_mols[i] for i in sampled_indices]
        return sampled_mols

    def sample_mol_batch(self, index=None, k=4):
        if index is None:
            index = self.sample_rxn_index(1)[0]
        assert index < len(self.reaction_data)
        rxn = self.reaction_data[index]
        valid_mols, weights = zip(*rxn['mol_weight'].items())

        sampled_mols = self.choose_mol(valid_mols, k=k, weights=weights)
        mol_property_batch = []
        for mol in sampled_mols:
            mol_property = self.mol_property_map[mol]
            mol_role = rxn['mol_role_map'][mol]
            mol_property['role'] = mol_role
            mol_property_batch.append(mol_property)
        if 'rsmiles_map' in rxn:
            rsmiles_map = random.choice(rxn['rsmiles_map'])
            for mol_property in mol_property_batch:
                canon_smiles = mol_property['canon_smiles']
                if canon_smiles in rsmiles_map:
                    mol_property['r_smiles'] = rsmiles_map[canon_smiles]
        return mol_property_batch

    def sample_rxn_index(self, num_samples):
        indices = range(len(self.reaction_data))
        weights = [d['weight'] for d in self.reaction_data]
        return np.random.choice(indices, num_samples, replace=False, p=weights)

    def __call__(self, rxn_num=1000, k=4):
        sampled_indices = self.sample_rxn_index(rxn_num)
        sampled_batch = [self.sample_mol_batch(idx, k=k) for idx in sampled_indices]
        return sampled_batch

    def generate_batch_uniform_rxn(self, rxn_num=1000, k=4):
        assert rxn_num <= len(self.valid_rxn_indices)
        sampled_rxn_indices = random.sample(self.valid_rxn_indices, rxn_num)
        sampled_batch = []
        for rxn_id in sampled_rxn_indices:
            rxn = self.reaction_data[rxn_id]
            sampled_mols = self.choose_mol(rxn['valid_mols'], k=k, weights=None)
            mol_property_batch = []
            for mol in sampled_mols:
                mol_property = self.mol_property_map[mol]
                mol_role = rxn['mol_role_map'][mol]
                mol_property['role'] = mol_role
                mol_property_batch.append(mol_property)
            sampled_batch.append(mol_property_batch)
        return sampled_batch

    def generate_batch_uniform_mol(self, rxn_num=1000, k=4):
        valid_mols = list(self.mol_counter.elements())
        assert rxn_num*k <= len(valid_mols)
        sampled_batch = []
        sampled_mol_ids = random.sample(range(len(valid_mols)), rxn_num*k)
        for i in range(rxn_num):
            sampled_batch.append([self.mol_property_map[valid_mols[mol_id]] for mol_id in sampled_mol_ids[i*k:(i+1)*k]])
        return sampled_batch

    def generate_batch_single(self, rxn_num=1000):
        valid_mols = list(self.mol_counter.elements())
        sampled_mols = random.sample(valid_mols, rxn_num)
        total_valid_mols = [[self.mol_property_map[mol]] for mol in sampled_mols]
        return total_valid_mols

    # visaulize probability for molecules in caption dataset.
    def visualize_mol_distribution(self):
        prob_dict = {mol:0.0 for mol in self.mol_property_map.keys()}
        N = len(prob_dict)
        M = len(self.reaction_data)
        assert N == len(self.mol_property_map)
        print(f'Number of molecules in Caption Dataset: {N}')
        print(f'Number of Reactions in Reaction Dataset: {M}')

        # prob distribution for molecules
        for rxn_dict in self.reaction_data:
            for mol, weight in rxn_dict['mol_weight'].items():
                prob_dict[mol] += weight * rxn_dict['weight']
        # sum of prob_dict.values() should already be 1.
        prob_values = np.array(list(prob_dict.values()))
        prob_values *= N

        # prob distribution for reactions
        rxn_weights = np.array([d['weight'] for d in self.reaction_data])
        # sum of rxn_weights should already be 1.
        rxn_weights *= M

        return prob_values, rxn_weights

    # visaulize the frequency for molecules in caption dataset.
    def visualize_mol_frequency(self, rxn_num=1000, k=4, epochs=100):
        sampled_mols_counter = Counter()
        sampled_rxns_counter = Counter()
        for _ in range(epochs):
            rxn_indices = self.sample_rxn_index(rxn_num)
            sampled_rxns_counter.update(rxn_indices)
            for index in rxn_indices:
                rxn = self.reaction_data[index]
                if len(rxn['valid_mols']) ==0:
                    continue
                valid_mols, weights = zip(*rxn['mol_weight'].items())
                mol_batch = self.choose_mol(valid_mols, k=k, weights=weights)
                sampled_mols_counter.update(mol_batch)
        sampled_mols_count = np.array([c for _, c in sorted(sampled_mols_counter.items())])
        sampled_rxns_count = np.array([c for _, c in sorted(sampled_rxns_counter.items())])
        return sampled_mols_count, sampled_rxns_count

    def _randomly(self, func, *args, **kwargs):
        # make fake weights and backup the weights
        for rxn_dict in self.reaction_data:
            rxn_dict['weight_bak'] = rxn_dict['weight']
            rxn_dict['weight'] = 1/len(self.reaction_data)
            rxn_dict['mol_weight_bak'] = rxn_dict['mol_weight']
            rxn_dict['mol_weight'] = {m:1/len(rxn_dict['mol_weight']) for m in rxn_dict['mol_weight']}

        # run the function
        result = func(*args, **kwargs)

        # weights recovery
        for rxn_dict in self.reaction_data:
            rxn_dict['weight'] = rxn_dict['weight_bak']
            del rxn_dict['weight_bak']
            rxn_dict['mol_weight'] = rxn_dict['mol_weight_bak']
            del rxn_dict['mol_weight_bak']

        return result
