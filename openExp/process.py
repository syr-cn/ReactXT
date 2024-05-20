import pandas as pd
import os
import re
import json
import rdkit
from rdkit import Chem
from utils import *
import argparse
import random

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def my_decorator(func):
    def wrapper(self, *args, **kwargs):

        def safe_func(s, x):
            try:
                return func(s, x)
            except Exception as error:
                if isinstance(error, (NotImplementedError, AttributeError)):
                    raise error
                return None

        old_len = len(self.rxn_list)
        self.rxn_list = [result for result in (safe_func(self, x) for x in self.rxn_list) if result]
        print(f'{func.__name__.ljust(40)}\t {old_len} -> {len(self.rxn_list)} ({old_len-len(self.rxn_list)} [{(old_len-len(self.rxn_list))/self.input_len*100:.2f}%] removed).')
        return
    return wrapper

DICT_KEYS = ['REACTANT','PRODUCT','CATALYST','SOLVENT','actions', 'action_string', 'score']
EXTRACT_KEYS = ['extracted_molecules', 'extracted_temperature', 'extracted_duration']
class RxnFilter():
    def __init__(
            self,
            root: str,
            name: str,
            input_filename: str,
            score_threshold: float,
            fuzzy_match_threshold: float,
            action_to_string_seperator: str=" ; ",
        ):
        self.root = root
        self.file_path = os.path.join(root, input_filename)
        self.name = name

        self.name_map = make_namemap_rules('rules/name_map')
        self.allowed_reagents = make_namemap_rules('rules/allowed')
        self.common_name_map = json_to_object('rules/common_name_map.json')
        self.invalid_reagents = json_to_object('rules/invalid_molecules.json')

        for key, value in self.common_name_map.items():
            if self.name_map[key] is None:
                self.name_map[key] = self.name_map[value]

        self.converter = ReadableConverter()
        self.postprocessor = PostprocessorCombiner([
            FilterPostprocessor(),
            WaitPostprocessor(),
            DrysolutionPostprocessor(),
            DuplicateActionsPostprocessor(),
            RemovePurifyPostprocessor(),
            SameTemperaturePostprocessor(),
            InitialMakesolutionPostprocessor(),
            NoActionPostprocessor(),
        ])
        self.action_to_string_seperator = action_to_string_seperator
        self.score_threshold = score_threshold
        self.fuzzy_match_threshold = fuzzy_match_threshold

        if 'csv' in self.file_path:
            df = pd.read_csv(self.file_path)
            self.rxn_list = df[['REACTANT','PRODUCT','CATALYST','SOLVENT','actions', 'score', 'source']].to_dict('records')
        else:
            self.rxn_list = json_to_object(self.file_path)
        self.input_len = len(self.rxn_list)

        print(f'reading {len(self.rxn_list)} reactions')

    # Two special type of methods: DO and REMOVE
    # DO methods process the data, and those cannot be processed is omitted
    # REMOVE methods drop invalid actions based on some rules
    def process(self):
        process_funcs = [
            self.DO_split_smiles,
            self.DO_string_to_actions,
            self.REMOVE_low_score,
            self.REMOVE_multiple_products,
            # self.REMOVE_single_reactant,
            self.DO_format_temperature,
            self.Do_format_yield,
            self.DO_drop_invalid_materials,
            self.REMOVE_duplicate_smiles,

            self.DO_map_common_material_names,
            self.DO_extract_molecules_fuzzy, # use fuzzy match for molecule-precursor mapping
            self.REMOVE_unmmaped_precursors,
            self.REMOVE_unmmaped_materials,
            self.DO_extract_temperatures,
            self.DO_extract_durations,
            # self.DO_remove_chemical_quantity,
            # self.DO_remove_product_quantity,

            self.REMOVE_InvalidActions,
            self.DO_combine_similar_actions,
            self.DO_filter_NoAction,
            self.REMOVE_FollowOtherProcedure,

            self.REMOVE_PR_duplicate_smiles,
            self.REMOVE_RR_duplicate_smiles,
            self.REMOVE_duplicate_rxns,

            self.DO_post_process,
            self.REMOVE_short_actions,
            self.DO_action_to_strings,
        ]
        for method in process_funcs:
            method()

    @my_decorator
    def DO_canon_smiles(self, rxn_dict, separator='.'):
        rxn_dict['REACTANT'] = smiles_split(rxn_dict['REACTANT'])
        rxn_dict['PRODUCT'] = smiles_split(rxn_dict['PRODUCT'])

        rxn_dict['CATALYST'] = smiles_split(rxn_dict['CATALYST']) if isinstance(rxn_dict['CATALYST'], str) else []
        rxn_dict['SOLVENT'] = smiles_split(rxn_dict['SOLVENT']) if isinstance(rxn_dict['SOLVENT'], str) else []

        for role in ['REACTANT', 'PRODUCT', 'CATALYST', 'SOLVENT']:
            rxn_dict[role] = remove_duplicates_preserve_order(rxn_dict[role])
        return rxn_dict

    @my_decorator
    def DO_split_smiles(self, rxn_dict, separator='.'):
        # if the molecules have been canonicalized already, we can just split it
        rxn_dict['REACTANT'] = smiles_split(rxn_dict['REACTANT'], separator=separator)
        rxn_dict['PRODUCT'] = smiles_split(rxn_dict['PRODUCT'], separator=separator)

        rxn_dict['CATALYST'] = smiles_split(rxn_dict['CATALYST'], separator=separator) if isinstance(rxn_dict['CATALYST'], str) else []
        rxn_dict['SOLVENT'] = smiles_split(rxn_dict['SOLVENT'], separator=separator) if isinstance(rxn_dict['SOLVENT'], str) else []

        for role in ['REACTANT', 'PRODUCT', 'CATALYST', 'SOLVENT']:
            rxn_dict[role] = remove_duplicates_preserve_order(rxn_dict[role])
        return rxn_dict

    @my_decorator
    def DO_string_to_actions(self, rxn_dict):
        rxn_dict['actions'] = self.converter.string_to_actions(rxn_dict['actions'])
        return rxn_dict

    @my_decorator
    def REMOVE_low_score(self, rxn_dict):
        if self.score_threshold:
            assert rxn_dict['score'] <= self.score_threshold
        return rxn_dict

    @my_decorator
    def REMOVE_multiple_products(self, rxn_dict):
        assert len(rxn_dict['PRODUCT']) < 2
        return rxn_dict

    @my_decorator
    def DO_format_temperature(self, rxn_dict):
        actions = []
        for action in rxn_dict['actions']:
            if hasattr(action, 'temperature') and action.temperature:
                action.temperature = action.temperature.replace('° C', ' °C')
            actions.append(action)
        rxn_dict['actions'] = actions
        return rxn_dict

    @my_decorator
    def Do_format_yield(self, rxn_dict):
        yield_list = [a for a in rxn_dict['actions'] if isinstance(a, Yield)]
        non_yield_list = [a for a in rxn_dict['actions'] if not isinstance(a, Yield)]
        if len(yield_list) == 0:
            non_yield_list.append(Yield(material=Chemical(rxn_dict['PRODUCT'][0])))
        else:
            assert len(yield_list)==1
            non_yield_list.extend(yield_list)
        rxn_dict['actions'] = non_yield_list
        return rxn_dict

    @my_decorator
    def DO_extract_molecules(self, rxn_dict):
        # This function is deprecated
        # set self.fuzzy_match_threshold=0 to do exact match.
        token_format = '${}$'
        exist_index = [j for i in EXTRACT_KEYS if i in rxn_dict for j in rxn_dict[i].values()]
        start_index = len(exist_index) if '$-1$' in exist_index else len(exist_index) + 1
        # start_index = 1
        param_dict, token_actions = dict(), []
        get_token = lambda: token_format.format(len(param_dict) + start_index)
        raw_names = []

        for action in rxn_dict['actions']:
            # Special handling for yield actions
            if isinstance(action, Yield):
                yield_mol_name = rxn_dict['PRODUCT'][0]
                action.material.name = param_dict[yield_mol_name] = token_format.format(-1)

            # for the actions with single material
            elif hasattr(action, 'material') and action.material:
                if isinstance(action.material, Chemical):
                    material_name = action.material.name
                    raw_names.append(material_name)

                    if material_name not in param_dict:
                        mapped_name = self.name_map[material_name]
                        if mapped_name and mapped_name in (rxn_dict['REACTANT']+rxn_dict['CATALYST']+rxn_dict['SOLVENT']):
                            param_dict[mapped_name] = get_token()

                    if material_name in param_dict: # successful map, change name to special token
                        action.material.name = param_dict[material_name]
                    elif material_name in self.allowed_reagents: # specail materials, nothing needs to be changed
                        pass
                    else: # regard as invalid if molecule not found in above two sources
                        pass
                        # allowed_count -= 1
                        # assert allowed_count > 0, f'allowed_count = {allowed_count}'

            # for the actions with multiple materials
            elif hasattr(action, 'materials') and action.materials:
                materials = []
                for material in action.materials:
                    material_name = material.name
                    raw_names.append(material_name)

                    if material_name not in param_dict:
                        mapped_name = self.name_map[material_name]
                        if mapped_name and mapped_name in (rxn_dict['REACTANT']+rxn_dict['CATALYST']+rxn_dict['SOLVENT']):
                            param_dict[mapped_name] = get_token()

                    if material_name in param_dict: # successful map, change name to special token
                        material.name = param_dict[material_name]
                    elif material_name in self.allowed_reagents: # specail materials, nothing needs to be changed
                        pass
                    else: # regard as invalid if molecule not found in above two sources
                        pass
                        # allowed_count -= 1
                        # assert allowed_count > 0, f'allowed_count = {allowed_count}'
                        

                    materials.append(material)
                action.materials = materials

            token_actions.append(action)

        rxn_dict['molecules'] = {name: self.name_map[name] for name in raw_names}
        for mol in rxn_dict['REACTANT']:
            if mol not in param_dict:
                assert 0
        rxn_dict['CATALYST'] = [mol for mol in rxn_dict['CATALYST'] if mol in param_dict]
        rxn_dict['SOLVENT'] = [mol for mol in rxn_dict['SOLVENT'] if mol in param_dict]

        rxn_dict['extracted_molecules'] = param_dict
        rxn_dict['actions'] = token_actions
        return rxn_dict

    @my_decorator
    def DO_extract_molecules_fuzzy(self, rxn_dict):
        # Fuzzy matching for molecules and materials.
        token_format = '${}$'
        exist_index = [j for i in EXTRACT_KEYS if i in rxn_dict for j in rxn_dict[i].values()]
        start_index = len(exist_index) if '$-1$' in exist_index else len(exist_index) + 1
        # start_index = 1
        param_dict, token_actions = dict(), []
        get_token = lambda: token_format.format(len(param_dict) + start_index)

        # get all materials
        material_name_smiles_dict = {}
        for action in rxn_dict['actions']:
            if isinstance(action, Yield):
                continue
            elif hasattr(action, 'material') and action.material:
                if isinstance(action.material, Chemical):
                    material_name = action.material.name
                    material_smiles = self.name_map[material_name]
                    material_name_smiles_dict[material_name] = material_smiles
            elif hasattr(action, 'materials') and action.materials:
                for material in action.materials:
                    material_name = material.name
                    material_smiles = self.name_map[material_name]
                    material_name_smiles_dict[material_name] = material_smiles

        # Fuzzy match between the name-smiles dict and reactants
        mapped_name_smiles_dict = {}
        unmapped_material_list = []
        for key, value in material_name_smiles_dict.items():
            if value and (value in (rxn_dict['REACTANT']+rxn_dict['CATALYST']+rxn_dict['SOLVENT'])): # matched with precursors
                mapped_name_smiles_dict[key] = value
            elif key in self.allowed_reagents: # matched with common compound, no need for mapping
                continue
            else:
                unmapped_material_list.append(key)
        unmapped_reactant_list = [i for i in rxn_dict['REACTANT'] if i not in mapped_name_smiles_dict.values()]

        if self.fuzzy_match_threshold>0:
            for reactant in unmapped_reactant_list:
                min_dis = 1e9
                for material_name in unmapped_material_list:
                    dis = normed_dis(reactant, material_name_smiles_dict[material_name])
                    if dis < min_dis:
                        min_dis = dis
                        best_material = material_name
                if min_dis < self.fuzzy_match_threshold: # successful fuzzy match
                    unmapped_material_list.remove(best_material)
                    unmapped_reactant_list.remove(reactant)
                    mapped_name_smiles_dict[best_material] = reactant
        
        # Caution! this may cause some unwanted mapping!
        # if len(unmapped_reactant_list)==1:
        #     longest_material = max(unmapped_material_list, key = len)
        #     if len(longest_material)>20 and material_name_smiles_dict[longest_material] is None:
        #         mapped_name_smiles_dict[longest_material] = unmapped_reactant_list[0]

        param_dict = {}
        # use fuzzy_matched results to do name map
        for action in rxn_dict['actions']:
            # Special handling for yield actions
            if isinstance(action, Yield):
                yield_mol_name = rxn_dict['PRODUCT'][0]
                action.material.name = param_dict[yield_mol_name] = token_format.format(-1)

            # for the actions with single material
            elif hasattr(action, 'material') and action.material:
                if isinstance(action.material, Chemical):
                    material_name = action.material.name

                    if material_name in mapped_name_smiles_dict:
                        mapped_name = mapped_name_smiles_dict[material_name]
                        if mapped_name not in param_dict:
                            param_dict[mapped_name] = get_token()

                    if material_name in mapped_name_smiles_dict: # successful map, change name to special token
                        action.material.name = param_dict[mapped_name_smiles_dict[material_name]]

            # for the actions with multiple materials
            elif hasattr(action, 'materials') and action.materials:
                materials = []
                for material in action.materials:
                    material_name = material.name

                    if material_name in mapped_name_smiles_dict:
                        mapped_name = mapped_name_smiles_dict[material_name]
                        if mapped_name not in param_dict:
                            param_dict[mapped_name] = get_token()

                    if material_name in mapped_name_smiles_dict: # successful map, change name to special token
                        material.name = param_dict[mapped_name_smiles_dict[material_name]]
                    materials.append(material)
                action.materials = materials

            token_actions.append(action)

        rxn_dict['extracted_molecules'] = param_dict
        rxn_dict['molecules'] = material_name_smiles_dict
        rxn_dict['actions'] = token_actions

        return rxn_dict

    @my_decorator
    def REMOVE_unmmaped_precursors(self, rxn_dict):
        param_dict = rxn_dict['extracted_molecules']
        for mol in rxn_dict['REACTANT']:
            if mol not in param_dict:
                assert 0
        rxn_dict['CATALYST'] = [mol for mol in rxn_dict['CATALYST'] if mol in param_dict]
        rxn_dict['SOLVENT'] = [mol for mol in rxn_dict['SOLVENT'] if mol in param_dict]

        return rxn_dict

    @my_decorator
    def REMOVE_unmmaped_materials(self, rxn_dict):
        max_fail_times = 1
        precursor_smiles_list = rxn_dict['REACTANT']+rxn_dict['CATALYST']+rxn_dict['SOLVENT']+rxn_dict['PRODUCT']
        precursor_smiles_list = rxn_dict['extracted_molecules']
        for key, value in rxn_dict['molecules'].items():
            if value in precursor_smiles_list:
                continue
            if key in self.allowed_reagents:
                continue
            if self.common_name_map[key] in self.allowed_reagents:
                continue
            max_fail_times -=1
            assert max_fail_times > 0

        return rxn_dict

    @my_decorator
    def DO_drop_invalid_materials(self, rxn_dict): # remove the unmapped materials if it's not an obvious molecule
        def filter_action(action):
            if isinstance(action, Yield):
                return action
            elif hasattr(action, 'material') and action.material:
                if isinstance(action.material, Chemical):
                    if action.material.name in self.invalid_reagents:
                        return None
            elif hasattr(action, 'solvent') and action.solvent:
                if isinstance(action.solvent, Chemical):
                    if action.solvent.name in self.invalid_reagents:
                        return None
            elif hasattr(action, 'materials') and action.materials:
                for material in action.materials:
                    if material.name in self.invalid_reagents:
                        return None
            return action

        action_list = [filter_action(a) for a in rxn_dict['actions']]
        action_list = [a for a in action_list if a]
        rxn_dict['actions'] = action_list
        return rxn_dict

    @my_decorator
    def DO_map_common_material_names(self, rxn_dict):
        # must be used after action matching
        action_list = []
        for action in rxn_dict['actions']:
            if isinstance(action, Yield):
                pass
            elif hasattr(action, 'material') and action.material:
                if isinstance(action.material, Chemical):
                    material_name = action.material.name
                    if '$' not in material_name:
                        if material_name in self.common_name_map:
                            action.material.name = self.common_name_map[material_name]
            elif hasattr(action, 'solvent') and action.solvent:
                if isinstance(action.solvent, Chemical):
                    material_name = action.solvent.name
                    if '$' not in material_name:
                        if material_name in self.common_name_map:
                            action.solvent.name = self.common_name_map[material_name]
            elif hasattr(action, 'materials') and action.materials:
                materials = []
                for material in action.materials:
                    material_name = material.name
                    if material_name in self.common_name_map:
                        material.name = self.common_name_map[material_name]
                    materials.append(material)
                action.materials = materials
            action_list.append(action)
        rxn_dict['actions'] = action_list
        return rxn_dict

    @my_decorator
    def DO_extract_temperatures(self, rxn_dict):
        token_format = '#{}#'
        exist_index = [j for i in EXTRACT_KEYS if i in rxn_dict for j in rxn_dict[i].values()]
        start_index = len(exist_index) if '$-1$' in exist_index else len(exist_index) + 1
        # start_index = 1
        param_dict, token_actions = dict(), []
        get_token = lambda: token_format.format(len(param_dict) + start_index)

        for action in rxn_dict['actions']:
            if hasattr(action, 'temperature') and action.temperature:
                temp = action.temperature
                if temp not in param_dict:
                    param_dict[temp] = get_token()
                action.temperature = param_dict[temp]

            token_actions.append(action)

        rxn_dict['extracted_temperature'] = param_dict
        rxn_dict['actions'] = token_actions
        return rxn_dict

    @my_decorator
    def DO_extract_durations(self, rxn_dict):
        token_format = '@{}@'
        exist_index = [j for i in EXTRACT_KEYS if i in rxn_dict for j in rxn_dict[i].values()]
        start_index = len(exist_index) if '$-1$' in exist_index else len(exist_index) + 1
        # start_index = 1
        param_dict, token_actions = dict(), []
        get_token = lambda: token_format.format(len(param_dict) + start_index)

        for action in rxn_dict['actions']:
            if hasattr(action, 'duration') and action.duration:
                duration = action.duration
                if duration not in param_dict:
                    param_dict[duration] = get_token()
                action.duration = param_dict[duration]

            token_actions.append(action)

        rxn_dict['extracted_duration'] = param_dict
        rxn_dict['actions'] = token_actions
        return rxn_dict

    @my_decorator
    def DO_remove_chemical_quantity(self, rxn_dict):
        action_list = []
        for act in rxn_dict['actions']:
            if hasattr(act, 'material'):
                act.material.quantity=[]
            elif hasattr(act, 'materials'):
                materials = []
                for material in act.materials:
                    material.quantity = []
                    materials.append(material)
                act.materials = materials
            action_list.append(act)
        rxn_dict['actions'] = action_list
        return rxn_dict

    @my_decorator
    def DO_remove_product_quantity(self, rxn_dict):
        action_list = []
        for act in rxn_dict['actions']:
            if isinstance(act, Yield):
                act.material.quantity=[]
            action_list.append(act)
        rxn_dict['actions'] = action_list
        return rxn_dict

    @my_decorator
    def REMOVE_single_reactant(self, rxn_dict):
        assert len(rxn_dict['REACTANT'])>1
        return rxn_dict

    @my_decorator
    def REMOVE_InvalidActions(self, rxn_dict):
        assert all((not isinstance(a, InvalidAction)) for a in rxn_dict['actions'])
        return rxn_dict

    @my_decorator
    def DO_combine_similar_actions(self, rxn_dict):
        nullable_actions = (Concentrate, Degas, DrySolid, DrySolution, Filter, Microwave, Reflux, Sonicate, Stir)
        action_list = [rxn_dict['actions'][0]]
        for action in rxn_dict['actions'][1:]:
            if isinstance(action, nullable_actions) and type(action)==type(action_list[-1]):
                if action_is_empty(action):
                    continue
                elif action_is_empty(action_list[-1]):
                    action_list.pop()
            action_list.append(action)
        rxn_dict['actions'] = action_list
        return rxn_dict

    @my_decorator
    def DO_filter_NoAction(self, rxn_dict):
        rxn_dict['actions'] = [act for act in rxn_dict['actions'] if (not isinstance(act, NoAction))]
        return rxn_dict

    @my_decorator
    def REMOVE_FollowOtherProcedure(self, rxn_dict):
        assert all((not isinstance(a, FollowOtherProcedure)) for a in rxn_dict['actions'])
        return rxn_dict

    @my_decorator
    def REMOVE_short_actions(self, rxn_dict):
        assert len(rxn_dict['actions'])>4
        return rxn_dict

    @my_decorator
    def REMOVE_PR_duplicate_smiles(self, rxn_dict):
        smiles_set_p = set(rxn_dict['PRODUCT'])
        smiles_set_r = set(rxn_dict['REACTANT'] + rxn_dict['CATALYST'] + rxn_dict['SOLVENT'])
        smiles_set = smiles_set_p&smiles_set_r
        assert len(smiles_set)==0
        return rxn_dict

    @my_decorator
    def REMOVE_RR_duplicate_smiles(self, rxn_dict):
        smiles_set_1 = set(rxn_dict['PRODUCT'] + rxn_dict['REACTANT'])
        smiles_set_2 = set(rxn_dict['CATALYST'] + rxn_dict['SOLVENT'])
        smiles_set = smiles_set_1&smiles_set_2
        assert len(smiles_set)==0
        return rxn_dict

    @my_decorator
    def REMOVE_duplicate_smiles(self, rxn_dict):
        smiles_list = rxn_dict['PRODUCT'] + rxn_dict['REACTANT'] + rxn_dict['CATALYST'] + rxn_dict['SOLVENT']
        smiles_set = set(smiles_list)
        assert len(smiles_set)==len(smiles_list)
        return rxn_dict

    def REMOVE_duplicate_rxns(self):
        old_len = len(self.rxn_list)
        new_list = []
        record_rxns = set()

        for rxn_dict in self.rxn_list:
            element = (
                frozenset(rxn_dict['REACTANT']),
                frozenset(rxn_dict['PRODUCT']),
                frozenset(rxn_dict['CATALYST']),
                frozenset(rxn_dict['SOLVENT']),
            )
            if element in record_rxns:
                continue
            record_rxns.add(element)
            new_list.append(rxn_dict)

        del self.rxn_list
        self.rxn_list = new_list
        print(f'{"REMOVE_duplicate_rxns".ljust(40)}\t {old_len} -> {len(self.rxn_list)} ({old_len-len(self.rxn_list)} [{(old_len-len(self.rxn_list))/self.input_len*100:.2f}%] removed).')

    @my_decorator
    def DO_post_process(self, rxn_dict):
        rxn_dict['actions'] = self.postprocessor.postprocess(rxn_dict['actions'])
        return rxn_dict

    @my_decorator
    def DO_action_to_strings(self, rxn_dict):
        action_strings = (self.converter.action_to_string(a) for a in rxn_dict['actions'])
        rxn_dict['action_string'] = self.action_to_string_seperator.join(action_strings) + self.converter.end_mark
        return rxn_dict

    def run(self):
        self.process()

        processed_len = len(self.rxn_list)
        print(f'final Reacion number: {processed_len} ({processed_len/self.input_len*100:.2f}%)')

        # save result into file (csv or json)
        # self.save_csv(self.rxn_list, os.path.join(self.root, 'processed.csv'))
        self.save_json(self.rxn_list, os.path.join(self.root, self.name, 'processed.json'))

        train_frac = 0.8
        valid_frac = 0.1
        random.shuffle(self.rxn_list)
        train_size = int(len(self.rxn_list) * train_frac)
        valid_size = int(len(self.rxn_list) * valid_frac)
        
        train_list = self.rxn_list[:train_size]
        valid_list = self.rxn_list[train_size:train_size+valid_size]
        test_list = self.rxn_list[train_size+valid_size:]

        self.save_json(train_list, os.path.join(self.root, self.name, 'train.json'))
        self.save_json(valid_list, os.path.join(self.root, self.name, 'valid.json'))
        self.save_json(test_list, os.path.join(self.root, self.name, 'test.json'))

    # deprecated
    def save_csv(self, rxn_dict_list, path):
        processed_list = []
        for idx, rxn_dict in enumerate(rxn_dict_list):
            processed_dict = {
                'index': idx,
                'REACTANT': '.'.join(rxn_dict['REACTANT']),
                'PRODUCT': '.'.join(rxn_dict['PRODUCT']),
                'CATALYST': '.'.join(rxn_dict['CATALYST']),
                'SOLVENT': '.'.join(rxn_dict['SOLVENT']),
                'actions': rxn_dict['action_string'],
            }
            processed_list.append(processed_dict)
        df = pd.DataFrame(processed_list)
        df.to_csv(path, index=False)

    def save_json(self, rxn_dict_list, path):
        processed_list = []
        for idx, rxn_dict in enumerate(rxn_dict_list):
            processed_dict = {
                'index': idx,
                'REACTANT': rxn_dict['REACTANT'],
                'PRODUCT': rxn_dict['PRODUCT'],
                'CATALYST': rxn_dict['CATALYST'],
                'SOLVENT': rxn_dict['SOLVENT'],
                'actions': rxn_dict['action_string'],
                'source': rxn_dict['source'],
                'extracted_molecules': rxn_dict['extracted_molecules'],
                'extracted_temperature': rxn_dict['extracted_temperature'],
                'extracted_duration': rxn_dict['extracted_duration'],
                'molecules': rxn_dict['molecules'],
                'score': rxn_dict['score'],
            }
            processed_list.append(processed_dict)
        object_to_json(processed_list, path)

def main(args):
    rxnfilter = RxnFilter(
        root=args.root,
        name=args.name,
        input_filename='rxn_actions.csv',
        score_threshold=1.0,
        fuzzy_match_threshold=0.00,
    )
    rxnfilter.run()

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser")

	# Script arguments
    parser.add_argument('--name', default='none', type=str)
    parser.add_argument('--root', default='data/processed', type=str)

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args()
    main(args)