from utils import *
import torch
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)

class Reaction_model:
    def __init__(self, train_list, test_list):
        self.train_list = train_list
        self.test_list = test_list

        model, tokenizer = get_default_model_and_tokenizer()
        self.rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
        
    @time_it
    def generate_random(self):
        pred = random.sample(self.train_list, k=len(self.test_list))
        pred = [i['actions'] for i in pred]
        return pred

    @time_it
    def generate_random_compatible_old(self):
        pred_list = []
        len_id_map = defaultdict(list)
        for train_rxn in self.train_list:
            len_id_map[len(train_rxn['extracted_molecules'])-1].append(train_rxn['index'])
        
        keys = sorted(k for k in len_id_map.keys())
        accumulated_counts = {}
        count = 0
        for key in keys:
            count += len(len_id_map[key])
            accumulated_counts[key] = count

        for rxn in self.test_list:
            test_token_num = len(rxn['extracted_molecules'])-1
            idx = random.randint(0, accumulated_counts[test_token_num] - 1)
            for key in keys:
                if idx < len(len_id_map[key]):
                    pred_list.append(self.train_list[len_id_map[key][idx]]['actions'])
                    break
                else:
                    idx -= len(len_id_map[key])
        return pred_list

    @time_it
    def generate_random_compatible(self):
        pred_list = []
        len_id_map = defaultdict(list)
        for train_rxn in self.train_list:
            len_id_map[len(train_rxn['extracted_molecules'])-1].append(train_rxn['index'])
        
        for rxn in self.test_list:
            mole_num = len(rxn['extracted_molecules'])-1
            pred_list.append(self.train_list[random.choice(len_id_map[mole_num])]['actions'])
        return pred_list
    
    @time_it
    def generate_nn(self, batch_size=2048):
        train_rxns = [f"{'.'.join(rxn['REACTANT'])}>>{rxn['PRODUCT'][0]}" for rxn in self.train_list]
        test_rxns = [f"{'.'.join(rxn['REACTANT'])}>>{rxn['PRODUCT'][0]}" for rxn in self.test_list]

        train_rxns_batches = [train_rxns[i:i+batch_size] for i in range(0, len(train_rxns), batch_size)]
        test_rxns_batches = [test_rxns[i:i+batch_size] for i in range(0, len(test_rxns), batch_size)]

        device = torch.device("cuda")
        train_fps = []
        for batch in tqdm(train_rxns_batches, desc='Generating fingerprints for training reactions'):
            batch_fps = self.rxnfp_generator.convert_batch(batch)
            train_fps.extend(batch_fps)
        train_fps = torch.tensor(train_fps, device=device) # N x 256

        most_similar_indices = []
        for batch in tqdm(test_rxns_batches, desc='Generating fingerprints for test reactions'):
            batch_fps = self.rxnfp_generator.convert_batch(batch)
            batch_fps = torch.tensor(batch_fps, device=device) # BS x 256
            batch_fps = batch_fps / torch.norm(batch_fps, dim=1, keepdim=True)

            similarity_matrix = torch.matmul(train_fps, batch_fps.T) # N x BS
            most_similar_indices.extend(torch.argmax(similarity_matrix, dim=0).tolist())

        return [self.train_list[i]['actions'] for i in most_similar_indices]

    def save_results(self, gt_list, pred_list, target_file):
        text_dict_list = [{
                "targets": gt,
                "indices": i,
                "predictions": pred,
            } for i, (gt, pred) in enumerate(zip(gt_list, pred_list))]

        with open(target_file, 'w') as f:
            json.dump(text_dict_list, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser")

    parser.add_argument('--name', default='none', type=str)
    parser.add_argument('--train_file', default=None, type=str)
    parser.add_argument('--test_file', default=None, type=str)
    parser.add_argument('--use_tok', default=False, action='store_true')
    args = parser.parse_args()
    return args

def read_dataset(args):
    print(f'Reading {args.train_file}...')
    with open(args.train_file, 'r', encoding='utf-8') as f:
        train_ds = json.load(f)
    print(f'{len(train_ds)} samples read.')
    print(f'Reading {args.test_file}...')
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_ds = json.load(f)
    print(f'{len(test_ds)} samples read.')
    return train_ds, test_ds

def run_baselines(args):
    set_random_seed(0)

    train_ds, test_ds = read_dataset(args)
    model = Reaction_model(train_ds, test_ds)
    calculator = Metric_calculator()
    gt_list = [i['actions'] for i in test_ds]
    
    print('Random:')
    pred_list = model.generate_random()
    calculator(gt_list, pred_list, args.use_tok)
    model.save_results(gt_list, pred_list, f'results/{args.name}/random.json')

    print('Random (compatible pattern):')
    pred_list = model.generate_random_compatible()
    calculator(gt_list, pred_list, args.use_tok)
    model.save_results(gt_list, pred_list, f'results/{args.name}/random_compatible.json')

    print('Nearest neighbor:')
    pred_list = model.generate_nn()
    calculator(gt_list, pred_list, args.use_tok)
    model.save_results(gt_list, pred_list, f'results/{args.name}/nn.json')
    # assert 0

if __name__ == "__main__":
    args=parse_args()
    run_baselines(args)