from rdkit import Chem
import os
import argparse
from tqdm import tqdm
import multiprocessing
import pandas as pd
from rdkit import RDLogger
import re
from utils import *

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def extract_smiles(s):
    start_token = "[START_SMILES]"
    end_token = "[END_SMILES]"
    start_index = s.find(start_token) + len(start_token)
    end_index = s.find(end_token)
    if start_index > -1 and end_index > -1:
        return s[start_index:end_index].strip()
    return s

def canonicalize_smiles_clear_map(smiles,return_max_frag=True):
    mol = Chem.MolFromSmiles(smiles,sanitize=not opt.synthon)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        except:
            if return_max_frag:
                return '',''
            else:
                return ''
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [Chem.MolFromSmiles(smiles,sanitize=not opt.synthon) for smiles in sub_smi]
            sub_mol_size = [(sub_smi[i], len(m.GetAtoms())) for i, m in enumerate(sub_mol) if m is not None]
            if len(sub_mol_size) > 0:
                return smi, canonicalize_smiles_clear_map(sorted(sub_mol_size,key=lambda x:x[1],reverse=True)[0][0],return_max_frag=False)
            else:
                return smi, ''
        else:
            return smi
    else:
        if return_max_frag:
            return '',''
        else:
            return ''


def compute_rank(input_smiles, prediction,raw=False,alpha=1.0):
    valid_score = [[k for k in range(len(prediction[j]))] for j in range(len(prediction))]
    invalid_rates = [0 for k in range(len(prediction[0]))]
    rank = {}
    max_frag_rank = {}
    highest = {}
    if raw:
        # no test augmentation
        assert len(prediction) == 1
        for j in range(len(prediction)):
            for k in range(len(prediction[j])):
                if prediction[j][k][0] == "":
                    invalid_rates[k] += 1
            # error detection
            de_error = [i[0] for i in sorted(list(zip(prediction[j], valid_score[j])), key=lambda x: x[1]) if i[0][0] != ""]
            prediction[j] = list(set(de_error))
            prediction[j].sort(key=de_error.index)
            for k, data in enumerate(prediction[j]):
                rank[data] = 1 / (alpha * k + 1)
    else:
        for j in range(len(prediction)): # aug_num, beam_size, 2
            for k in range(len(prediction[j])):
                # predictions[i][j][k] = canonicalize_smiles_clear_map(predictions[i][j][k])
                if prediction[j][k][0] == "":
                    valid_score[j][k] = opt.beam_size + 1
                    invalid_rates[k] += 1
            # error detection and deduplication
            de_error = [i[0] for i in sorted(list(zip(prediction[j], valid_score[j])), key=lambda x: x[1]) if i[0][0] != ""]
            prediction[j] = list(set(de_error))
            prediction[j].sort(key=de_error.index)
            for k, data in enumerate(prediction[j]):
                if data in rank:
                    rank[data] += 1 / (alpha * k + 1)
                else:
                    rank[data] = 1 / (alpha * k + 1)
                if data in highest:
                    highest[data] = min(k,highest[data])
                else:
                    highest[data] = k
        for key in rank.keys():
            rank[key] += highest[key] * -1
            rank[key] += abs(len(key[0])-len(input_smiles)) * -0.2
            rank[key] += len(key[0]) * -0.2
    return rank,invalid_rates

def read_dataset(opt):
    print(f'Reading {opt.path}...')
    with open(opt.path, 'r', encoding='utf-8') as f:
        test_tgt = [json.loads(line) for line in f.readlines()]
        if opt.raw:
            test_tgt = test_tgt[::opt.augmentation]
    filtered_tgt = {}
    idx_key = 'ds_idx' if 'ds_idx' in test_tgt[0] else 'index'
    for dic in test_tgt:
        if dic[idx_key] not in filtered_tgt:
            filtered_tgt[dic[idx_key]] = dic
    test_tgt = list(filtered_tgt.values())
    test_tgt.sort(key=lambda x: x[idx_key])
    print(f'{len(test_tgt)} samples read.')
    input_list = [extract_smiles(i['input']) for i in test_tgt]
    gt_list = [i['targets'].replace('[START_SMILES]', '').replace('[END_SMILES]', '').replace('SPL1T-TH1S-Pl3A5E','').strip().replace(' ','.') for i in test_tgt]
    pred_list = [[smi.strip().replace(' ','.') for smi in i['predictions']] for i in test_tgt]
    return input_list, gt_list, pred_list

def main(opt):
    input_list, gt_list, pred_list = read_dataset(opt)
    if opt.raw:
        opt.augmentation=1
    print('Reading predictions from file ...')

    # inputs
    print("Input Length", len(gt_list))
    ras_src_smiles = input_list[::opt.augmentation]
    with multiprocessing.Pool(processes=opt.process_number) as pool:
        ras_src_smiles = pool.map(func=canonicalize_smiles_clear_map,iterable=ras_src_smiles)
    ras_src_smiles = [i[0] for i in ras_src_smiles]

    # predictions
    print("Prediction Length", len(pred_list))
    pred_lines = [i.split('>')[0] for d in pred_list for i in d]
    data_size = len(pred_lines) // (opt.augmentation * opt.beam_size) if opt.length == -1 else opt.length
    pred_lines = pred_lines[:data_size * (opt.augmentation * opt.beam_size)]
    print("Canonicalizing predictions using Process Number ",opt.process_number)
    with multiprocessing.Pool(processes=opt.process_number) as pool:
        raw_predictions = pool.map(func=canonicalize_smiles_clear_map,iterable=pred_lines)

    predictions = [[[] for j in range(opt.augmentation)] for i in range(data_size)]  # data_len x augmentation x beam_size
    for i, line in enumerate(raw_predictions):
        predictions[i // (opt.beam_size * opt.augmentation)][i % (opt.beam_size * opt.augmentation) // opt.beam_size].append(line)

    # ground truth
    print("Origin Length", len(gt_list))
    targets = [''.join(gt_list[i].strip().split(' ')) for i in tqdm(range(0,data_size * opt.augmentation,opt.augmentation))]
    with multiprocessing.Pool(processes=opt.process_number) as pool:
        targets = pool.map(func=canonicalize_smiles_clear_map, iterable=targets)

    print("predictions Length", len(predictions), len(predictions[0]), len(predictions[0][0]))
    print("Target Length", len(targets))

    ground_truth = targets
    print("Origin Target Lentgh, ", len(ground_truth))
    print("Cutted Length, ",data_size)
    print('\n')
    accuracy = [0 for j in range(opt.n_best)]
    topn_accuracy_chirality = [0 for _ in range(opt.n_best)]
    topn_accuracy_wochirality = [0 for _ in range(opt.n_best)]
    topn_accuracy_ringopening = [0 for _ in range(opt.n_best)]
    topn_accuracy_ringformation = [0 for _ in range(opt.n_best)]
    topn_accuracy_woring = [0 for _ in range(opt.n_best)]
    total_chirality = 0
    total_ringopening = 0
    total_ringformation = 0
    atomsize_topk = []
    accurate_indices = [[] for j in range(opt.n_best)]
    max_frag_accuracy = [0 for j in range(opt.n_best)]
    invalid_rates = [0 for j in range(opt.beam_size)]
    sorted_invalid_rates = [0 for j in range(opt.beam_size)]
    unique_rates = 0
    ranked_results = []

    for i in tqdm(range(len(predictions))):
        accurate_flag = False
        if opt.detailed:
            chirality_flag = False
            ringopening_flag = False
            ringformation_flag = False
            pro_mol = Chem.MolFromSmiles(ras_src_smiles[i])
            rea_mol = Chem.MolFromSmiles(ground_truth[i][0])
            try:
                pro_ringinfo = pro_mol.GetRingInfo()
                rea_ringinfo = rea_mol.GetRingInfo()
                pro_ringnum = pro_ringinfo.NumRings()
                rea_ringnum = rea_ringinfo.NumRings()
                size = len(rea_mol.GetAtoms()) - len(pro_mol.GetAtoms())
                # if (int(ras_src_smiles[i].count("@") > 0) + int(ground_truth[i][0].count("@") > 0)) == 1:
                if ras_src_smiles[i].count("@") > 0 or ground_truth[i][0].count("@") > 0:
                    total_chirality += 1
                    chirality_flag = True
                if pro_ringnum < rea_ringnum:
                    total_ringopening += 1
                    ringopening_flag = True
                if pro_ringnum > rea_ringnum:
                    total_ringformation += 1
                    ringformation_flag = True
            except:
                pass
                # continue

        inputs = input_list[i*opt.augmentation:(i+1)*opt.augmentation]
        gt = gt_list[i*opt.augmentation:(i+1)*opt.augmentation]
        rank, invalid_rate = compute_rank(ras_src_smiles[i], predictions[i], raw=opt.raw,alpha=opt.score_alpha)

        rank_ = {k[0]: v for k, v in sorted(rank.items(), key=lambda item: item[1], reverse=True)}
        if opt.detailed:
            print('Index', i)
            print('inputs', json.dumps(inputs, indent=4))
            print('targets', json.dumps(gt, indent=4))
            print('input', ras_src_smiles[i])
            print('target', targets[i][0])
            print('rank', json.dumps(rank_,indent=4))
            print('invalid_rate', json.dumps(invalid_rate,indent=4))
            print('\n')
        for j in range(opt.beam_size):
            invalid_rates[j] += invalid_rate[j]
        rank = list(zip(rank.keys(),rank.values()))
        rank.sort(key=lambda x:x[1],reverse=True)
        rank = rank[:opt.n_best]
        ranked_results.append([item[0][0] for item in rank])

        for j, item in enumerate(rank):
            if item[0][0] == ground_truth[i][0]:
                if not accurate_flag:
                    accurate_flag = True
                    accurate_indices[j].append(i)
                    for k in range(j, opt.n_best):
                        accuracy[k] += 1
                    if opt.detailed:
                        atomsize_topk.append((size,j))
                        if chirality_flag:
                            for k in range(j,opt.n_best):
                                topn_accuracy_chirality[k] += 1
                        else:
                            for k in range(j,opt.n_best):
                                topn_accuracy_wochirality[k] += 1
                        if ringopening_flag:
                            for k in range(j,opt.n_best):
                                topn_accuracy_ringopening[k] += 1
                        if ringformation_flag:
                            for k in range(j,opt.n_best):
                                topn_accuracy_ringformation[k] += 1
                        if not ringopening_flag and not ringformation_flag:
                            for k in range(j,opt.n_best):
                                topn_accuracy_woring[k] += 1

        if opt.detailed and not accurate_flag:
            atomsize_topk.append((size,opt.n_best))
        for j, item in enumerate(rank):
            if item[0][1] == ground_truth[i][1]:
                for k in range(j,opt.n_best):
                    max_frag_accuracy[k] += 1
                break
        for j in range(len(rank),opt.beam_size):
            sorted_invalid_rates[j] += 1
        unique_rates += len(rank)

    for i in range(opt.n_best):
        if i in [0,1,2,3,4,5,6,7,8,9,19,49]:
        # if i in range(10):
            print("Top-{} Acc:{:.3f}%, MaxFrag {:.3f}%,".format(i+1,accuracy[i] / data_size * 100,max_frag_accuracy[i] / data_size * 100),
                  " Invalid SMILES:{:.3f}% Sorted Invalid SMILES:{:.3f}%".format(invalid_rates[i] / data_size / opt.augmentation * 100,sorted_invalid_rates[i] / data_size / opt.augmentation * 100))
    print(' '.join([f'{accuracy[i] / data_size * 100:.3f}' for i in [0,2,4,9]]))
    print("Unique Rates:{:.3f}%".format(unique_rates / data_size / opt.beam_size * 100))

    if opt.detailed:
        print_topk = [1,3,5,10]
        save_dict = {}
        atomsize_topk.sort(key=lambda x:x[0])
        differ_now = atomsize_topk[0][0]
        topn_accuracy_bydiffer = [0 for _ in range(opt.n_best)]
        total_bydiffer = 0
        for i,item in enumerate(atomsize_topk):
            if differ_now < 11 and differ_now != item[0]:
                for j in range(opt.n_best):
                    if (j+1) in print_topk:
                        save_dict[f'top-{j+1}_size_{differ_now}'] = topn_accuracy_bydiffer[j] / total_bydiffer * 100
                        print("Top-{} Atom differ size {} Acc:{:.3f}%, Number:{:.3f}%".format(j+1,
                                              differ_now,
                                               topn_accuracy_bydiffer[j] / total_bydiffer * 100,
                                               total_bydiffer/data_size * 100))
                total_bydiffer = 0
                topn_accuracy_bydiffer = [0 for _ in range(opt.n_best)]
                differ_now = item[0]
            for k in range(item[1],opt.n_best):
                topn_accuracy_bydiffer[k] += 1
            total_bydiffer += 1
        for j in range(opt.n_best):
            if (j + 1) in print_topk:
                print("Top-{} Atom differ size {} Acc:{:.3f}%, Number:{:.3f}%".format(j + 1,
                      differ_now,
                      topn_accuracy_bydiffer[j] / total_bydiffer * 100,
                      total_bydiffer / data_size * 100))
                save_dict[f'top-{j+1}_size_{differ_now}'] = topn_accuracy_bydiffer[j] / total_bydiffer * 100

        for i in range(opt.n_best):
            if (i+1) in print_topk:
                if total_chirality > 0:
                    print("Top-{} Accuracy with chirality:{:.3f}%".format(i + 1, topn_accuracy_chirality[i] / total_chirality * 100))
                    save_dict[f'top-{i+1}_chilarity'] = topn_accuracy_chirality[i] / total_chirality * 100
                print("Top-{} Accuracy without chirality:{:.3f}%".format(i + 1, topn_accuracy_wochirality[i] / (data_size - total_chirality) * 100))
                save_dict[f'top-{i+1}_wochilarity'] = topn_accuracy_wochirality[i] / (data_size - total_chirality) * 100
                if total_ringopening > 0:
                    print("Top-{} Accuracy ring Opening:{:.3f}%".format(i + 1, topn_accuracy_ringopening[i] / total_ringopening * 100))
                    save_dict[f'top-{i+1}_ringopening'] = topn_accuracy_ringopening[i] / total_ringopening * 100
                if total_ringformation > 0:
                    print("Top-{} Accuracy ring Formation:{:.3f}%".format(i + 1, topn_accuracy_ringformation[i] / total_ringformation * 100))
                    save_dict[f'top-{i+1}_ringformation'] = topn_accuracy_ringformation[i] / total_ringformation * 100
                print("Top-{} Accuracy without ring:{:.3f}%".format(i + 1, topn_accuracy_woring[i] / (data_size - total_ringopening - total_ringformation) * 100))
                save_dict[f'top-{i+1}_wocring'] = topn_accuracy_woring[i] /  (data_size - total_ringopening - total_ringformation)* 100
        print(total_chirality)
        print(total_ringformation)
        print(total_ringopening)
        # df = pd.DataFrame(list(save_dict.items()))
        df = pd.DataFrame(save_dict,index=[0])
        df.to_csv("detailed_results.csv")
    if opt.save_accurate_indices != "":
        with open(opt.save_accurate_indices, "w") as f:
            total_accurate_indices = []
            for indices in accurate_indices:
                total_accurate_indices.extend(indices)
            total_accurate_indices.sort()

            # for index in total_accurate_indices:
            for index in accurate_indices[0]:
                f.write(str(index))
                f.write("\n")

    if opt.save_file != "":
        with open(opt.save_file,"w") as f:
            for res in ranked_results:
                for smi in res:
                    f.write(smi)
                    f.write("\n")
                for i in range(len(res),opt.n_best):
                    f.write("")
                    f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--beam_size', type=int, default=10,help='Beam size')
    parser.add_argument('--n_best', type=int, default=10,help='n best')
    parser.add_argument('--path', type=str, required=True, help="Path to file containing the predictions and ground truth.")
    parser.add_argument('--augmentation', type=int, default=20)
    parser.add_argument('--score_alpha', type=float, default=1.0)
    parser.add_argument('--length', type=int, default=-1)
    parser.add_argument('--process_number', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--synthon', action="store_true", default=False)
    parser.add_argument('--detailed', action="store_true", default=False)
    parser.add_argument('--raw', action="store_true", default=False)
    parser.add_argument('--save_file', type=str,default="")
    parser.add_argument('--save_accurate_indices', type=str,default="")

    opt = parser.parse_args()
    print(opt)
    main(opt)