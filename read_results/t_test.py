from utils import *
import scipy.stats as stats

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser")

    parser.add_argument('--name', default='none', type=str)
    parser.add_argument('--path_exp', default=None, type=str)
    parser.add_argument('--path_ref', default=None, type=str)
    parser.add_argument('--use_tok', default=False, action='store_true')
    args = parser.parse_args()
    return args

def read_dataset(data_path):
    print(f'Reading {data_path}...')
    with open(data_path, 'r', encoding='utf-8') as f:
        test_tgt = [json.loads(line) for line in f.readlines()]
    print(f'{len(test_tgt)} samples read.')
    gt_list = [i['targets'] for i in test_tgt]
    pred_list = [i['predictions'] for i in test_tgt]
    return gt_list, pred_list

def t_test(mean_exp, std_exp, mean_ref, std_ref, n):
    numerator = mean_exp - mean_ref
    denominator = np.sqrt((std_exp**2 / n) + (std_ref**2 / n))
    t_statistic = numerator / denominator
    df = (((std_exp**2 / n) + (std_ref**2 / n))**2) / (((std_exp**2 / n)**2 / (n-1)) + ((std_ref**2 / n)**2 / (n-1)))

    p_value = 2 * stats.t.sf(np.abs(t_statistic), df)
    return t_statistic, p_value

def read_result(args):
    gt_list_exp, pred_list_exp = read_dataset(args.path_exp)
    gt_list_ref, pred_list_ref = read_dataset(args.path_ref)
    calculator = Metric_calculator()
    result_exp = calculator.get_result_list(gt_list_exp, pred_list_exp, args.use_tok)
    result_ref = calculator.get_result_list(gt_list_ref, pred_list_ref, args.use_tok)

    for key in ['bleu2', 'bleu4', 'rouge_1', 'rouge_2', 'rouge_l', 'lev_score', 'meteor_score']:
        if not isinstance(result_exp[key], list):
            continue
        levene_s, levene_p = stats.levene(result_exp[key], result_ref[key])
        t_stat, p_val = stats.ttest_ind(result_exp[key], result_ref[key], equal_var=(levene_p > 0.05))
        print(f'{key} (mean={float(np.mean(result_exp[key])):.4f}, levene p={levene_p:.3f}):\t{t_stat:.6f}\t{p_val}')

if __name__ == "__main__":
    args=parse_args()
    read_result(args)