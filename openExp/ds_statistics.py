from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

name = 'processed'
data_path = f'data/processed/processed.json'

def main():
    rxn_dict_list = json_to_object(data_path)
    action_len = []
    action_names = []
    reactant_len = []
    os.makedirs(f'figs/{name}', exist_ok=True)
    for rxn_dict in rxn_dict_list:
        action_len.append(len(rxn_dict['actions'].split(' ; ')))
        action_names.extend([i.split(' ', 1)[0].strip('.') for i in rxn_dict['actions'].split(' ; ')])
        reactant_len.append(len(rxn_dict['REACTANT']))

    data_series = pd.Series(reactant_len)
    frequency = data_series.value_counts()
    frequency /= len(reactant_len)
    plt.figure(figsize=(10, 3))
    plt.bar(frequency.index, frequency.values)
    plt.axvline(x=data_series.mean(), color='r', linestyle='--')
    plt.xlim([0, 6])
    plt.savefig(f'figs/{name}/ReactantLen_barplot.jpg')
    plt.clf()

    data_series = pd.Series(action_names)
    frequency = data_series.value_counts()
    frequency /= len(action_names)
    fig = plt.figure(figsize=(10, 4))
    plt.bar(frequency.index, frequency.values)
    plt.xticks(rotation='vertical')
    fig.autofmt_xdate(rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f'figs/{name}/ActNames_barplot.jpg')
    plt.clf()

    data_series = pd.Series(action_len)
    frequency = data_series.value_counts()
    frequency /= len(action_len)
    plt.figure(figsize=(10, 3))
    plt.bar(frequency.index, frequency.values)
    plt.axvline(x=data_series.mean(), color='r', linestyle='--')
    plt.xlim([4, 24.5])
    plt.savefig(f'figs/{name}/ActLen_barplot.jpg')


def main2():
    rxn_dict_list = json_to_object(data_path)
    score_list = []
    for rxn_dict in rxn_dict_list:
        score_list.append(rxn_dict['score'])

    counts, bin_edges = np.histogram(score_list, bins=100)
    plt.figure(figsize=(10, 3))
    plt.bar(bin_edges[:-1], counts, width = bin_edges[1]-bin_edges[0])
    plt.savefig('figs/Scores_barplot.jpg')
    plt.clf()

if __name__=='__main__':
    main()
    # main2()
    