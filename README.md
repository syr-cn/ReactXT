# ReactXT: Understanding Molecular “Reaction-ship” via Reaction-Contextualized Molecule-Text Pretraining

This repo contains the pytorch implementation of the paper "ReactXT: Understanding Molecular “Reaction-ship” via Reaction-Contextualized Molecule-Text Pretraining" (ACL 2024).

Authors: Zhiyuan Liu*, Yaorui Shi*, An Zhang, Sihang Li, Enzhi Zhang, Xiang Wang, Kenji Kawaguchi, Tat-Seng Chua

## Comparison to previous molecule-text generative modeling methods

![fig1](./figures/comparison.jpg)


## Framework of ReactXT

![fig1](./figures/frameworks.jpg)


## Requirements

Our environment is detailed in `environment.yml`. To create a new environment `reactxt`, run the following command:

```bash
conda env create -f environment.yml
```

## Reproduce the results

Our datasets and pretrained model checkpoints can be downloaded from [here](https://osf.io/e68v4/files/osfstorage)

### Reaction-Contextualized Molecule-Text Pretraining

Please run the following command to perform ReactXT pretraining based on the MolCA checkpoint.
The original MolCA checkpoint can be downloaded from [MolCA](https://github.com/eltociear/MolCA)

```bash
bash scripts/run_pretrain.sh
```

### Finetuning on downstream tasks

1. Experimental Procedure Prediction on OpenExp

```bash
bash scripts/run_action.sh
```

2. Molecule Captioning on PubChem324k and CheBI-20

```bash
bash scripts/run_caption.sh
bash scripts/run_chebi.sh
```

3. Retro-synthesis Prediction on USPTO-50k

```bash
bash scripts/run_retro.sh
```


# Citation

If you find this paper useful, please cite us:

```bib
@inproceedings{liu2024reactxt,
    title={ReactXT: Understanding Molecular “Reaction-ship” via Reaction-Contextualized Molecule-Text Pretraining},
    author={Liu, Zhiyuan and Shi, Yaorui and Zhang, An and Li, Sihang and Zhang, Enzhi and Wang, Xiang and Kawaguchi, Kenji and Chua, Tat-Seng},
    booktitle={{ACL}},
    publisher={Association for Computational Linguistics},
    year={2024},
    url={https://openreview.net/forum?id=V-ejDfLiwe}
}
```
