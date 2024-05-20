# ReactXT: Understanding Molecular “Reaction-ship” via Reaction-Contextualized Molecule-Text Pretraining

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

### Reaction-Contextualized Molecule-Text Pretraining

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