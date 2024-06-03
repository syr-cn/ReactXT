# OpenExp

This folder contains the code to process the dataset OpenExp.

![fig1](../figures/openexp.jpg)

We collected 2.2M chemical reactions and associated experiment procedures from [USPTO](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873) dataset and the Open Reaction Database. Then we apply LMs to translate the unstructured experiment procedures into structured descriptions. This repo contains code for the remaining data filtering and processing steps.

The **latest version of OpenExp** can be downloaded from [here](https://osf.io/3dv4k).

## Download and Reproduce

If you want to reproduce the data processing, the collected data and the molecular name mapping rules can be downloaded from [here](https://osf.io/gzqa7).

Here's a brief description of how to run the code:

```bash
pip install -r requirements.txt
bash run.sh
```

This dataset is built under help of [smiles2actions](https://github.com/rxn4chemistry/smiles2actions), [STOUT](https://github.com/Kohulan/Smiles-TO-iUpac-Translator) and [TextChemT5](https://github.com/GT4SD/multitask_text_and_chemistry_t5).