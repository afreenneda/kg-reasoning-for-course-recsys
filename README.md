# PERLM: Faithful Path-Based Explainable Recommendation via Language
This repository contains the source code of the submitted paper "PERLM: Faithful Path-Based Explainable Recommendation via Language
Modeling over Knowledge Graphs".

If this repository IS useful for your research, we would appreciate an acknowledgment by citing our paper:

```
Balloccu, Giacomo, Ludovico Boratto, Christian Cancedda, Gianni Fenu, and Mirko Marras. 
"Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph." arXiv preprint arXiv:2310.16452 (2023).
```

## Requirements
- Python 3.8

Install the required packages:
```pip install -r requirements.txt```

Download the datasets and the **embeddings**(to run the plm-rec implementation) from the **data.zip** and **embedding-weights.zip** archive at the drive repository: https://drive.google.com/drive/folders/1e0uFWb6iJ6MXHtslZsqV8qRYC0Pl_AR7?usp=sharing
Then extract both **data.zip** and **embedding-weights.zip** inside the **top level** of the repository (i.e. the level in which setup.py is located). 

## Usage
##### Design philosophy: 
The experiments which are reported in the paper can be run with ease by means of the provided bash scripts.
This holds both for dataset generation and model training.
To access the lower level details, one can directly use the python scripts which are called by the same bash scripts.

Note: all experiments have been run with fixed seed in order to ease reproducibility of the results.

### 0. Install the repository
From the top-level (i.e. the folder which contains setup.py and the pathlm folder)
Run:
```sh
pip install . 
```
### 1. Path Dataset generation
To create the `preprocessed/mapping` folder needed by the random walk algorithm, run from the top level:

```
python pathlm/data_mappers/map_dataset.py --data <dataset_name> --model pearlm
```

To generate all datasets, run from the top level:
```sh
source build_datasets.sh
```
Each dataset is generated by the pipeline described in 'create_dataset.sh' which is in charge of:
1. Generation of a dataset of **at most X unique paths per user**
2. Concatenation of the results into a single .txt file
3. (Optional) Pruning of the concatenated .txt file (This is only useful if the start entity is chosen instead of the standard 'USER')
4. Move of the concatenated and pruned .txt file into the 'data' folder which is used to tokenize and train the models

### 2. Bulk Training
From the top-level (i.e. the folder which contains setup.py and the pathlm folder).
Install the repository with ```pip install .```

Then, proceed according to the chosen experiment to run as described below.
Each bash script can be customised as desired in order to run alternative experiments
##### PERLM
To bulk train PERLM, run from the top level:
```sh
CUDA_DEVICE_NUM=0
source run_perlm_experiments.sh $CUDA_DEVICE_NUM
```


##### PLM-Rec
To train PLM-Rec, run from the top level:
```sh
CUDA_DEVICE_NUM=0
source run_plm-rec_experiments.sh $CUDA_DEVICE_NUM
```

### 3. Training
Before training a specific model, tokenize the dataset, running from the top level:
```
python pathlm/models/lm/tokenize_dataset.py --data <dataset_name> --sample_size <sample_size>
```

##### PERLM
To train a specific PEARLM, run from the top level:
```
python pathlm/models/lm/pearlm_main.py --data <dataset_name> --model <base-clm-model> --sample_size <sample_size>
```
##### PLM-Rec
To train a specific PLM, run from the top level:
```
python pathlm/models/lm/plm_main.py --data <dataset_name> --model <base-clm-model> --sample_size <sample_size> 
```

