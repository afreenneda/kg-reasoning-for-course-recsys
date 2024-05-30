## Explainable Recommendation through Knowledge Graphs in Education

## Requirements
- Python 3.8
- Install the required packages:
```sh
pip install -r requirements.txt 
```

Download the datasets from the Google drive: https://drive.google.com/drive/folders/1MAxH1HbowFU7uJeegtVbMZgVTIuVAmMq?usp=drive_link
Then extract **data.zip** inside the **top level** of the repository (i.e. the level in which setup.py is located). 

The base workflow is:

### 0. Install the repository
From the top-level (i.e. the folder which contains setup.py and the pathlm folder)
Run:
```sh
pip install .
```
### 1.Models
- DATASET_NAME {coco, mooper, mooccube}
- MODEL_NAMES:
- traditional methods{nfm, bprmf}
- knowledge aware methods{cke, kgat}
- path based methods{pgpr, cafe}
- casual language models{plm, pearlm}

### 2. Dataset generation
To create the `preprocessed/mapping` folder needed by the algorithm, run from the top level:
```sh
python pathlm/data_mappers/map_dataset.py --data <dataset_name> --model <model_name>
```

Note: For Casual Language Model, each dataset is generated by the pipeline described in 'create_dataset.sh' which is in charge of:
1. Generation of a dataset of **at most X unique paths per user**
2. Concatenation of the results into a single .txt file
3. (Optional) Pruning of the concatenated .txt file (This is only useful if the start entity is chosen instead of the standard 'USER')
4. Move of the concatenated and pruned .txt file into the 'data' folder which is used to tokenize and train the models

### 3. Bulk Training
From the top-level (i.e. the folder which contains setup.py and the pathlm folder).
Install the repository with ```pip install .```

Then, proceed according to the chosen experiment to run as described below.
Each bash script can be customised as desired in order to run alternative experiments

##### TransE Embeddings
Methods such as (PGPR, CAFE, PLM, PEARLM) have dependencies with the TransE Embeddings, run mapper to map in standard format and transe model for TransE representation
```sh
python pathlm/data_mappers/map_dataset.py --data <dataset_name> --model transe
```
```sh
python pathlm/models/embeddings/train_transe_model.py --dataset <dataset_name>
```

##### Run path based method{PGPR, CAFE}
Before training, preprocessed data for the respective model, run from the top level:
#### PGPR
```sh
python pathlm/models/rl/PGPR/preprocess.py --dataset <dataset_name>
```
train agent by executing:
```sh
python pathlm/models/rl/PGPR/train_agent.py --dataset <dataset_name>
```
#### CAFE
```sh
python pathlm/models/rl/CAFE/preprocess.py --dataset <dataset_name>
```
learn the policy by executing:
```sh
 python pathlm/models/rl/CAFE/train_neural_symbol.py --dataset <dataset_name>
```

##### Run traditional method{NFM, BPRMF}
Before training map data to model's standard form, run from the top level:
#### NFM
```sh
python pathlm/models/traditional/NFM/main_nfm.py --dataset <dataset_name>
```
#### BPRMF
```sh
python pathlm/models/traditional/BPRMF/main_bprmf.py --dataset <dataset_name>
```

##### Run knowledge aware methods{CKE, KGAT}
Before training map data to model's standard form, run from the top level:
#### CKE
```sh
python pathlm/models/knowledge_aware/CKE/main.py --dataset <dataset_name>
```
#### KGAT
```sh
python pathlm/models/knowledge_aware/KGAT/main.py --dataset <dataset_name>
```

##### Casual Language Modeling for Path Reasoning
### Path Sampling
Sampling can be employed by running 
```sh
 create_dataset.sh
```
script and specify the positional parameters:
1. dataset: {coco, mooper, mooccube}
2. sample_size: represent the amount of paths sampled for each user
3. n_hop: represent the fixed hop size for the paths sampled
4. n_proc: number of processors to employ for multiprocessing operations

```sh
bash create_dataset.sh {dataset_name} {sample_size} {n_hop} {n_proc}
```

### Tokenizing
Before training a PLM or PEARLM, tokenize the dataset, running from the top level:
```sh
python pathlm/models/lm/tokenize_dataset.py --data {dataset_name} --sample_size {sample_size} --n_hop {n_hop} --nproc {n_proc}
```

##### PLM-Rec
To train PLM-Rec, run from the top level:
```sh
python pathlm/models/lm/plm_main.py --data {dataset_name} --sample_size {sample_size} --n_hop {n_hop} --nproc {n_proc}
```


##### PERLM
To train PEARLM, run from the top level:
```
python pathlm/models/lm/pearlm_main.py --data {dataset_name} --sample_size {sample_size} --n_hop {n_hop} --nproc {n_proc}
```

##### Metrics
This list collects the formulas and short descriptions of the metrics currently implemented by the evaluation module. All recommendation metrics are calculated for a user top-k.

#### Recommendation Quality
- NDCG: The extent to which the recommended products are useful for the user. Weights the position of the item in the top-k. $$NDCG@k=\frac{DCG@k}{IDCG@k}$$ where: $$DCG@k=\sum_{i=1}^{k}\frac{rel_i}{log_2(i+1)}=rel_1+\sum_{i=2}^{k}\frac{rel_i}{log_2(i+1)}$$ $$IDCG@k = \text{sort descending}(rel)$$
- MMR: The extent to which the first recommended product is useful for the user. $$MMR = \frac{1}{\text{first hit position}}$$
- Coverage: Proportion of items recommended among all the item catalog. $$\frac{| \text{Unique Recommended items}|}{| \text{Items in Catalog} |}$$
- Diversity: Proportion of genres covered by the recommended items among the recommended items. $$\frac{| \text{Unique Genres} |}{| \text{Recommended items} |}$$
- Novelty: Inverse of popularity of the items recommended to the user $$\frac{\sum_{i \in I}| 1 - \text{Pop}(i) |}{| \text{Recommended items} |}$$
- Serendipity: Proportion of items which may be surprising for the user, calculated as the the proportion of items recommended by the benchmarked models that are not recommended by a prevedible baseline. In our case the baseline was MostPop. $$\frac{| \text{Recommended items} \cup \text{Recommended items by most pop} |}{| \text{Recommended items} |}$$

#### Hyper parameters
The hyper parameters that have been considered in the grid search are listed below, alongside a brief description and its codename used in the experiments:

PGPR
-hidden : number of hidden units of each layer of the shared embedding neural network, that is used as a backbone by the actor and the critic prediction heads
-ent_weight: weight of the entropy loss that quantifies entropy in the action distribution

CAFE
-embed_size: size of the embedding of entities and relations for neural modules employed by CAFE's symbolic model
-rank_weight: weight of the ranking loss component in the total loss.

KGAT
-adj_type: weighting technique applied to each connection on the KG adjacency matrix A
--bilateral (bi), pre and post multiply A by the inverse of the square root of the diagonal matrix of out degrees of each node
--single (si), pre multiply A by the inverse of the of the diagonal matrix of out degrees of each node
-embed_size: size of user and entity embeddings
-kge_size: size of the relation embeddings

CKE
-adj_type (weighting technique applied to each connection on the KG adjacency matrix A )
--bilateral (bi), pre and post multiply A by the inverse of the square root of the diagonal matrix of out degrees of each node
--single (si), pre multiply A by the inverse of the of the diagonal matrix of out degrees of each node
-embed_size (size of user and entity embeddings)
-kge_size (size of the relation embeddings)

PLM
-num_epochs: Max number of epochs.
-model: The base huggingface model from where eredit the architecture one from {distilgpt2, gpt2, gpt2-large}
-batch_size: Batch size.
-sample_size: Dataset sample size (to dermine which dataset to use)
-n_hop: Dataset hop size (to dermine which dataset to use)
-logit_processor_type: Decoding strategy empty for PLM
-n_seq_infer: Number of sequences generated for each user should be > k

PEARLM
-num_epochs: Max number of epochs.
-model: The base huggingface model from where eredit the architecture one from {distilgpt2, gpt2, gpt2-large}
-batch_size: Batch size.
-sample_size: Dataset sample size (to dermine which dataset to use)
-n_hop: Dataset hop size (to dermine which dataset to use)
-logit_processor_type: Decoding strategy 'gcd' for PEARLM
-n_seq_infer: Number of sequences generated for each user should be > k

Optimal hyper parameters:
Each model is configured with a set of optimal hyper parameters, according to the dataset upon which it is trained. In order to train a given model with customized hyper parameters, it is necessary to set them from command line. Each can be set by adding as new command line arguments the pair (--param_name param_value) while also specifying the model_name and the dataset to use.

PGPR
hidden [512,256]
ent_weight 0.001

CAFE
embed_size 200
rank_weight 1.0

KGAT
adj_type si
embed_size 64
kge_size 64

CKE
adj_type si
embed_size 64
kge_size 64

PLM
-num_epochs: 5
-model: distilgpt2
-batch_size: 512
-sample_size: 250
-n_hop: 3
-logit_processor_type: Decoding strategy empty for PLM
-n_seq_infer: 30

PEARLM
-num_epochs: 5
-model: distilgpt2
-batch_size: 512
-sample_size: 250
-n_hop: 3
-logit_processor_type: Decoding strategy 'gcd'
-n_seq_infer: 30
