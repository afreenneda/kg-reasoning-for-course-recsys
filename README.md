## A Reproducibility Study of Explainable Course Recommendation over Knowledge Graphs: From Reinforcement Learning to Generative Modeling

> **Abstract:** *Knowledge graphs are crucial for providing transparent reasoning paths in recommender systems. Most methods generate multi-hop paths to connect users with recommended items, using intermediate entities and relationships to provide explanations. While effective, these methods are primarily evaluated in domains like entertainment and e-commerce, with limited studies in education that often yield inconsistent insights due to varied datasets, techniques, and metrics. In this work, we present a reproducibility study of knowledge-graph reasoning methods in course recommendation, a central task in education. We transform three public educational datasets into knowledge graph structures and reproduce four state-of-the-art explainable reasoning methods. Our evaluation compares these methods with non-explainable reasoning and traditional collaborative filtering baselines, focusing on metrics of utility, beyond-utility, and explainability. Results show that generative modeling methods achieve higher utility, broader catalog coverage, and more diverse explanation paths. In sparse datasets, non-explainable reasoning and traditional collaborative filtering methods perform the best in some beyond-utility metrics, with sparsity impacting traditional more than explainability-related ones. However, as dataset sparsity decreases, generative methods remain competitive from these perspectives as well.*

## Requirements
- Python 3.8
- Install the required packages:
```sh
pip install -r requirements.txt 
```

Download the processed datasets and paths from here: [[**Link**](https://drive.google.com/drive/folders/17IJopOSlbkNJLmaenAZXP93mQ0HfHovt?usp=sharing)]

Then extract **data** inside the **top level** of the repository (i.e. the level in which setup.py is located). 

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
- traditional collaborative filtering methods{nfm, bprmf}
- non-explainable KG methods{cke, kgat}
- explainable path-based methods{pgpr, cafe}
- explainable generative models{plm, pearlm}

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
Each bash script can be customized as desired in order to run alternative experiments

##### TransE Embeddings
Methods such as (PGPR, CAFE, PLM, PEARLM) have dependencies with the TransE Embeddings, run mapper to map in a standard format and transe model for TransE representation
```sh
python pathlm/data_mappers/map_dataset.py --data <dataset_name> --model transe
```
```sh
python pathlm/models/embeddings/train_transe_model.py --dataset <dataset_name>
```

##### Run path-based explainable KG method{PGPR, CAFE}
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

##### Run traditional collaborative filtering method{NFM, BPRMF}
Before training map data to model's standard form, run from the top level:
#### NFM
```sh
python pathlm/models/traditional/NFM/main_nfm.py --dataset <dataset_name>
```
#### BPRMF
```sh
python pathlm/models/traditional/BPRMF/main_bprmf.py --dataset <dataset_name>
```

##### Run non-explainable KG method{CKE, KGAT}
Before training map data to the model's standard form, run from the top level:
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
2. sample_size: represent the number of paths sampled for each user
3. n_hop: represents the fixed hop size for the paths sampled
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
- Normalized Discounted Cumulative Gain(NDCG): The extent to which the recommended products are useful for the user. Weights the position of the item in the top-k. $$NDCG@k=\frac{DCG@k}{IDCG@k}$$ where: $$DCG@k=\sum_{i=1}^{k}\frac{rel_i}{log_2(i+1)}=rel_1+\sum_{i=2}^{k}\frac{rel_i}{log_2(i+1)}$$ $$IDCG@k = \text{sort descending}(rel)$$
- Mean Reciprocal Rank(MRR): The extent to which the model recommends the most relevant courses at the top ranks. $$MRR = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}$$
- Coverage: Proportion of items recommended among all the item catalog. $$\frac{| \text{Unique Recommended items}|}{| \text{Items in Catalog} |}$$
- Diversity: Proportion of genres covered by the recommended items among the recommended items. $$\frac{| \text{Unique Genres} |}{| \text{Recommended items} |}$$
- Novelty: Inverse of the popularity of the items recommended to the user $$\frac{\sum_{i \in I}| 1 - \text{Pop}(i) |}{| \text{Recommended items} |}$$
- Serendipity: Proportion of items that may be surprising for the user, calculated as the proportion of items recommended by the benchmarked models that are not recommended by a credible baseline. In our case the baseline was MostPop. $$\frac{| \text{Recommended items} \cup \text{Recommended items by most pop} |}{| \text{Recommended items} |}$$
- Linking Interaction Recency(LIR): Measures the average recency of a user’s linking interactions in the selected reasoning paths used to explain recommended products.
  Given a user $u$, a chronologically sorted list of $u$’s interactions $T_u$, and a set of reasoning paths $\( \tilde{L}^k_u \)$ selected for explaining the recommended products $\( \tilde{P}_u \)$ with         
  $\( |\tilde{L}^k_u| = |\tilde{P}_u| \)$, the LIR over the selected reasoning paths is defined as:

  $$\text{LIR}(\tilde{L}^k_u) = \frac{1}{|\tilde{L}^k_u|} \sum_{l=\{e_0 \xleftarrow{r_1} e_1 \xleftarrow{r_2} \ldots\} \in \tilde{L}^k_u} \text{IR}(e_1, t) \quad \text{with } (e_1, t) \in T_u$$

  where $\( \text{IR}(e_1, t) \)$ represents the interaction recency for entity $e_1$ at time $t$ in $T_u$. LIR values close to 0 indicate that the linking interactions are older, while values closer to 1 indicate more recent linking interactions.
- Linking Interaction Diversity(LID): Measures the diversity of entities encountered in the reasoning paths used to explain recommended products for a user.
  Given a user $u$ and a set of reasoning paths $\( \tilde{L}^k_u \)$ selected to explain the recommended products $\( \tilde{P}_u \)$, where $\( |\tilde{L}^k_u| 
= |\tilde{P}_u| \)$, LID is defined as:
 $$\text{LID}(\tilde{L}^k_u) = \frac{|\{e_1 \mid \forall l = \{e_0 \xleftarrow{r_1} e_1 \xleftarrow{r_2} \ldots\} \in \tilde{L}^k_u\}|}{|\tilde{L}^k_u|}$$

  where $e_1$ represents unique entities in the reasoning paths. LID values range from [0, 1], with values close to 0 indicating low diversity in linking 
  interactions, and values closer to 1 indicating high diversity.
- Shared Entity Popularity(SEP): Quantifies the popularity of shared entities involved in the reasoning paths that explain recommended products for a user.
  Given a user $u$, a set of entity popularity lists $\( E_{\lambda} \)$ $(sorted by popularity for each entity type \( \lambda \))$, and a set of reasoning paths 
  $\( \tilde{L}^k_u \)$ selected to explain the recommended products $\( \tilde{P}_u \)$, where $\( |\tilde{L}^k_u| = |\tilde{P}_u| \)$, SEP is defined as:

  $$\text{SEP}(\tilde{L}^k_u) = \frac{1}{|\tilde{L}^k_u|} \sum_{l=\{e_0 \xleftarrow{r_1} \ldots \xleftarrow{r_k} e_k\} \in \tilde{L}^k_u} \text{EP}(e_{k-1}, v) 
  \quad \text{where} \quad (e_{k-1}, v) \in E_{\lambda}$$
  
  Here, $\( e_{k-1} \)$ is the shared entity in the reasoning path, and $v$ represents its popularity score in $\( E_{\lambda} \)$. SEP values range from $0$ to 
  $1$, with values close to 0 indicating low popularity and values near 1 indicating high popularity for shared entities.

- Shared Entity Diversity(SED): Measures the diversity of shared entities across the reasoning paths used to explain recommended products.
  Given a user $u$ and a set of reasoning paths $\( \tilde{L}^k_u \)$ chosen to explain the recommended products $\( \tilde{P}_u \)$, where $\( |\tilde{L}^k_u| = 
  |\tilde{P}_u| \)$, the SED is defined as:

  $$\text{SED}(\tilde{L}^k_u) = \frac{\left| \{ e_{k-1} \mid \forall l = \{ e_0 \xleftarrow{r_1} \ldots \xleftarrow{r_k} e_k \} \in \tilde{L}^k_u \} \right|} 
  {|\tilde{L}^k_u|}$$
  Here, $\( e_{k-1} \)$ represents the shared entity in each reasoning path. SED values range from $0$ to $1$, with values close to 0 indicating low diversity and 
  values close to 1 indicating high diversity of shared entities in the explanation paths.
  
- Explanation Type Diversity(ETD): Quantifies the diversity of explanation types for the recommended products provided to a user.
  Given a user $u$ with a top-k list of recommended products $\( \tilde{P}_u \)$ and the corresponding explanation paths $\( \hat{L}_u \)$                        
  
  let $$\( 𝜔_{\hat{L}_u} = \{ 𝜔_l \mid l \in \hat{L}_u \} \)$$ be the set of unique path types in the explanations for the recommended list. ETD is calculated as:  
![ETD Formula](https://latex.codecogs.com/png.latex?\text{ETD}(\tilde{L}_u)%20=%20\frac{|\omega%20\hat{L}_u|}{\min(k,%20|\omega%20L|)})

  Here, $L$ represents the set of all possible paths between users and products, and $𝜔_L$ is the set of all possible explanation types. ETD values range from $0$ to $1$, where values close to 0 indicate low diversity and values close to 1 indicate high diversity of explanation types for the recommended products.




#### Hyper parameters
The hyper parameters that have been considered in the grid search are listed below, alongside a brief description and the codename used in the experiments:

PGPR
- hidden : number of hidden units of each layer of the shared embedding neural network, that is used as a backbone by the actor and the critic prediction heads
- ent_weight: weight of the entropy loss that quantifies entropy in the action distribution

CAFE
- embed_size: size of the embedding of entities and relations for neural modules employed by CAFE's symbolic model
- rank_weight: weight of the ranking loss component in the total loss.

KGAT
- adj_type: weighting technique applied to each connection on the KG adjacency matrix A
   - bilateral (bi), pre and post multiply A by the inverse of the square root of the diagonal matrix of out degrees of each node
   - single (si), pre multiply A by the inverse of the of the diagonal matrix of out degrees of each node
- embed_size: size of user and entity embeddings
- kge_size: size of the relation embeddings

CKE
- adj_type (weighting technique applied to each connection on the KG adjacency matrix A )
   - bilateral (bi), pre and post multiply A by the inverse of the square root of the diagonal matrix of out degrees of each node
   - single (si), pre multiply A by the inverse of the of the diagonal matrix of out degrees of each node
- embed_size (size of user and entity embeddings)
- kge_size (size of the relation embeddings)

PLM
- num_epochs: Max number of epochs.
- model: The base huggingface model from where eredit the architecture one from {distilgpt2, gpt2, gpt2-large}
- batch_size: Batch size.
- sample_size: Dataset sample size (to dermine which dataset to use)
- n_hop: Dataset hop size (to dermine which dataset to use)
- logit_processor_type: Decoding strategy empty for PLM
- n_seq_infer: Number of sequences generated for each user should be > k

PEARLM
- num_epochs: Max number of epochs.
- model: The base huggingface model from where eredit the architecture one from {distilgpt2, gpt2, gpt2-large}
- batch_size: Batch size.
- sample_size: Dataset sample size (to dermine which dataset to use)
- n_hop: Dataset hop size (to dermine which dataset to use)
- logit_processor_type: Decoding strategy 'gcd' for PEARLM
- n_seq_infer: Number of sequences generated for each user should be > k

Optimal hyper parameters:
Each model is configured with a set of optimal hyper parameters, according to the dataset upon which it is trained. In order to train a given model with customized hyper parameters, it is necessary to set them from command line. Each can be set by adding as new command line arguments the pair (--param_name param_value) while also specifying the model_name and the dataset to use.

#### COCO
PGPR
 - hidden [256,128]
 - ent_weight 0.01

CAFE
 - embed_size 200
 - rank_weight 10.0

KGAT
 - adj_type si
 - embed_size 128
 - kge_size 128

CKE
 - adj_type si
 - embed_size 32
 - kge_size 32

PLM
 - num_epochs: 3
 - model: distilgpt2
 - batch_size: 128
 - sample_size: 250
 - n_hop: 3
 - logit_processor_type: Decoding strategy empty for PLM
 - n_seq_infer: 30

PEARLM
 - num_epochs: 3
 - model: distilgpt2
 - batch_size: 128
 - sample_size: 250
 - n_hop: 3
 - logit_processor_type: Decoding strategy 'gcd'
 - n_seq_infer: 30

#### MOOPer
PGPR
 - hidden [256,128]
 - ent_weight 0.01

CAFE
 - embed_size 200
 - rank_weight 10.0

KGAT
 - adj_type si
 - embed_size 128
 - kge_size 128

CKE
 - adj_type si
 - embed_size 32
 - kge_size 32

PLM
 - num_epochs: 3
 - model: distilgpt2
 - batch_size: 128
 - sample_size: 250
 - n_hop: 3
 - logit_processor_type: Decoding strategy empty for PLM
 - n_seq_infer: 30

PEARLM
 - num_epochs: 3
 - model: distilgpt2
 - batch_size: 128
 - sample_size: 250
 - n_hop: 3
 - logit_processor_type: Decoding strategy 'gcd'
 - n_seq_infer: 30

#### MOOCCube
PGPR
 - hidden [256,128]
 - ent_weight 0.01

CAFE
 - embed_size 200
 - rank_weight 10.0

KGAT
 - adj_type si
 - embed_size 128
 - kge_size 128

CKE
 - adj_type si
 - embed_size 128
 - kge_size 128

PLM
 - num_epochs: 3
 - model: distilgpt2
 - batch_size: 128
 - sample_size: 250
 - n_hop: 3
 - logit_processor_type: Decoding strategy empty for PLM
 - n_seq_infer: 30

PEARLM
 - num_epochs: 3
 - model: distilgpt2
 - batch_size: 128
 - sample_size: 250
 - n_hop: 3
 - logit_processor_type: Decoding strategy 'gcd'
 - n_seq_infer: 30

# Data

> [!TIP]  
> The data is already available in the repository, while the paths can be downloaded from the previous link. The paths_random_walk folder must be placed in the folder of the corresponding dataset (e.g. data/coco/paths_random_walk).

### Datasets info
|                             | COCO        | MOOPer     | MOOCCube   |
|-----------------------------|-------------|------------|------------|
| **Interaction Information** |             |            |            |
| Users(core)                 | 24,036(10)  | 13,885(6)  | 6,486(10)  |
| Courses(core)               | 8,196(10)   | 266(6)     | 549(10)    |
| Interactions                | 378,469     | 145,850    | 96,579     |
| Density                     | 0.002       | 0.040      | 0.030      |
| **Knowledge Information**   |             |            |
| Entities (Types)            | 11,246(7)   | 6,072(7)   |25,110(6)   |
| Relations (Types)           | 104,983 (6) | 21,020(3)  |142,415(3)  |
| Avg. Degree(All)            | 28.07       | 25.05      |27.43       |
| Avg. Degree(courses)        | 64.86       | 17.53      |52.31       |

# Results
### Recommendation utility and beyond utility metrics
**COCO**
|    Familly      |   Method       | NDCG | MRR  |  PRECISION | RECALL  |SERENDIPITY | DIVERSITY | NOVELTY | COVERAGE |
|----------|----------|------|------|------------|---------|------------|-----------|---------|----------|
| TCF   |NFM | 0.08  |  0.06 | 0.02  |0.02 | 0.88 | 0.26 | 0.99 | 0.65   |
| TCF   |BPRFM | 0.10  |  0.08 | 0.02  |0.02 | 0.85 | 0.24 | 0.99 | 0.41   |
| NE-KG   |KGAT | 0.08  |  0.06 | 0.01  |0.01 | 0.84 | 0.24 | 0.98 | 0.64   |
| NE-KG   |CKE | 0.08  |  0.06 | 0.01  |0.01 | 0.82 | 0.28 | 0.98 | 0.38   |
| E-KG   |PGPR | 0.05  |  0.03 | 0.01  |0.01 | 0.73 | 0.30 | 0.99 | 0.24   |
| E-KG   |CAFE | 0.04  |  0.03 | 0.01  |0.02 | 0.29 | 0.23 | 0.99 | 0.02   |
| E-KG   |PLM | 0.06  |  0.04 | 0.01  |0.01 | 0.54 | 0.13 | 0.62 | 0.34   |
| E-KG   |PEARLM | 0.41  |  0.38 | 0.10  |0.10 | 0.58 | 0.17 | 0.62 | 0.80   |

**MOOPer**
|    Familly      |   Method       | NDCG | MRR  |  PRECISION | RECALL  |SERENDIPITY | DIVERSITY | NOVELTY | COVERAGE |
|----------|----------|------|------|------------|---------|------------|-----------|---------|----------|
| TCF   |NFM | 0.36  |  0.30 | 0.08  |0.08 | 0.73 | 0.84 | 0.85 | 0.60   |
| TCF   |BPRFM | 0.42  |  0.35 | 0.09  |0.09 | 0.87 | 0.79 | 0.85 | 0.96   |
| NE-KG   |KGAT | 0.38  |  0.31 | 0.08  |0.08 | 0.75 | 0.80 | 0.84 | 0.97   |
| NE-KG   |CKE | 0.41  |  0.34 | 0.08  |0.08 | 0.84 | 0.79 | 0.81 | 0.83   |
| E-KG   |PGPR | 0.20  |  0.12 | 0.06  |0.06 | 0.56 | 0.85 | 0.84 | 0.80   |
| E-KG   |CAFE | 0.31  |  0.26 | 0.07  |0.07 | 0.66 | 0.80 | 0.84 | 0.60   |
| E-KG   |PLM | 0.53  |  0.46 | 0.12  |0.15 | 0.83 | 0.86 | 0.85 | 0.99   |
| E-KG   |PEARLM | 0.60  |  0.52 | 0.15  |0.16 | 0.84 | 0.84 | 0.83 | 0.99   |

**MOOCCube**
|    Familly      |   Method       | NDCG | MRR  |  PRECISION | RECALL  |SERENDIPITY | DIVERSITY | NOVELTY | COVERAGE |
|----------|----------|------|------|------------|---------|------------|-----------|---------|----------|
| TCF   |NFM | 0.15  |  0.11 | 0.03  |0.03 | 0.55 | 0.97 | 0.83 | 0.21   |
| TCF   |BPRFM | 0.16  |  0.12 | 0.03  |0.03 | 0.54 | 0.98 | 0.80 | 0.63   |
| NE-KG   |KGAT | 0.15  |  0.11 | 0.03  |0.03 | 0.59 | 0.98 | 0.82 | 0.60   |
| NE-KG   |CKE | 0.15  |  0.11 | 0.03  |0.03 | 0.57 | 0.98 | 0.79 | 0.62   |
| E-KG   |PGPR | 0.10  |  0.06 | 0.03  |0.03 | 0.52 | 0.99 | 0.73 | 0.37   |
| E-KG   |CAFE | 0.13  |  0.10 | 0.03  |0.03 | 0.27 | 0.95 | 0.76 | 0.16   |
| E-KG   |PLM | 0.10  |  0.08 | 0.02  |0.02 | 0.54 | 0.74 | 0.63 | 0.58   |
| E-KG   |PEARLM | 0.39  |  0.33 | 0.10  |0.10 | 0.61 | 0.74 | 0.66 | 0.99   |

*Families*: Traditional Collaborative Filtering(TCF); Non-Explainable Knowledge Graph(NE-KG); Explainable Knowledge Graph(E-KG)
