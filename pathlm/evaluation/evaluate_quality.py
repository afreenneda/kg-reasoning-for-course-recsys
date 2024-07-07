import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple

from pathlm.utils import check_dir
from pathlm.evaluation.eval_utils import *
from pathlm.datasets.data_utils import *
from pathlm.evaluation.beyond_accuracy_metrics import *
from pathlm.evaluation.utility_metrics import *

# Define REC_QUALITY_METRICS_TOPK
REC_QUALITY_METRICS_TOPK = ["NDCG", "MRR", "PRECISION", "RECALL", "SERENDIPITY", "DIVERSITY", "NOVELTY"]
NDCG = "NDCG"
MRR = "MRR"
PRECISION = "PRECISION"
RECALL = "RECALL"
SERENDIPITY = "SERENDIPITY"
DIVERSITY = "DIVERSITY"
NOVELTY = "NOVELTY"
COVERAGE = "COVERAGE"
PFAIRNESS = "PFAIRNESS"


def print_rec_quality_metrics(avg_rec_quality_metrics: Dict[str, float], method='inline'):
    if method == 'latex':
        print(' & '.join(list(avg_rec_quality_metrics.keys())))
        print(' & '.join([str(round(value, 2)) for value in avg_rec_quality_metrics.values()]))
    elif method == 'inline':
        print(', '.join([f'{metric}: {round(value, 2)}' for metric, value in avg_rec_quality_metrics.items()]))
    elif method == 'endline':
        for metric, value in avg_rec_quality_metrics.items():
            print(f'{metric}: {round(value, 2)}')


def evaluate_rec_quality(dataset_name: str, topk_items: Dict[int, List[int]], test_labels: Dict[int, List[int]],
                         k: int = 10, method_name=None, metrics: List[str] = REC_QUALITY_METRICS_TOPK) -> Tuple[
    Dict[str, float], Dict[str, List[float]]]:

    rec_quality_metrics = {metric: list() for metric in metrics}
    recommended_items_all_user_set = set()

    n_items_in_catalog = get_item_count(dataset_name)  # Needed for coverage
    pid2popularity = get_item_pop(dataset_name)  # Needed for novelty
    pid2genre = get_item_genre(dataset_name)  # Needed for diversity
    mostpop_topk = compute_mostpop_topk(dataset_name, k)  # Needed for serendipity

    user_count = len(test_labels)  # Total users including those without recommendations
    topk_sizes = []
    ndcg_zero_count = 0
    ndcg_non_zero_count = 0

    with tqdm(desc=f"Evaluating rec quality for {method_name}", total=user_count) as pbar:
        for uid in test_labels.keys():
            if uid in topk_items:
                topk = topk_items[uid]
                hits = [1 if pid in test_labels[uid] else 0 for pid in topk[:k]]

                # If the model has predicted less than k items, pad with zeros
                hits.extend([0] * (k - len(hits)))

                for metric in REC_QUALITY_METRICS_TOPK:
                    if metric == NDCG:
                        metric_value = ndcg_at_k(hits, k)
                        if metric_value == 0:
                            ndcg_zero_count += 1
                        else:
                            ndcg_non_zero_count += 1
                    elif metric == MRR:
                        metric_value = mmr_at_k(hits, k)
                    elif metric == PRECISION:
                        metric_value = precision_at_k(hits, k)
                    elif metric == RECALL:
                        test_set_len = max(max(1, len(topk)), len(test_labels[uid]))
                        metric_value = recall_at_k(hits, k, test_set_len)
                    elif metric == SERENDIPITY:
                        metric_value = serendipity_at_k(topk, mostpop_topk.get(uid, []), k)
                    elif metric == DIVERSITY:
                        metric_value = diversity_at_k(topk, pid2genre)
                    elif metric == NOVELTY:
                        metric_value = novelty_at_k(topk, pid2popularity)
                    elif metric == PFAIRNESS:
                        continue  # Skip for now
                    rec_quality_metrics[metric].append(metric_value)

                # For coverage
                recommended_items_all_user_set.update(set(topk))
            else:
                # No recommendations for this user, assign 0 for each metric
                for metric in REC_QUALITY_METRICS_TOPK:
                    rec_quality_metrics[metric].append(0.0)
                ndcg_zero_count += 1

            topk_sizes.append(len(topk))
            pbar.update(1)

    # Compute average values for evaluation
    avg_rec_quality_metrics = {metric: np.mean(values) for metric, values in rec_quality_metrics.items()}
    avg_rec_quality_metrics[COVERAGE] = coverage(recommended_items_all_user_set, n_items_in_catalog)

    # Print results
    total_users_with_recommendations = len(test_labels.keys())
    print(f'Number of users: {total_users_with_recommendations}, average topk size: {np.mean(topk_sizes):.2f}')
    #print(f'Number of users with NDCG zero: {ndcg_zero_count}, Number of users with NDCG other than zero: {ndcg_non_zero_count}')
    print(f'Total users used in calculating average: {total_users_with_recommendations}')

    print_rec_quality_metrics(avg_rec_quality_metrics)
    return rec_quality_metrics, avg_rec_quality_metrics


def evaluate_rec_quality_from_results(dataset_name: str, model_name: str, test_labels: Dict[int, List[int]],
                                      k: int = 10, metrics: List[str] = REC_QUALITY_METRICS_TOPK) -> Tuple[
    Dict[str, float], Dict[str, List[float]]]:
    topks = get_precomputed_topks(dataset_name, model_name)
    # TOPK size is fixed to 10
    return evaluate_rec_quality(dataset_name, topks, test_labels, k, metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pgpr', help='which model to evaluate')
    parser.add_argument('--dataset', type=str, default='coco', help='which dataset to evaluate')
    parser.add_argument('--sample_size', type=str, default='250', help='')
    parser.add_argument('--n_hop', type=str, default='3', help='')
    parser.add_argument('--k', type=int, default=10, help='')
    parser.add_argument('--decoding_strategy', type=str, default='gcd', help='')
    args = parser.parse_args()

    if 'plm-rec' in args.model:
        args.model = f'end-to-end@{args.dataset}@{args.model}@distilgpt2@{args.sample_size}@{args.n_hop}@'
    if args.model in ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        args.model = f'end-to-end@{args.dataset}@{args.model}@{args.sample_size}@{args.n_hop}@{args.decoding_strategy}'

    result_dir = get_result_dir(args.dataset, args.model)
    test_set = get_set(args.dataset, set_str='test')

    with open(os.path.join(result_dir, f'top{args.k}_items.pkl'), 'rb') as f:
        topk_items = pickle.load(f)
    print(f"No. of users: {len(topk_items.keys())}, no. of recommendations: {sum([len(v) for v in topk_items.values()])}")


    evaluate_rec_quality_from_results(args.dataset, args.model, test_set)

