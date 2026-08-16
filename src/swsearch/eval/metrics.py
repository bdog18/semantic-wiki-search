import json
from collections import defaultdict
from typing import Callable


def load_test_set(filepath: str) -> list[dict]:
    """Load test queries and ground truth from a JSON file: a list of
    {"query": str, "relevant_articles": [str, ...]} entries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_mrr(test_set: list[dict], retrieval_function: Callable[[str], list[str]]) -> dict:
    """Mean Reciprocal Rank: average of the reciprocal rank of the first relevant result."""
    reciprocal_ranks = []

    for entry in test_set:
        relevant = {a.lower() for a in entry['relevant_articles']}
        retrieved = [a.lower() for a in retrieval_function(entry['query'])]

        rr = 0.0
        for i, item in enumerate(retrieved):
            if item in relevant:
                rr = 1 / (i + 1)
                break
        reciprocal_ranks.append(rr)

    return {"MRR": round(sum(reciprocal_ranks) / len(test_set), 4)}


def compute_mean_average_precision(test_set: list[dict], retrieval_function: Callable[[str], list[str]]) -> dict:
    """Mean Average Precision (MAP): average of the average precision for each query."""
    average_precisions = []

    for entry in test_set:
        relevant = {a.lower() for a in entry['relevant_articles']}
        retrieved = [a.lower() for a in retrieval_function(entry['query'])]
        num_relevant = 0
        precision_sum = 0.0

        for i, item in enumerate(retrieved):
            if item in relevant:
                num_relevant += 1
                precision_sum += num_relevant / (i + 1)

        average_precision = precision_sum / len(relevant) if relevant else 0.0
        average_precisions.append(average_precision)

    return {"MAP": round(sum(average_precisions) / len(test_set), 4)}


def compute_top_k_accuracy(test_set: list[dict], retrieval_function: Callable[[str], list[str]], ks=(1, 3, 5, 10)) -> dict:
    """Top-K Accuracy: whether any relevant article appears in the top-k results."""
    results = {}
    total = len(test_set)

    for k in ks:
        hits = 0
        for entry in test_set:
            relevant = {a.lower() for a in entry['relevant_articles']}
            retrieved = [a.lower() for a in retrieval_function(entry['query'])[:k]]
            if relevant & set(retrieved):
                hits += 1
        results[f"Top-{k} Accuracy"] = round(hits / total, 4)

    return results


def compute_mean_r_precision(test_set: list[dict], retrieval_function: Callable[[str], list[str]]) -> dict:
    """Mean R-Precision: average precision at R, where R is the number of relevant articles for each query."""
    precisions = []

    for entry in test_set:
        relevant = {a.lower() for a in entry['relevant_articles']}
        R = len(relevant)
        retrieved = [a.lower() for a in retrieval_function(entry['query'])[:R]]
        hits = set(retrieved) & relevant
        precision_at_R = len(hits) / R if R > 0 else 0.0
        precisions.append(precision_at_R)

    return {"Mean R-Precision": round(sum(precisions) / len(test_set), 4)}


def compute_precision_at_k(test_set: list[dict], retrieval_function: Callable[[str], list[str]], ks=(1, 3, 5, 10)) -> dict:
    """Precision@K: proportion of retrieved articles in top-k that are relevant."""
    results = defaultdict(float)
    total = len(test_set)

    for entry in test_set:
        relevant = {a.lower() for a in entry['relevant_articles']}
        retrieved = [a.lower() for a in retrieval_function(entry['query'])]
        for k in ks:
            hits = set(retrieved[:k]) & relevant
            results[k] += len(hits) / k

    return {f"Precision@{k}": round(results[k] / total, 4) for k in ks}


def compute_recall_at_k(test_set: list[dict], retrieval_function: Callable[[str], list[str]], ks=(1, 3, 5, 10)) -> dict:
    """Recall@K: proportion of relevant articles found in the top-k results."""
    results = defaultdict(float)
    total = len(test_set)

    for entry in test_set:
        relevant = {a.lower() for a in entry['relevant_articles']}
        retrieved = [a.lower() for a in retrieval_function(entry['query'])]
        for k in ks:
            hits = set(retrieved[:k]) & relevant
            results[k] += len(hits) / len(relevant)

    return {f"Recall@{k}": round(results[k] / total, 4) for k in ks}



def evaluate_all_metrics(test_set: list[dict], retrieval_function: Callable[[str], list[str]], ks=(1, 3, 5, 10)) -> dict:
    """Run all retrieval evaluation metrics and return a combined results dict."""
    metrics = {}
    metrics.update(compute_mrr(test_set, retrieval_function))
    metrics.update(compute_mean_r_precision(test_set, retrieval_function))
    metrics.update(compute_mean_average_precision(test_set, retrieval_function))
    metrics.update(compute_top_k_accuracy(test_set, retrieval_function, ks))
    # metrics.update(compute_precision_at_k(test_set, retrieval_function, ks))
    # metrics.update(compute_recall_at_k(test_set, retrieval_function, ks))
    return metrics
