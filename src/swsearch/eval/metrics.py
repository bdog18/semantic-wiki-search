"""Retrieval metrics, computed from one pass over the query set.

Every metric here reads the same thing: the ranked list of article titles a
query returned, against the set the test file marks relevant. So the search
runs *once* per query and each metric scores the same retrieved lists.

That is not a micro-optimisation. Each compute_* function used to call
retrieval_function itself, and compute_top_k_accuracy called it once per k
*per query* -- so `swsearch evaluate` over the 155-query set with
ks=(1,3,5,10) issued 1,085 full-index searches to answer a question that
needs 155. Against a 42M-paragraph index that is the difference between
minutes and most of an hour, and every one of those extra searches returned
the identical list.
"""

import json
from typing import Callable

# Retrieved titles and relevant titles are compared case-insensitively: the
# test file is hand-written, and "World War II" vs "World War ii" is a
# labelling artifact, not a retrieval miss.
Retrieved = list[list[str]]


def load_test_set(filepath: str) -> list[dict]:
    """Load test queries and ground truth from a JSON file: a list of
    {"query": str, "relevant_articles": [str, ...]} entries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def retrieve_all(test_set: list[dict], retrieval_function: Callable[[str], list[str]]) -> Retrieved:
    """Run every query once, returning lowercased ranked titles per query."""
    return [[title.lower() for title in retrieval_function(entry['query'])] for entry in test_set]


def _relevant_sets(test_set: list[dict]) -> list[set[str]]:
    return [{a.lower() for a in entry['relevant_articles']} for entry in test_set]


def compute_mrr(test_set: list[dict], retrieved: Retrieved) -> dict:
    """Mean Reciprocal Rank: average of the reciprocal rank of the first relevant result."""
    reciprocal_ranks = []

    for relevant, titles in zip(_relevant_sets(test_set), retrieved):
        rr = 0.0
        for i, item in enumerate(titles):
            if item in relevant:
                rr = 1 / (i + 1)
                break
        reciprocal_ranks.append(rr)

    return {"MRR": round(sum(reciprocal_ranks) / len(test_set), 4)}


def compute_mean_average_precision(test_set: list[dict], retrieved: Retrieved) -> dict:
    """Mean Average Precision (MAP): average of the average precision for each query."""
    average_precisions = []

    for relevant, titles in zip(_relevant_sets(test_set), retrieved):
        num_relevant = 0
        precision_sum = 0.0

        for i, item in enumerate(titles):
            if item in relevant:
                num_relevant += 1
                precision_sum += num_relevant / (i + 1)

        average_precisions.append(precision_sum / len(relevant) if relevant else 0.0)

    return {"MAP": round(sum(average_precisions) / len(test_set), 4)}


def compute_top_k_accuracy(test_set: list[dict], retrieved: Retrieved, ks=(1, 3, 5, 10)) -> dict:
    """Top-K Accuracy: whether any relevant article appears in the top-k results."""
    relevant_sets = _relevant_sets(test_set)
    total = len(test_set)

    results = {}
    for k in ks:
        hits = sum(
            1 for relevant, titles in zip(relevant_sets, retrieved) if relevant & set(titles[:k])
        )
        results[f"Top-{k} Accuracy"] = round(hits / total, 4)

    return results


def compute_mean_r_precision(test_set: list[dict], retrieved: Retrieved) -> dict:
    """Mean R-Precision: average precision at R, where R is the number of relevant articles for each query."""
    precisions = []

    for relevant, titles in zip(_relevant_sets(test_set), retrieved):
        r = len(relevant)
        hits = set(titles[:r]) & relevant
        precisions.append(len(hits) / r if r > 0 else 0.0)

    return {"Mean R-Precision": round(sum(precisions) / len(test_set), 4)}


def evaluate_all_metrics(test_set: list[dict], retrieval_function: Callable[[str], list[str]], ks=(1, 3, 5, 10)) -> dict:
    """Run every retrieval metric against one pass over the query set.

    retrieval_function should return at least max(ks) titles per query; the
    k-sensitive metrics slice that list rather than re-querying for each k.
    """
    retrieved = retrieve_all(test_set, retrieval_function)

    metrics = {}
    metrics.update(compute_mrr(test_set, retrieved))
    metrics.update(compute_mean_r_precision(test_set, retrieved))
    metrics.update(compute_mean_average_precision(test_set, retrieved))
    metrics.update(compute_top_k_accuracy(test_set, retrieved, ks))
    return metrics
