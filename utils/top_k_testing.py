import json
from collections import defaultdict

def load_test_set(filepath):
    """
    Load test queries and ground truth from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing test queries and relevant articles.

    Returns:
        list: List of dictionaries with keys 'query' and 'relevant_articles'.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_top_k_accuracy(test_set, retrieval_function, ks=[1, 3, 5, 10]):
    """
    Compute Top-K Accuracy: Measures if any relevant article appears in the top-k results.

    Args:
        test_set (list): List of queries and associated relevant articles.
        retrieval_function (function): Function that takes a query and returns ranked article titles.
        ks (list): Values of K to evaluate (e.g., Top-1, Top-3, Top-5, etc.).

    Returns:
        dict: Dictionary with Top-K accuracy values for each K.
    """
    results = {}
    total = len(test_set)

    for k in ks:
        hits = 0
        for entry in test_set:
            relevant = set(a.lower() for a in entry['relevant_articles'])
            retrieved = [a.lower() for a in retrieval_function(entry['query'])[:k]]

            # Count hit if any relevant article is found in top-k
            if relevant & set(retrieved):
                hits += 1

        results[f"Top-{k} Accuracy"] = round(hits / total, 4)

    return results


def compute_precision_at_k(test_set, retrieval_function, ks=[1, 3, 5, 10]):
    """
    Compute Precision@K: Measures the proportion of retrieved articles in top-k that are relevant.

    Args:
        test_set (list): Test data with queries and relevant articles.
        retrieval_function (function): Retrieval function for ranking articles.
        ks (list): List of K values to compute precision for.

    Returns:
        dict: Dictionary with Precision@K scores.
    """
    results = defaultdict(float)
    total = len(test_set)

    for entry in test_set:
        relevant = set(a.lower() for a in entry['relevant_articles'])
        retrieved = [a.lower() for a in retrieval_function(entry['query'])]

        for k in ks:
            top_k = set(retrieved[:k])
            hits = top_k & relevant
            results[k] += len(hits) / k

    return {f"Precision@{k}": round(results[k] / total, 4) for k in ks}


def compute_recall_at_k(test_set, retrieval_function, ks=[1, 3, 5, 10]):
    """
    Compute Recall@K: Measures the proportion of relevant articles found in the top-k results.

    Args:
        test_set (list): Test data with queries and relevant articles.
        retrieval_function (function): Retrieval function for ranking articles.
        ks (list): List of K values to compute recall for.

    Returns:
        dict: Dictionary with Recall@K scores.
    """
    results = defaultdict(float)
    total = len(test_set)

    for entry in test_set:
        relevant = set(a.lower() for a in entry['relevant_articles'])
        retrieved = [a.lower() for a in retrieval_function(entry['query'])]

        for k in ks:
            top_k = set(retrieved[:k])
            hits = top_k & relevant
            results[k] += len(hits) / len(relevant)

    return {f"Recall@{k}": round(results[k] / total, 4) for k in ks}


def compute_mrr(test_set, retrieval_function):
    """
    Compute Mean Reciprocal Rank (MRR): Average of the reciprocal rank of the first relevant result.

    Args:
        test_set (list): Test data with queries and relevant articles.
        retrieval_function (function): Function returning a ranked list of articles.

    Returns:
        dict: Dictionary with the MRR score.
    """
    reciprocal_ranks = []

    for entry in test_set:
        relevant = set(a.lower() for a in entry['relevant_articles'])
        retrieved = [a.lower() for a in retrieval_function(entry['query'])]

        rr = 0.0
        for i, item in enumerate(retrieved):
            if item in relevant:
                rr = 1 / (i + 1)
                break  # only the first hit matters
        reciprocal_ranks.append(rr)

    return {"MRR": round(sum(reciprocal_ranks) / len(test_set), 4)}


def evaluate_all_metrics(test_set, retrieval_function, ks=[1, 3, 5, 10]):
    """
    Run all retrieval evaluation metrics and return a combined results dictionary.

    Args:
        test_set (list): List of queries with relevant articles.
        retrieval_function (function): Retrieval model to be evaluated.
        ks (list): List of cutoff values for Top-K, Precision@K, and Recall@K.

    Returns:
        dict: Combined metrics dictionary.
    """
    metrics = {}
    metrics.update(compute_top_k_accuracy(test_set, retrieval_function, ks))
    metrics.update(compute_precision_at_k(test_set, retrieval_function, ks))
    metrics.update(compute_recall_at_k(test_set, retrieval_function, ks))
    metrics.update(compute_mrr(test_set, retrieval_function))
    return metrics


def retrieval_function(query):
    """
    Wraps your search function to return only a list of titles (strings) 
    for compatibility with metric functions.
    """
    # Replace this with your real search logic
    results = semantic_search(query)  # <- returns list of dicts
    return [r["title"] for r in results]


def semantic_search(query):
    return [
        {'title': 'Whitson, Texas',
        'url': 'https://en.wikipedia.org/wiki?curid=74340115'},
        {'title': 'My Kink Is Karma',
        'url': 'https://en.wikipedia.org/wiki?curid=74920372'},
        {'title': 'Marcial Moreno-Mañas',
        'url': 'https://en.wikipedia.org/wiki?curid=74635281'},
        {'title': 'Iran at the 2022 Asian Games',
        'url': 'https://en.wikipedia.org/wiki?curid=74864425'},
        {'title': 'Sir Charles Saxton, 2nd Baronet',
        'url': 'https://en.wikipedia.org/wiki?curid=74896144'},
        {'title': "List of Girls' Crystal comic stories",
        'url': 'https://en.wikipedia.org/wiki?curid=74899201'},
        {'title': 'Not My Neighbour',
        'url': 'https://en.wikipedia.org/wiki?curid=74886103'},
        {'title': 'Alaska Airlines Flight 2059',
        'url': 'https://en.wikipedia.org/wiki?curid=75127810'},
        {'title': 'Ting Ting Chaoro',
        'url': 'https://en.wikipedia.org/wiki?curid=74877608'},
        {'title': 'Detdet Pepito',
        'url': 'https://en.wikipedia.org/wiki?curid=74590708'}
    ]


# Run evaluation with dummy model
if __name__ == "__main__":
    test_set = load_test_set("../data/test_data/test_queries.json")
    results = evaluate_all_metrics(test_set, retrieval_function)

    print("Evaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
