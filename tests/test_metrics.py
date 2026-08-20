from swsearch.eval.metrics import (
    compute_mean_average_precision,
    compute_mean_r_precision,
    compute_mrr,
    compute_top_k_accuracy,
    evaluate_all_metrics,
    retrieve_all,
)

# Two queries with known ranked results, so every metric below is checkable by
# hand rather than against a golden number nobody can re-derive.
TEST_SET = [
    {"query": "q1", "relevant_articles": ["Alpha", "Beta"]},
    {"query": "q2", "relevant_articles": ["Gamma"]},
]
RESULTS = {
    "q1": ["Zulu", "Alpha", "Beta"],   # first hit at rank 2
    "q2": ["Gamma", "Yankee", "Xray"],  # first hit at rank 1
}


def _retrieval(query):
    return RESULTS[query]


def test_retrieve_all_lowercases_and_preserves_rank_order():
    assert retrieve_all(TEST_SET, _retrieval) == [
        ["zulu", "alpha", "beta"],
        ["gamma", "yankee", "xray"],
    ]


def test_evaluate_all_metrics_searches_each_query_exactly_once():
    """The reason these functions take pre-retrieved lists.

    Each compute_* used to call retrieval_function itself, and
    compute_top_k_accuracy called it once per k *per query* -- 7 searches per
    query at ks=(1,3,5,10) where 1 does. Against the full 42M-paragraph index
    that turned a 155-query evaluation into 1,085 searches.
    """
    calls = []

    def counting_retrieval(query):
        calls.append(query)
        return RESULTS[query]

    evaluate_all_metrics(TEST_SET, counting_retrieval, ks=(1, 3, 5, 10))

    assert calls == ["q1", "q2"]


def test_mrr_averages_reciprocal_of_first_hit():
    retrieved = retrieve_all(TEST_SET, _retrieval)
    # q1's first relevant result is at rank 2 (1/2), q2's at rank 1 (1/1).
    assert compute_mrr(TEST_SET, retrieved) == {"MRR": round((0.5 + 1.0) / 2, 4)}


def test_top_k_accuracy_slices_rather_than_requerying():
    retrieved = retrieve_all(TEST_SET, _retrieval)
    result = compute_top_k_accuracy(TEST_SET, retrieved, ks=(1, 3))
    # At k=1 only q2 hits; at k=3 both do.
    assert result == {"Top-1 Accuracy": 0.5, "Top-3 Accuracy": 1.0}


def test_map_rewards_surfacing_several_relevant_articles():
    retrieved = retrieve_all(TEST_SET, _retrieval)
    # q1: hits at ranks 2 and 3 -> (1/2 + 2/3) / 2 relevant = 0.5833
    # q2: hit at rank 1 -> (1/1) / 1 = 1.0
    expected = round(((1 / 2 + 2 / 3) / 2 + 1.0) / 2, 4)
    assert compute_mean_average_precision(TEST_SET, retrieved) == {"MAP": expected}


def test_r_precision_cuts_at_the_relevant_count():
    retrieved = retrieve_all(TEST_SET, _retrieval)
    # q1 has R=2, and its top 2 are ["zulu", "alpha"] -> 1/2.
    # q2 has R=1, and its top 1 is ["gamma"] -> 1/1.
    assert compute_mean_r_precision(TEST_SET, retrieved) == {"Mean R-Precision": 0.75}


def test_metrics_are_case_insensitive():
    test_set = [{"query": "q", "relevant_articles": ["World War II"]}]
    retrieved = retrieve_all(test_set, lambda q: ["world war ii"])
    assert compute_mrr(test_set, retrieved) == {"MRR": 1.0}


def test_query_with_no_hits_scores_zero_without_raising():
    test_set = [{"query": "q", "relevant_articles": ["Nothing"]}]
    retrieved = retrieve_all(test_set, lambda q: ["a", "b"])
    assert compute_mrr(test_set, retrieved) == {"MRR": 0.0}
    assert compute_mean_average_precision(test_set, retrieved) == {"MAP": 0.0}
    assert compute_mean_r_precision(test_set, retrieved) == {"Mean R-Precision": 0.0}


def test_evaluate_all_metrics_reports_every_metric():
    results = evaluate_all_metrics(TEST_SET, _retrieval, ks=(1, 5))
    assert set(results) == {"MRR", "MAP", "Mean R-Precision", "Top-1 Accuracy", "Top-5 Accuracy"}
