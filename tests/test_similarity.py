import pytest
from ..source import similarity

def test_calculate_partial_score():
    moments1 = [1, 2, 3, 4, 5]
    moments2 = [2, 3, 4, 5, 6]

    partial_score = similarity.calculate_partial_score(moments1, moments2)
    assert partial_score == 1.0  # (1 + 1 + 1 + 1 + 1) / 5

def test_get_similarity_measure():
    partial_score = 0.5
    similarity_measure = similarity.get_similarity_measure(partial_score)
    assert similarity_measure == 2/3  # 1 / (1 + 0.5)
