import pytest
from ..source import similarity

def test_calculate_partial_score():
    # Define two simple moment lists
    moments1 = [1, 2, 3, 4, 5]
    moments2 = [2, 3, 4, 5, 6]

    # Call your function
    partial_score = similarity.calculate_partial_score(moments1, moments2)

    # Check if the partial_score is correct
    assert partial_score == 1.0  # Average of differences between each pair

def test_get_similarity_measure():
    # Define a simple partial score
    partial_score = 0.5

    # Call your function
    similarity_measure = similarity.get_similarity_measure(partial_score)

    # Check if the similarity_measure is correct
    assert similarity_measure == 2/3  # 1 / (1 + 0.5)
