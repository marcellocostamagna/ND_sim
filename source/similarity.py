# Script to compare two fingerprints and provide a single-value similarity measure

def calculate_partial_score(moments1: list, moments2:list):
    partial_score = 0
    for i in range(len(moments1)):
        partial_score += abs(moments1[i] - moments2[i])
    return partial_score / len(moments1)

def get_similarity_measure(partial_score):
    return 1/(1 + partial_score)

def get_similarity_score(fingerprint_1, fingerprint_2):
    partial_score = calculate_partial_score(fingerprint_1, fingerprint_2)
    similarity = get_similarity_measure(partial_score)
    return similarity

  