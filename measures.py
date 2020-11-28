import numpy as np

def WER(s, ref):
    return ER(s.split(" "), ref.split(" "))

def CER(s, ref):
    return ER(list(s), list(ref))

def ER(s, ref):
    """
        FROM wikipedia levenshtein distance
        s: list of words/char in sentence to measure
        ref: list of words/char in reference
    """

    costs = np.zeros((len(s) + 1, len(ref) + 1))
    for i in range(len(s) + 1):
        costs[i, 0] = i
    for j in range(len(ref) + 1):
        costs[0, j] = j

    for j in range(1, len(ref) + 1):
        for i in range(1, len(s) + 1):
            cost = None
            if s[i-1] == ref[j-1]:
                cost = 0
            else:
                cost = 1
            costs[i,j] = min(
                costs[i-1, j] + 1,
                costs[i, j-1] + 1,
                costs[i-1, j-1] + cost
            )

    return costs[-1,-1] / len(ref)
