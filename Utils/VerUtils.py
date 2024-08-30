import numpy as np


def roll_mat_gen(x, k):

    assert x.ndim == 1, "x must be a vector"

    X_rm = []

    N = len(x)

    for i in range(N - k):

        X_rm.append(x[i : i + k])

    return np.array(X_rm)