import numpy as np

def Likelihood_C(Choices, Rewards, ag_est, al_est, T_est):

    Ps = []
    
    Q = np.array([0.0, 0.0])

    for i in range(len(Choices)):

        Choice = int(Choices[i])
        Reward = Rewards[i]

        P = Choice_Probability(Q, Choice, T_est)

        Q[Choice] = Q[Choice] + ag_est * np.max([Reward - Q[Choice], 0]) + al_est * np.min([Reward - Q[Choice], 0])

        Ps.append(P)

    return np.sum(np.log(Ps))

def Choice_Probability(Q, Choice, T):

    return np.exp(Q[Choice] / T) / (np.exp(Q[Choice] / T) + np.exp(Q[1 - Choice] / T))

def Entropy(ps: np.ndarray, zero_thr = 0.0000001, base = 2):

    ps = ps / np.sum(ps)

    ent = 0

    for p in ps:

        if p < zero_thr or p > 1 - zero_thr:

            ent = ent + 0

        else:

            ent = ent + p * np.log(p) / np.log(base)

    return np.abs(ent)

def CodCon(X, base = 2):

    CDD = []

    for x in X:

        CDD.append(np.sum([x_ * base ** (len(x) - i - 1) for i, x_ in enumerate(x)]))

    return np.array(CDD)