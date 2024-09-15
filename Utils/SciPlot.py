import numpy as np

def ConfidenceBoundsGen(Data, Confidence_Level = 1.96):

    Data = np.array(Data)

    assert Data.ndim < 3 and Data.ndim > 0, "Data must be vector or 2 way matrix"

    if Data.ndim == 2:

        y_Upper = []
        y_Lower = []

        [num_samples, num_timePoint] = Data.shape

        for t_p in range(num_timePoint):

            U_tmp, L_tmp = ConfidenceBound(Data[:, t_p], Confidence_Level)

            y_Upper.append(L_tmp)
            y_Lower.append(U_tmp)

        return np.array(y_Upper), np.array(y_Lower)
    
    else:

        return 0
    
def ConfidenceBound(Data, Confidence_Level):

    UB_CI = np.mean(Data) + Confidence_Level * np.std(Data) / np.sqrt(len(Data))
    LB_CI = np.mean(Data) - Confidence_Level * np.std(Data) / np.sqrt(len(Data))

    return UB_CI, LB_CI

def AestDim(Q, thr = 3):

    CanDims_t = []

    for i in range(1, int(Q / 2) + 1):

        if np.mod(Q, i) == 0:

            CanDims_t.append([i, int(Q / i)])

    BestDims = CanDims_t[-1]

    if max([BestDims[0] / BestDims[1], BestDims[1] / BestDims[0]]) <= thr:

        return np.sort(CanDims_t[-1])
    
    else:
        
        return AestDim(Q + 1)