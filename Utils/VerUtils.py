import numpy as np


def roll_mat_gen(x, k):

    assert x.ndim == 1, "x must be a vector"

    X_rm = []

    N = len(x)

    for i in range(N - k):

        X_rm.append(x[i : i + k])

    return np.array(X_rm)

def RandomBlockSampling(Length: int, NumBlock: int, NumSample_inBlock: int):

    Block_Len = int(Length / NumBlock)

    # assert NumSample_inBlock <= Block_Len, 'Number of Samples in Blocks must be less than length of Block'

    if  NumSample_inBlock >= Block_Len:

        NumSample_inBlock = Block_Len

    SampleMat = []

    for Block in range(NumBlock - 1):

        SampleMat.append(np.random.permutation(Block_Len)[:NumSample_inBlock] + Block * Block_Len)

    SampleMat.append(Length - 1 - np.random.permutation(Block_Len)[:NumSample_inBlock])

    return np.array(SampleMat)

def DeterminedBlockSampling(Length: int, NumBlock: int, NumSample_inBlock: int):

    Block_Len = int(Length / NumBlock)

    # assert NumSample_inBlock <= Block_Len, 'Number of Samples in Blocks must be less than length of Block'

    if  NumSample_inBlock >= Block_Len:

        NumSample_inBlock = Block_Len

    SampleMat = []

    for Block in range(NumBlock - 1):

        SampleMat.append(np.int32(np.arange(NumSample_inBlock)) + Block * Block_Len)

    SampleMat.append(Length - 1 - np.int32(np.arange(NumSample_inBlock)))

    return np.array(SampleMat)