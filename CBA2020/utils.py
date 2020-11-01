import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_similarity

def createPR(S, GThard):
    # Ground Truth
    GT = GThard#.astype('bool')
    R = [0]
    P = [1]
    startV = np.amax(S)
    endV = np.amin(S)

    x = np.linspace(startV, endV, num=100)
    for i in x:
        B = (S >= i).astype('bool')
         # True Positives
        TP = float(np.count_nonzero((GT & B)))
        # False Negatives
        FN = float(np.count_nonzero((GT & (~B))))
        # False Positives
        FP = float(np.count_nonzero(((~GT) & B)))
        # Precision
        P.append((TP/(TP + FP)))
        # Recall
        R.append((TP/(TP + FN)))

    return P, R

# Normalize columns of matrix
def normc(A):
    B = np.zeros((A.shape[0],A.shape[1]))
    for i in range (A.shape[1]):
        B[:,i] = A[:,i]/LA.norm(A[:,i])
    return B

def evaluateSimilarity(inputFeatures):
    
    S = np.ones((inputFeatures.shape[0],inputFeatures.shape[0]), dtype=float)
    maxValue = 0
    
    for idx1 in range(inputFeatures.shape[0]):
        v1 = np.nonzero(inputFeatures[idx1,:])
        for idx2 in range(idx1,inputFeatures.shape[0]):
            v2 = np.nonzero(inputFeatures[idx2,:])
            commonEntries = np.intersect1d(v1,v2)
            meanNEntries = (len(v1)+len(v2))/2
            if meanNEntries != 0:
                S[idx1,idx2] = commonEntries.shape[0]/meanNEntries
            else:
                S[idx1,idx2] = 0
    return S