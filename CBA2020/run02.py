import time
import sys
import numpy as np
import random
import matplotlib.pyplot as plt

from scipy import sparse
from numpy import linalg as LA
from sklearn.metrics.pairwise import cosine_similarity
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory


def main():

    tm = TemporalMemory(
        # Must be the same dimensions as the SP
        columnDimensions=(2048,),
        #columnDimensions=(32768,),
        # How many cells in each mini-column.
        cellsPerColumn=16,
        # A segment is active if it has >= activationThreshold connected synapses
        # that are active due to infActiveState
        activationThreshold=4,#1,4(melhor),
        initialPermanence=0.55,
        connectedPermanence=0.5,
        # Minimum number of active synapses for a segment to be considered during
        # search for the best-matching segments.
        minThreshold=1, #1
        # The max number of synapses added to a segment during learning
        maxNewSynapseCount=20, #6
        permanenceIncrement=0.01,
        permanenceDecrement=0.01,
        predictedSegmentDecrement=0.0005,#0.0001,#0.0005,
        maxSegmentsPerCell=100, #8 16(colou)
        maxSynapsesPerSegment=100, #8 16(colou)
        seed=42
    )

    numberImages = 288
    DIR = "/media/cappizzino/OS/Documents and Settings/cappi/Documents/MATLAB/MCN_v0_1"

    # Experiments
    # Ground truth
    GT = np.identity(numberImages, dtype = bool)
    for i in range(GT.shape[0]):
        for j in range(GT.shape[0]-1):
            if i==j:
                GT[i,j]=1

    # MCN (MCN descriptors)
    print 'MCN'
    id_max1=[]
    id_max2=[]
    with open('outputSDR1.txt', 'r') as f:
        D1_MCN = [[int(entry) for entry in line.split(',')] for line in f.readlines()]
    for i in range(len(D1_MCN)):
        id_max1.append(max(D1_MCN[i]))

    with open('outputSDR2.txt', 'r') as f:
        D2_MCN = [[int(entry) for entry in line.split(',')] for line in f.readlines()]
    for i in range(len(D2_MCN)):
        id_max2.append(max(D2_MCN[i]))

    id_max = max(max(id_max1),max(id_max2))
    '''
    D1_sparse = sparse.lil_matrix((len(D1_MCN), id_max+1), dtype='int8')
    for i in range(len(D1_MCN)):
        D1_sparse[i,D1_MCN[i]] = 1

    D2_sparse = sparse.lil_matrix((len(D2_MCN), id_max+1), dtype='int8')
    for i in range(len(D2_MCN)):
        D2_sparse[i,D2_MCN[i]] = 1
    '''
    D1_sparse = sparse.lil_matrix((numberImages, id_max+1), dtype='int8')
    for i in range(numberImages):
        D1_sparse[i,D1_MCN[i]] = 1

    D2_sparse = sparse.lil_matrix((numberImages, id_max+1), dtype='int8')
    for i in range(numberImages):
        D2_sparse[i,D2_MCN[i]] = 1

    S_MCN = pairwiseDescriptors(D1_sparse, D2_sparse)

    # Pairwise (raw descriptors)
    print 'Pairwise descriptors'
    D1 = np.loadtxt(DIR + "/D1.txt", dtype='f', delimiter=',')
    D2 = np.loadtxt(DIR + "/D2.txt", dtype='f', delimiter=',')
    S_pairwise = cosine_similarity(D1[:numberImages], D2[:numberImages])
    
    # Dimension Reduction and binarizarion
    print 'Dimension Reduction'
    P = np.random.randn(D1.shape[1], 1024)
    P = normc(P)
    #D1h = np.dot(D1[:numberImages],P)
    #D2h = np.dot(D2[:numberImages],P)
    #S_Dh = cosine_similarity(D1h, D2h)

     # sLSBH (binarized descriptors)
    print 'sLSBH'
    D1_slsbh = getLSBH(D1[:numberImages],P,0.25) #0.025 0.25
    D2_slsbh = getLSBH(D2[:numberImages],P,0.25)
    #D1_slsbh = np.loadtxt(DIR + "/D1_slsbh.txt", dtype='i', delimiter=',')
    #D2_slsbh = np.loadtxt(DIR + "/D2_slsbh.txt", dtype='i', delimiter=',')
    Sb_pairwise = pairwiseDescriptors(D1_slsbh[:numberImages], D2_slsbh[:numberImages])
    
    '''
    # Binarizarion ans Sparsification
    print 'Binarizarion and Sparsification'
    D1_slsbh = np.zeros((D1h.shape[0],2*D1h.shape[1]), dtype = bool)
    D2_slsbh = np.zeros((D2h.shape[0],2*D2h.shape[1]), dtype = bool)
    for i in range(numberImages):
        D1_slsbh[i,:] = generate_LSBH((D1h[i,:]),(D1h.shape[1]),0.25)
        D2_slsbh[i,:] = generate_LSBH((D2h[i,:]),(D2h.shape[1]),0.25)
    Sb_pairwise = pairwiseDescriptors(D1_slsbh, D2_slsbh)
    '''
    '''
    print 'Spatial Pooler descriptors'
    # Create an array to represent active columns, all initially zero. This
    # will be populated by the compute method below. It must have the same
    # dimensions as the Spatial Pooler.
    activeColumns = np.zeros(2048)
    D1_htm=[]
    D2_htm=[]
    id_max=[]
    id_max1=[]
    id_max2=[]

    for i in range (numberImages):
        # Execute Spatial Pooling algorithm over input space.
        # Feed the examples to the SP
        sp.compute(D1_slsbh[i,:], False, activeColumns)
        activeColumnIndices = np.nonzero(activeColumns)[0]
        D1_htm.append(activeColumnIndices)
        id_max1.append(max(activeColumnIndices))

    for i in range (numberImages):
        # Execute Spatial Pooling algorithm over input space.
        # Feed the examples to the SP
        sp.compute(D2_slsbh[i,:], False, activeColumns)
        activeColumnIndices = np.nonzero(activeColumns)[0]
        D2_htm.append(activeColumnIndices)
        id_max2.append(max(activeColumnIndices))
    
    id_max = max(max(id_max1),max(id_max2))
 
    D1_sparse = sparse.lil_matrix((numberImages, id_max+1), dtype='int8')
    for i in range(numberImages):
        D1_sparse[i,D1_htm[i]] = 1

    D2_sparse = sparse.lil_matrix((numberImages, id_max+1), dtype='int8')
    for i in range(numberImages):
        D2_sparse[i,D2_htm[i]] = 1

    S_SP = pairwiseDescriptors(D1_sparse, D2_sparse)
    '''

    print 'Temporal Pooler (1) descriptors'
    D1_tm=[]
    D2_tm=[]
    id_max=[]
    id_max1=[]
    id_max2=[]
    '''
    for _ in range(5):
        for i in range(numberImages):
            activeColumnIndices = np.nonzero(D1_slsbh[i,:])[0]
            tm.compute(activeColumnIndices, learn=True)
            #tm.compute(D1_htm[i], learn=True)
        for i in range(numberImages):
            activeColumnIndices = np.nonzero(D2_slsbh[i,:])[0]
            tm.compute(activeColumnIndices, learn=True)
            #tm.compute(D2_htm[i], learn=True)
        tm.reset()
    '''
    for i in range(numberImages):
        for _ in range(1):
            activeColumnIndices = np.nonzero(D1_slsbh[i,:])[0]
            #print activeColumnIndices
            time.sleep(2)
            tm.compute(activeColumnIndices, learn=True)
            #tm.compute(D1_htm[i], learn=True)
            activeCells = tm.getWinnerCells()
            #print activeCells
            #time.sleep(5)
            D1_tm.append(activeCells)
            id_max1.append(max(activeCells))
            #tm.reset()

    print 'Temporal Pooler (2) descriptors'
    '''
    for _ in range(2):
        for i in range(numberImages):
            activeColumnIndices = np.nonzero(D2_slsbh[i,:])[0]
            tm.compute(activeColumnIndices, learn=True)
            #tm.compute(D2_htm[i], learn=True)
    '''

    for i in range(numberImages):
        activeColumnIndices = np.nonzero(D2_slsbh[i,:])[0]
        tm.compute(activeColumnIndices, learn=False)
        #tm.compute(D2_htm[i], learn=True)
        activeCells = tm.getWinnerCells()
        D2_tm.append(activeCells)
        id_max2.append(max(activeCells))

    id_max = max(max(id_max1),max(id_max2))
 
    D1_sparse = sparse.lil_matrix((numberImages, id_max+1), dtype='int8')
    for i in range(numberImages):
        D1_sparse[i,D1_tm[i]] = 1

    D2_sparse = sparse.lil_matrix((numberImages, id_max+1), dtype='int8')
    for i in range(numberImages):
        D2_sparse[i,D2_tm[i]] = 1

    S_TM = pairwiseDescriptors(D1_sparse, D2_sparse)

    # Results
    print 'Results'
    fig, ax = plt.subplots()

    P, R = createPR(S_pairwise,GT)
    ax.plot(R, P, label='pairwise / raw (avgP=%f)' %np.trapz(P,R))

    P, R = createPR(S_MCN,GT)
    ax.plot(R, P, label='MCN (avgP=%f)' %np.trapz(P,R))

    #P, R = createPR(S_Dh,GT)
    #ax.plot(R, P, label='pairwise RP / raw (avgP=%f)' %np.trapz(P,R))

    P, R = createPR(Sb_pairwise,GT)
    ax.plot(R, P, label='sLSBH / raw (avgP=%f)' %np.trapz(P,R))

    #P, R = createPR(S_SP,GT)
    #ax.plot(R, P, label='HTM SP (avgP=%f)' %np.trapz(P,R))

    P, R = createPR(S_TM,GT)
    ax.plot(R, P, label='HTM TM (avgP=%f)' %np.trapz(P,R))

    ax.legend()
    ax.grid(True)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

# Normalize columns of matrix
def normc(A):
    B = np.zeros((A.shape[0],A.shape[1]))
    for i in range (A.shape[1]):
        B[:,i] = A[:,i]/LA.norm(A[:,i])
    return B

# Normalize rows of matrix
def normr(A):
    B = np.zeros((A.shape[0],A.shape[1]))
    for i in range (A.shape[0]):
        B[i,:] = A[i,:]/LA.norm(A[i,:])
    return B

# Pairwise (raw descriptors)
def pairwiseDescriptors(D1,D2):
    # Pairwise comparison
    if sparse.issparse(D1):
        S = D1.dot(D2.transpose())
        D1 = D1.toarray()
        D2 = D2.toarray()
    else:
        S = D1.dot(np.transpose(D2))
    
    nOnes_D1 = np.sum(D1, axis=1)
    nOnes_D2 = np.sum(D2, axis=1)
    D1t = np.transpose(np.vstack((np.ones(len(nOnes_D1)),nOnes_D1)))
    D2t = np.vstack((nOnes_D2,np.ones(len(nOnes_D2))))
    mean_nOnes = D1t.dot(D2t)/2
    S = S / mean_nOnes
    return S

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

def generate_LSBH(y,m,s):
    n = int(round(m*s))
    # random projection
    #inp_vector = inp_vector.reshape(1,-1)
    #y = np.dot(inp_vector, self.projections.T)
    #y = y[0]
    # vector projected
    #proj = y
    # dense outputSDR
    #dense = (y > 0).astype('int')
    #y = 
    # sparsification
    largest = y[np.argsort(y)[-n:]]
    z1 = np.zeros(m).astype('int')
    for i in range(m):
        for j in range(largest.shape[0]):
            if y[i] == largest[j]:
                z1[i] = 1
                break
            else:
                z1[i] = 0
    
    result = np.argpartition(y, n-1)
    smallest = y[result[:n]]
    z2 = np.zeros(m).astype('int')
    for i in range(m):
        for j in range(largest.shape[0]):
            if y[i] == smallest[j]:
                z2[i] = 1
                break
            else:
                z2[i] = 0

    z = np.concatenate((z1, z2), axis=0)
    return z

def getLSBH(Y,P,s):
    n = int(round(P.shape[1]*s))
    # random projection
    Y2 = np.dot(Y,P)
    # sort
    IDX = np.sort(Y2, axis=1) 
    iMax = Y2.shape[1]-1
    L1 = np.zeros((Y2.shape[0],Y2.shape[1]), dtype = int, order='F')
    L2 = np.zeros((Y2.shape[0],Y2.shape[1]), dtype = int, order='F')

    for i in range(Y2.shape[0]):
        for j in range(n):
            indx=np.where((Y2[i,:]==IDX[i,iMax-j]))
            L1[i,indx] = 1
            indx=np.where((Y2[i,:]==IDX[i,j]))
            L2[i,indx] = 1
    L = np.concatenate((L1, L2), axis=1)
    return L

if __name__ == "__main__":
    main()