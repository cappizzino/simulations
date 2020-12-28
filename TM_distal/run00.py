import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from nupic.encoders.scalar import ScalarEncoder
from nupic.algorithms.temporal_memory import TemporalMemory
from htmresearch.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakPairMemory as TM

from utils import *
from miniColumn import MCN

class Params:
    pass

def main():

    DIR = "./sim_data"

    # Odom Encoder 
    xSDR = ScalarEncoder(w=21,minval=0,maxval=20,n=256)
    ySDR = ScalarEncoder(w=21,minval=0,maxval=20,n=256)
    xyWidth = xSDR.getWidth() + ySDR.getWidth()

    # Visual input
    D = np.loadtxt(DIR + '/seq_multi_loop_noise05_al5.txt', dtype='i', delimiter=',')
    numberImages = D[:,0].size
    nColumns = D[0,:].size
    #time.sleep(10)
    
    # Odom input
    odom = np.loadtxt(DIR + '/seq_multi_loop_noise05_al5_gt.txt', dtype='f', delimiter=',')
    x = odom[:,0]
    y = odom[:,1]

    # Encoder Odom input
    odomSDR = np.zeros((numberImages,xyWidth), dtype=int)
    for i in range(1):
        _xSDR = np.zeros(xSDR.getWidth(), dtype=int)
        xSDR.encodeIntoArray(x[i], _xSDR)
        _ySDR = np.zeros(ySDR.getWidth(), dtype=int)
        ySDR.encodeIntoArray(y[i], _ySDR)
        odomSDR[i,:] = np.concatenate([_xSDR, _ySDR])
    
    tm0 = TM(
        columnCount=nColumns,
        cellsPerColumn=4,
        initialPermanence=0.21,
        connectedPermanence=0.5,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
        minThreshold=15,
        basalInputSize= 512,
        reducedBasalThreshold=1000,
        activationThreshold=1000,
        apicalInputSize=0,
        maxSynapsesPerSegment=-1,
        sampleSize=1,
        seed = 42
        )
        
    tm = TemporalMemory(
        # Must be the same dimensions as the SP
        columnDimensions=(2048,),
        # How many cells in each mini-column.
        cellsPerColumn=4,
        # A segment is active if it has >= activationThreshold connected synapses
        # that are active due to infActiveState
        activationThreshold=13,
        initialPermanence=0.21,
        connectedPermanence=0.5,
        # Minimum number of active synapses for a segment to be considered during
        # search for the best-matching segments.
        minThreshold=1,
        # The max number of synapses added to a segment during learning
        maxNewSynapseCount=3,
        #permanenceIncrement=0.01,
        #permanenceDecrement=0.01,
        predictedSegmentDecrement=0.0005,
        maxSegmentsPerCell=3,
        maxSynapsesPerSegment=3,
        seed=42
    )

    #time.sleep(10)

    # Simple HTM parameters
    params = Params()
    params.maxPredDepth = 0
    params.probAdditionalCon = 0.05 # probability for random connection
    params.nCellPerCol = 32 # number of cells per minicolumn
    params.nInConPerCol = int(round(np.count_nonzero(D) / D.shape[0]))
    #print params.nInConPerCol
    params.minColumnActivity = int(round(0.25*params.nInConPerCol))
    params.nColsPerPattern = 10     # minimum number of active minicolumns k_min
    params.kActiveColumn = 100      # maximum number of active minicolumns k_max
    params.kMin = 1

    # run HTM
    t = time.time()
    print ('Simple HTM')
    htm = MCN('htm',params)

    outputSDR = []
    max_index = []

    for i in range (min(numberImages,D.shape[0])):
        loop = 0 
        #print('\n-------- ITERATION %d ---------' %i)
        # skip empty vectors
        if np.count_nonzero(D[i,:]) == 0:
            print('empty vector, skip\n')
            continue
        loop += 1
        #print D[i,:]
        htm.compute(D[i,:])

        max_index.append(max(htm.winnerCells))
        outputSDR.append(htm.winnerCells)
        
    elapsed = time.time() - t
    print("Elapsed time: %f seconds\n" %elapsed)

    # create output SDR matrix from HTM winner cell output
    M = np.zeros((len(outputSDR),max(max_index)+1), dtype=int)
    for i in range(len(outputSDR)):
        for j in range(len(outputSDR[i])):
            winner = outputSDR[i][j]
            M[i][winner] = 1

    # Temporal Pooler descriptors
    print 'Temporal Pooler descriptors'
    D1_tm=[]
    id_max1=[]
    t = time.time()

    for i in range(min(numberImages,D.shape[0])):
        D1_sp = np.nonzero(D[i,:])[0]
        tm.compute(D1_sp, learn=True)
        activeCells = tm.getWinnerCells()
        D1_tm.append(activeCells)
        id_max1.append(max(activeCells))
    
    elapsed = time.time() - t
    print( "Elapsed time: %f seconds\n" %elapsed)

    # create output SDR matrix from HTM winner cell output
    T = np.zeros((len(D1_tm),max(id_max1)+1), dtype=int)
    for i in range(len(D1_tm)):
        for j in range(len(D1_tm[i])):
            winner = D1_tm[i][j]
            T[i][winner] = 1


    # Temporal Pooler - Distal connections
    print 'Temporal Pooler - Distal connections'
    D2_tm=[]
    id_max2=[]
    t = time.time()

    for i in range(min(numberImages,D.shape[0])):
        D2_sp = np.nonzero(D[i,:])[0]
        basalInputs = np.nonzero(odomSDR[i,:])[0]
        tm0.compute(sorted(D2_sp), sorted(basalInputs), apicalInput=(), basalGrowthCandidates=None, apicalGrowthCandidates=None, learn=True)
        activeCells2 = tm0.getWinnerCells()
        D2_tm.append(activeCells2)
        id_max2.append(max(activeCells2))
    
    elapsed = time.time() - t
    print( "Elapsed time: %f seconds\n" %elapsed)

    # create output SDR matrix from HTM winner cell output
    T2 = np.zeros((len(D2_tm),max(id_max2)+1), dtype=int)
    for i in range(len(D2_tm)):
        for j in range(len(D2_tm[i])):
            winner = D2_tm[i][j]
            T2[i][winner] = 1

    # Create ground truth and show precision-recall curves
    GT_data = np.loadtxt(DIR + '/seq_multi_loop_noNoise_gt.txt', dtype='i', delimiter=',',skiprows=1)
    GT = np.zeros((numberImages,numberImages), dtype=int)
    for i in range(GT.shape[0]):
        for j in range(i,GT.shape[1]):
            GT[i,j] = (np.any(GT_data[i,:] != GT_data[j,:])==False)

    # Results
    print ('Results')
    fig, ax = plt.subplots()

    S0 = evaluateSimilarity(D)
    P, R = createPR(S0,GT)
    ax.plot(R, P, label='InputSDR: (avgP=%f)' %np.trapz(P,R))

    S1 = evaluateSimilarity(M)
    P, R = createPR(S1,GT)
    ax.plot(R, P, label='MCN (avgP=%f)' %np.trapz(P,R))

    S2 = evaluateSimilarity(T)
    P, R = createPR(S2,GT)
    ax.plot(R, P, label='HTM (avgP=%f)' %np.trapz(P,R))

    S3 = evaluateSimilarity(T2)
    P, R = createPR(S3,GT)
    ax.plot(R, P, label='HTM Distal (avgP=%f)' %np.trapz(P,R))

    ax.legend()
    ax.grid(True)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    '''
    fig2, (ax2, ax3) = plt.subplots(nrows=1, ncols=2) # two axes on figure

    ax2.imshow(S0, vmin=0, vmax=50, interpolation='nearest', cmap='binary')
    ax2.set_title('Input descriptors')

    ax3.imshow(S2, vmin=0, vmax=30, interpolation='nearest', cmap='binary')
    ax3.set_title('Winner cell outputs')

    plt.show()
    '''
                

if __name__ == "__main__":
    main()
