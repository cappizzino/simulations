#!/usr/bin/env python
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from nupic.encoders.scalar import ScalarEncoder
from nupic.algorithms.temporal_memory import TemporalMemory
from htmresearch.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakPairMemory as TM
from miniColumn import MCN

from utils import *

def label_plot(ax, is_pr=True):
    x, y = (('recall', 'precision')
    if is_pr else ('false pos rate', 'true pos rate'))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

class HTM(object):
    """Class implementing Traditional Temporal Memory"""
    def __init__(self,
                bottomUpInputSize,
                bottomUpOnBits,
                seed
                ):
        
        self.bottomUpInputSize = bottomUpInputSize
        self.bottomUpOnBits = bottomUpOnBits
        self.seed = seed

        self.trainingIterations = 0

        self.tm = TemporalMemory(
                # Must be the same dimensions as the SP
                columnDimensions=(self.bottomUpInputSize,),
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
                seed=self.seed
                )

    def compute(self, bottomUpSDR, learn):
        if learn:
        # During learning we provide the current pose angle as bottom up input
            self.train(bottomUpSDR)
            self.trainingIterations += 1
        else:
            print >>sys.stderr, "Learn: ", learn

    def train(self, bottomUp):
        #print >> sys.stderr, "Bottom up: ", bottomUp
        self.tm.compute(bottomUp,
                        learn=True)

class LVF(object):
    """Class implementing Localization with Vision Features"""

    def __init__(self,
                minX,
                maxX,
                minY,
                maxY,
                bottomUpInputSize,
                bottomUpOnBits,
                seed
                ):

        self.xEncoder = ScalarEncoder(5, minX, maxX, n=75, forced=True)
        self.yEncoder = ScalarEncoder(5, minY, maxY, n=75, forced=True)
        self.externalSize = self.xEncoder.getWidth()**2
        self.externalOnBits = self.xEncoder.w**2

        self.bottomUpInputSize = bottomUpInputSize
        self.bottomUpOnBits = bottomUpOnBits
        self.seed = seed

        self.trainingIterations = 0
        self.testIterations = 0
        self.maxPredictionError = 0
        self.totalPredictionError = 0
        self.numMissedPredictions = 0
        
        self.tm = TM(columnCount = self.bottomUpInputSize,
                basalInputSize = self.externalSize,
                cellsPerColumn=4,
                initialPermanence=0.4,
                connectedPermanence=0.5,
                minThreshold= self.externalOnBits,
                sampleSize=40,
                permanenceIncrement=0.1,
                permanenceDecrement=0.00,
                activationThreshold=int(0.75*(self.externalOnBits+self.bottomUpOnBits)),
                basalPredictedSegmentDecrement=0.00,
                seed = self.seed
                )
    
    def compute(self, x, y, bottomUpSDR, learn):
        # Encode the inputs appropriately and train the HTM
        externalSDR = self.encodePosition(x,y)

        if learn:
        # During learning we provide the current pose angle as bottom up input
            self.trainTM(bottomUpSDR, externalSDR)
            self.trainingIterations += 1
        else:
            print >>sys.stderr, "Learn: ", learn
    
    def encodePosition(self, x, y):
        """Return the SDR for x,y"""
        xe = self.xEncoder.encode(x)
        ye = self.yEncoder.encode(y)
        ex = np.outer(xe,ye)
        return ex.flatten().nonzero()[0]

    def trainTM(self, bottomUp, externalInput):
        #print >> sys.stderr, "Bottom up: ", bottomUp
        #print >> sys.stderr, "ExternalInput: ",externalInput
        self.tm.compute(bottomUp,
                        basalInput=externalInput,
                        learn=True)

class Params:
    pass

def load(DIR):
    # Visual input
    D = np.loadtxt(DIR + '/seq_multi_loop_noise05_al5.txt', dtype='i', delimiter=',')
    numberImages = D[:,0].size
    nColumns = D[0,:].size

    # Odom input
    odom = np.loadtxt(DIR + '/seq_multi_loop_noise05_al5_gt.txt', dtype='f', delimiter=',')
    x = odom[:,0]
    y = odom[:,1]
    x = (x + 0.0075*np.random.randn(1, x.size))[0]
    y = (y + 0.0075*np.random.randn(1, y.size))[0]

    return D, numberImages, nColumns, x, y

def main(seed):

    DIR = "./sim_data"
    D, numberImages, nColumns, x, y = load(DIR)

    # Network LVF
    lvf = LVF(minX=np.min(x),
            maxX=np.max(x),
            minY=np.min(y),
            maxY=np.max(y),
            bottomUpInputSize=nColumns, 
            bottomUpOnBits=40,
            seed=seed)
    
    # Network HTM
    htm = HTM(bottomUpInputSize=nColumns, 
            bottomUpOnBits=40,
            seed=seed)

    # Minicolumn Network
    params = Params()
    params.maxPredDepth = 0
    params.probAdditionalCon = 0.05 # probability for random connection
    params.nCellPerCol = 32         # number of cells per minicolumn
    params.nInConPerCol = int(round(np.count_nonzero(D) / D.shape[0]))
    params.minColumnActivity = int(round(0.25*params.nInConPerCol))
    params.nColsPerPattern = 10     # minimum number of active minicolumns k_min
    params.kActiveColumn = 100      # maximum number of active minicolumns k_max
    params.kMin = 1

    mcn = MCN('MCN',params)
    '''
    # *******************************************
    # Experiment with Apical Tiebreak Pair Memory
    # *******************************************
    print 'Apical Tiebreak Pair Memory'
    outputSDR = []
    max_index = []
    t = time.time()

    for i in range(numberImages):
        lvf.compute(x[i], y[i], bottomUpSDR=np.nonzero(D[i,:])[0], learn=True)
        winnerCells = lvf.tm.getWinnerCells()
        outputSDR .append(winnerCells)
        max_index.append(max(winnerCells))

    # create output SDR matrix from winner cell output
    M = np.zeros((len(outputSDR),max(max_index)+1), dtype=int)
    for i in range(len(outputSDR)):
        for j in range(len(outputSDR[i])):
            winner = outputSDR[i][j]
            M[i][winner] = 1

    elapsed = time.time() - t
    print( "Elapsed time: %f seconds\n" %elapsed)

    # *******************************************
    # Experiment with HTM
    # *******************************************
    print 'Temporal Pooler descriptors'
    outputSDR = []
    max_index = []
    t = time.time()

    for i in range(numberImages):
        htm.compute(bottomUpSDR=np.nonzero(D[i,:])[0], learn=True)
        winnerCells = htm.tm.getWinnerCells()
        outputSDR .append(winnerCells)
        max_index.append(max(winnerCells))

    # create output SDR matrix from winner cell output
    T = np.zeros((len(outputSDR),max(max_index)+1), dtype=int)
    for i in range(len(outputSDR)):
        for j in range(len(outputSDR[i])):
            winner = outputSDR[i][j]
            T[i][winner] = 1

    elapsed = time.time() - t
    print( "Elapsed time: %f seconds\n" %elapsed)
    '''
    # *******************************************
    # Experiment with MCN
    # *******************************************
    #print 'Minicolumn descriptors'
    outputSDR = []
    max_index = []
    t = time.time()

    for i in range(numberImages):
        mcn.compute(D[i,:])
        outputSDR.append(mcn.winnerCells)
        max_index.append(max(mcn.winnerCells))

    # create output SDR matrix from winner cell output
    MC = np.zeros((len(outputSDR),max(max_index)+1), dtype=int)
    for i in range(len(outputSDR)):
        for j in range(len(outputSDR[i])):
            winner = outputSDR[i][j]
            MC[i][winner] = 1

    elapsed = time.time() - t
    #print( "Elapsed time: %f seconds\n" %elapsed)

    # *******************************************
    # Create ground truth
    # *******************************************
    GT_data = np.loadtxt(DIR + '/seq_multi_loop_noNoise_gt.txt', dtype='i', delimiter=',',skiprows=1)
    GT = np.zeros((numberImages,numberImages), dtype=int)
    for i in range(GT.shape[0]):
        for j in range(i,GT.shape[1]):
            GT[i,j] = (np.any(GT_data[i,:] != GT_data[j,:])==False)
    
    # *******************************************
    # Results
    # *******************************************
    print ('Results')
    #fig, ax = plt.subplots()

    #S0 = evaluateSimilarity(D)
    #P, R = createPR(S0,GT)
    #ax.plot(R, P, label='InputSDR: (avgP=%f)' %np.trapz(P,R))

    #S1 = evaluateSimilarity(M)
    #P, R = createPR(S1,GT)
    #ax.plot(R, P, label='LVF (avgP=%f)' %np.trapz(P,R))

    #S2 = evaluateSimilarity(T)
    #P, R = createPR(S2,GT)
    #ax.plot(R, P, label='HTM (avgP=%f)' %np.trapz(P,R))

    S3 = evaluateSimilarity(MC)
    P, R, T = createPRT(S3,GT,10)
    #ax.plot(R, P, label='MCN (avgP=%f)' %np.trapz(P,R))

    return P, R, T

    #ax.legend()
    #ax.grid(True)
    #plt.xlabel("Recall")
    #plt.ylabel("Precision")
    #plt.show()

if __name__ == "__main__":


    nExp = 10

    curves = []
    for i in range(nExp):
        curves.append(main(seed = i + 120))

    sampled_thresholds = np.linspace(0.0, 1.0, 100)
    sampled_precisions = []
    sampled_recalls = []

    # assume curves is a list of (precision, recall, threshold)
    # tuples where each of those three is a numpy array
    for precision, recall, threshold in curves:
        sampled_precisions.append(np.interp(sampled_thresholds, threshold[:-1], precision[:-1]))
        sampled_recalls.append(np.interp(sampled_thresholds, threshold[:-1], recall[:-1]))

    fig, ax = plt.subplots()

    GREEN = '#72d218'
    RED = '#f4424b'
    BLUE = '#72bdff'

    ax.errorbar(
        np.mean(sampled_recalls, axis=0),
        np.mean(sampled_precisions, axis=0),
        np.std(sampled_recalls, axis=0),
        np.std(sampled_precisions, axis=0),
        color=GREEN,
        errorevery=5,
        ecolor=RED,
        lw=2)

    label_plot(ax)
    plt.show()


'''
    meanP = np.mean(Pm, axis=0)
    meanR = np.mean(Rm, axis=0)

    plt.plot(meanR, meanP, lw=2,color='red')

    stdP = np.std(Pm, axis=0)
    plt.fill_between(meanR, meanP + stdP, meanP - stdP, alpha=0.3, linewidth=0, color='grey')

    ax.legend()
    ax.grid(True)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.show()
    '''