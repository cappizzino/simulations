#!/usr/bin/env python
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from nupic.encoders.scalar import ScalarEncoder
from htmresearch.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakPairMemory as TM

from utils import *

class LVF(object):
    """Class implementing Localization with Vision Features"""

    def __init__(self,
                minX,
                maxX,
                minY,
                maxY,
                bottomUpInputSize,
                bottomUpOnBits,
                ):

        self.xEncoder = ScalarEncoder(5, minX, maxX, n=75, forced=True)
        self.yEncoder = ScalarEncoder(5, minY, maxY, n=75, forced=True)
        self.externalSize = self.xEncoder.getWidth()**2
        self.externalOnBits = self.xEncoder.w**2

        self.bottomUpInputSize = bottomUpInputSize
        self.bottomUpOnBits = bottomUpOnBits

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

def main():

    DIR = "./sim_data"

    # Visual input
    D = np.loadtxt(DIR + '/seq_multi_loop_noise05_al5.txt', dtype='i', delimiter=',')
    numberImages = D[:,0].size
    nColumns = D[0,:].size

    # Odom input
    odom = np.loadtxt(DIR + '/seq_multi_loop_noise05_al5_gt.txt', dtype='f', delimiter=',')
    x = odom[:,0]
    y = odom[:,1]

    # Network LVF
    lvf = LVF(minX=np.min(x),
            maxX=np.max(x),
            minY=np.min(y),
            maxY=np.max(y),
            bottomUpInputSize=nColumns, 
            bottomUpOnBits=40)

    # Experiment with Apical Tiebreak Pair Memory
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
    ax.plot(R, P, label='LVF (avgP=%f)' %np.trapz(P,R))

    ax.legend()
    ax.grid(True)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

if __name__ == "__main__":
    main()