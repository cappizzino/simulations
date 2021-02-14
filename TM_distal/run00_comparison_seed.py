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

        self.xEncoder = ScalarEncoder(5, minX, 10*maxX, n=75, forced=True)
        self.yEncoder = ScalarEncoder(5, minY, 10*maxY, n=75, forced=True)
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
    noise = 0.05
    odom = np.loadtxt(DIR + '/seq_multi_loop_noise05_al5_gt.txt', dtype='f', delimiter=',')
    x = odom[:,0]
    y = odom[:,1]
    x = (x + noise*np.random.randn(1, x.size))[0]
    y = (y + noise*np.random.randn(1, y.size))[0]

    # *******************************************
    # Create ground truth
    # *******************************************
    GT_data = np.loadtxt(DIR + '/seq_multi_loop_noNoise_gt.txt', dtype='i', delimiter=',',skiprows=1)
    GT = np.zeros((numberImages,numberImages), dtype=int)
    for i in range(GT.shape[0]):
        for j in range(i,GT.shape[1]):
            GT[i,j] = (np.any(GT_data[i,:] != GT_data[j,:])==False)

    return D, numberImages, nColumns, x, y, GT

def main(seed):

    #DIR = "./sim_data"
    #D, numberImages, nColumns, x, y = load(DIR)

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
    
    # *******************************************
    # Experiment with MCN
    # *******************************************
    print 'Minicolumn descriptors'
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
    print( "Elapsed time: %f seconds\n" %elapsed)

    # *******************************************
    # Create ground truth
    # *******************************************
    #GT_data = np.loadtxt(DIR + '/seq_multi_loop_noNoise_gt.txt', dtype='i', delimiter=',',skiprows=1)
    #GT = np.zeros((numberImages,numberImages), dtype=int)
    #for i in range(GT.shape[0]):
    #    for j in range(i,GT.shape[1]):
    #        GT[i,j] = (np.any(GT_data[i,:] != GT_data[j,:])==False)
    
    # *******************************************
    # Results
    # *******************************************
    print ('Results')
    #fig, ax = plt.subplots()

    S0 = evaluateSimilarity(D)
    P0, R0 = createPR(S0,GT)
    #ax.plot(R, P, label='InputSDR: (avgP=%f)' %np.trapz(P,R))

    S1 = evaluateSimilarity(M)
    P1, R1 = createPR(S1,GT)
    #ax.plot(R, P, label='LVF (avgP=%f)' %np.trapz(P,R))

    S2 = evaluateSimilarity(T)
    P2, R2 = createPR(S2,GT)
    #ax.plot(R, P, label='HTM (avgP=%f)' %np.trapz(P,R))

    S3 = evaluateSimilarity(MC)
    P3, R3 = createPR(S3,GT)
    #ax.plot(R, P, label='MCN (avgP=%f)' %np.trapz(P,R))

    return P0, R0, P1, R1, P2, R2, P3, R3

    #ax.legend()
    #ax.grid(True)
    #plt.xlabel("Recall")
    #plt.ylabel("Precision")
    #plt.show()

if __name__ == "__main__":

    # load Data
    DIR = "./sim_data"
    D, numberImages, nColumns, x, y, GT = load(DIR)

    nExp = 5

    Pm0 = np.zeros((nExp, 101))
    Rm0 = np.zeros((nExp, 101))
    avgP0 = np.zeros(nExp)

    Pm1 = np.zeros((nExp, 101))
    Rm1 = np.zeros((nExp, 101))
    avgP1 = np.zeros(nExp)

    Pm2 = np.zeros((nExp, 101))
    Rm2 = np.zeros((nExp, 101))
    avgP2 = np.zeros(nExp)

    Pm3 = np.zeros((nExp, 101))
    Rm3 = np.zeros((nExp, 101))
    avgP3 = np.zeros(nExp)

    for i in range(nExp):
        Pm0[i,:], Rm0[i,:], Pm1[i,:], Rm1[i,:], Pm2[i,:], Rm2[i,:], Pm3[i,:], Rm3[i,:] = main(seed = i + 42)
        avgP0[i] = np.trapz(Pm0[i,:],Rm0[i,:])
        avgP1[i] = np.trapz(Pm1[i,:],Rm1[i,:])
        avgP2[i] = np.trapz(Pm2[i,:],Rm2[i,:])
        avgP3[i] = np.trapz(Pm3[i,:],Rm3[i,:])
        #print( "AVG: %f\n" %avgP1[i])
        #ax.plot(Rm[i,:], Pm[i,:], alpha=0.3, label='MCN (avgP=%f)' %np.trapz(Pm[i,:],Rm[i,:]))

    #meanP0 = np.mean(Pm0, axis=0)
    #meanR0 = np.mean(Rm0, axis=0)
    avgP0_mean = np.mean(avgP0)

    #meanP1 = np.mean(Pm1, axis=0)
    #meanR1 = np.mean(Rm1, axis=0)
    avgP1_mean = np.mean(avgP1)

    #meanP2 = np.mean(Pm2, axis=0)
    #meanR2 = np.mean(Rm2, axis=0)
    avgP2_mean = np.mean(avgP2)

    #meanP3 = np.mean(Pm3, axis=0)
    #meanR3 = np.mean(Rm3, axis=0)
    avgP3_mean = np.mean(avgP3)   

    # Calculate the standard deviation
    avgP0_std = np.std(avgP0)
    avgP1_std = np.std(avgP1)
    avgP2_std = np.std(avgP3)
    avgP3_std = np.std(avgP3)

    # Create lists for the plot
    alg = ['InputSDR', 'HTM-DC', 'HTM-CLA', 'MCN']
    x_pos = np.arange(len(alg))
    CTEs = [avgP0_mean, avgP1_mean, avgP2_mean, avgP3_mean]
    error = [avgP0_std, avgP1_std, avgP2_std, avgP3_std]

    # Build the plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x, y)
    ax1.grid(True)
    ax1.axis('equal')
    ax1.set_title('Odometry position')
    ax1.set_xlabel("x(m)")
    ax1.set_ylabel("y(m)")

    ax2.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax2.set_ylabel('Average')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(alg)
    ax2.set_title('Average')
    ax2.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    #plt.savefig('bar_plot_with_error_bars.png')
    plt.show()

    #plt.plot(meanR, meanP, lw=2,color='red')

    #stdP = np.std(Pm, axis=0)
    #plt.fill_between(meanR, meanP + stdP, meanP - stdP, alpha=0.3, linewidth=0, color='grey')

    #ax.legend()
    #ax.grid(True)
    #plt.xlabel("Recall")
    #plt.ylabel("Precision")
    #plt.show()