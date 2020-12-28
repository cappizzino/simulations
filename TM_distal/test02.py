#!/usr/bin/env python
import numpy as np
from itertools import izip as zip, count

#from htmresearch.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakSequenceMemory as TM
from htmresearch.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakPairMemory as TM
#from htmresearch.algorithms.apical_dependent_temporal_memory import TripleMemory as TM

# Utility routine for printing the input vector
def formatRow(x):
  s = ''
  for c in range(len(x)):
    if c > 0 and c % 10 == 0:
      s += ' '
    s += str(x[c])
  s += ' '
  return s

# Step 1: create Temporal Pooler instance with appropriate parameters
tm = TM(columnCount=10,
        basalInputSize= 4,
        cellsPerColumn=4,
        initialPermanence=0.21,
        connectedPermanence=0.0,
        minThreshold=1,
        reducedBasalThreshold=1,
        permanenceIncrement=0.1,
        permanenceDecrement=0.0,
        activationThreshold=2,
        apicalInputSize=4,
        )

# Step 2: create input vectors to feed to the temporal memory. 
D = [[1,0,0,0,1,0,0,0,0,1],[0,1,0,1,0,0,0,0,1,0],[1,1,0,0,0,0,1,0,0,0]]
x = [[1,0,1,0],[0,1,0,1],[1,1,0,0]]

Din0 =  np.nonzero(D[0])[0]
Din1 =  np.nonzero(D[1])[0]
Din2 =  np.nonzero(D[2])[0]

xin0 = np.nonzero(x[0])[0]
xin1 = np.nonzero(x[1])[0]
xin2 = np.nonzero(x[2])[0]

# Step 3: send this simple sequence to the temporal memory for learning
print ''
tm.compute(sorted(Din0), xin0, apicalInput=(), basalGrowthCandidates=None, apicalGrowthCandidates=None, learn=True)
print 'Active Cells: ' + repr(tm.activeCells)
print 'Winner Cells: ' + repr(tm.winnerCells)
print tm.predictedCells

print ''
tm.compute(sorted(Din1), xin1, apicalInput=(), basalGrowthCandidates=None, apicalGrowthCandidates=None, learn=True)
print 'Active Cells: ' + repr(tm.activeCells)
print 'Winner Cells: ' + repr(tm.winnerCells)
print tm.predictedCells

print ''
tm.compute(sorted(Din2), xin2, apicalInput=(), basalGrowthCandidates=None, apicalGrowthCandidates=None, learn=True)
print 'Active Cells: ' + repr(tm.activeCells)
print 'Winner Cells: ' + repr(tm.winnerCells)
print tm.predictedCells

print ''
tm.compute(sorted(Din1), xin1, apicalInput=(), basalGrowthCandidates=None, apicalGrowthCandidates=None, learn=True)
print 'Active Cells: ' + repr(tm.activeCells)
print 'Winner Cells: ' + repr(tm.winnerCells)
print tm.predictedCells

print ''
tm.compute(sorted(Din2), xin2, apicalInput=(), basalGrowthCandidates=None, apicalGrowthCandidates=None, learn=True)
print 'Active Cells: ' + repr(tm.activeCells)
print 'Winner Cells: ' + repr(tm.winnerCells)
print tm.predictedCells