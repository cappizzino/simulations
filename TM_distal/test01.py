#!/usr/bin/env python
import numpy as np
from nupic.algorithms.temporal_memory import TemporalMemory
from group_by import groupby2

from nupic.bindings.math import SparseMatrixConnections

class TM(TemporalMemory):
  def __init__(self, **kwargs):
    super(TM, self).__init__(**kwargs)
  
  def test(self):
      print 'test'

  def activateCells(self, activeColumns, gridCell=[], learn=True):
    """
    Calculate the active cells, using the current active columns and dendrite
    segments. Grow and reinforce synapses.
    :param activeColumns: (iter) A sorted list of active column indices.
    :param learn: (bool) If true, reinforce / punish / grow synapses.
      **Pseudocode:**
      
      ::
        for each column
          if column is active and has active distal dendrite segments
            call activatePredictedColumn
          if column is active and doesn't have active distal dendrite segments
            call burstColumn
          if column is inactive and has matching distal dendrite segments
            call punishPredictedColumn
    """
    print 'Activate Cells'
    print gridCell
    prevActiveCells = self.activeCells
    prevWinnerCells = self.winnerCells
    #prevWinnerCells = gridCell
    self.activeCells = []
    self.winnerCells = []

    segToCol = lambda segment: int(segment.cell / self.cellsPerColumn)
    identity = lambda x: x

    for columnData in groupby2(activeColumns, identity,
                               self.activeSegments, segToCol,
                               self.matchingSegments, segToCol):
      (column,
       activeColumns,
       columnActiveSegments,
       columnMatchingSegments) = columnData
      if activeColumns is not None:
        if columnActiveSegments is not None:
          cellsToAdd = self.activatePredictedColumn(column,
                                                    columnActiveSegments,
                                                    columnMatchingSegments,
                                                    prevActiveCells,
                                                    prevWinnerCells,
                                                    learn)

          self.activeCells += cellsToAdd
          self.winnerCells += cellsToAdd
        else:
          (cellsToAdd,
           winnerCell) = self.burstColumn(column,
                                          columnMatchingSegments,
                                          prevActiveCells,
                                          prevWinnerCells,
                                          learn)

          self.activeCells += cellsToAdd
          self.winnerCells.append(winnerCell)
      else:
        if learn:
          self.punishPredictedColumn(column,
                                     columnActiveSegments,
                                     columnMatchingSegments,
                                     prevActiveCells,
                                     prevWinnerCells)

  def burstColumn(self, column, columnMatchingSegments, prevActiveCells,
                  prevWinnerCells, learn):

    #print 'Column:' + repr(column)
    start = self.cellsPerColumn * column
    cellsForColumn = xrange(start, start + self.cellsPerColumn)
    #print self.connections.numSynapses(columnMatchingSegments)

    return self._burstColumn(
      self.connections, self._random, self.lastUsedIterationForSegment, column,
      columnMatchingSegments, prevActiveCells, prevWinnerCells, cellsForColumn,
      self.numActivePotentialSynapsesForSegment, self.iteration,
      self.maxNewSynapseCount, self.initialPermanence, self.permanenceIncrement,
      self.permanenceDecrement, self.maxSegmentsPerCell,
      self.maxSynapsesPerSegment, learn)

 
D = [[1,0,0,0,1,0,0,0,0,1],[0,1,0,1,0,0,0,0,1,0]]
x = [[1,0,1,0],[0,1,0,1]]
x_cell = [[40,42,44],[41,43,45]]

seg = []
Din0 =  np.nonzero(D[0])[0]
Din1 =  np.nonzero(D[1])[0]

xin0 = np.nonzero(x[0])[0]
xin1 = np.nonzero(x[1])[0]

x_cell0 = x_cell[0]
x_cell1 = x_cell[1]

tm = TM(
    # Must be the same dimensions as the SP
    columnDimensions=(10,),
    # How many cells in each mini-column.
    cellsPerColumn=4,
    # A segment is active if it has >= activationThreshold connected synapses
    # that are active due to infActiveState
    activationThreshold=2,
    initialPermanence=0.21,
    connectedPermanence=0.0,
    # Minimum number of active synapses for a segment to be considered during
    # search for the best-matching segments.
    minThreshold=1,
    # The max number of synapses added to a segment during learning
    maxNewSynapseCount=3,
    permanenceIncrement=0.01,
    permanenceDecrement=0.01,
    predictedSegmentDecrement=0.0005,
    maxSegmentsPerCell=2,
    maxSynapsesPerSegment=3,
    seed=42
)

tm0 = TM(
    # Must be the same dimensions as the SP
    columnDimensions=(10,),
    # How many cells in each mini-column.
    cellsPerColumn=4,
    # A segment is active if it has >= activationThreshold connected synapses
    # that are active due to infActiveState
    activationThreshold=2,
    initialPermanence=0.21,
    connectedPermanence=0.0,
    # Minimum number of active synapses for a segment to be considered during
    # search for the best-matching segments.
    minThreshold=1,
    # The max number of synapses added to a segment during learning
    maxNewSynapseCount=3,
    permanenceIncrement=0.01,
    permanenceDecrement=0.01,
    predictedSegmentDecrement=0.0005,
    maxSegmentsPerCell=255,
    maxSynapsesPerSegment=255,
    seed=42
)

def printConnectionDetails(tm, column):
    for cell in range(tm.cellsPerColumn):
        cell = cell + tm.cellsPerColumn*column
        segments = tm.connections.segmentsForCell(cell)
        print segments
        for segment in segments:
            num_synapses = tm.connections.numSynapses(segment)
            for synapse in tm.connections.synapsesForSegment(segment):
                presynCell = tm.connections.synapsesForPresynapticCell(synapse)                    
                permanence = tm.connections.dataForSynapse(synapse)
                print presynCell
                #print('cell', format(cell,'2d'), 'segment', format(segment,'2d'), 'has synapse to cell', format(presynCell,'2d'), 'with permanence', format(permanence,'.2f'))
            connected_synapses = tm.connections.numConnectedSynapses(segment)
            print('cell', format(cell,'2d'), 'segment', format(segment,'2d'), 'has', connected_synapses, 'connected synapse(s)')

'''
tm.activateCells(sorted(Din0), learn=True)
#print tm.connections.computeActivity(tm.activeCells,tm.connectedPermanence)
tm.activeCells = xin0
tm.activateDendrites(learn=True)
#print tm.getActiveCells()
#print tm.getWinnerCells()
print tm.getPredictiveCells()

tm.activateCells(sorted(Din1), learn=True)
tm.activeCells = xin1
tm.activateDendrites(learn=True)
#print tm.getActiveCells()
#print tm.getWinnerCells()
print tm.getPredictiveCells()

tm.activateCells(sorted(Din0), learn=True)
tm.activeCells = xin0
tm.activateDendrites(learn=True)
#print tm.getActiveCells()
#print tm.getWinnerCells()
print tm.getPredictiveCells()

print tm.connections
#tm.activateDendrites(learn=True)
'''
# Perform one time step of the Temporal Memory algorithm.
print ''
tm0.activateCells(sorted(Din0))
tm0.activateDendrites(learn=True)
print 'Active Cells: ' + repr(tm0.activeCells)
print 'Winner Cells: ' + repr(tm0.winnerCells)
print 'Predictive Cells: ' + repr(tm0.getPredictiveCells())

print ''
tm0.activateCells(sorted(Din1),gridCell=x_cell[0])
tm0.activateDendrites(learn=True)
print 'Active Cells: ' + repr(tm0.activeCells)
print 'Winner Cells: ' + repr(tm0.winnerCells)
print 'Predictive Cells: ' + repr(tm0.getPredictiveCells())
print 'Data: ' + repr(tm0.connections.)

print ''
tm0.activateCells(sorted(Din0),gridCell=x_cell[1])
tm0.activateDendrites(learn=True)
print 'Active Cells: ' + repr(tm0.activeCells)
print 'Winner Cells: ' + repr(tm0.winnerCells)
print 'Predictive Cells: ' + repr(tm0.getPredictiveCells())


#print tm0.activeCells
#print tm0.connections.computeActivity(tm0.activeCells,tm0.connectedPermanence)

#print tm0.connections.segmentFlatListLength()

#print tm0.connections.segmentsForCell(6)
#print tm0.connections.numSynapses(segment)

#segment = tm.connections.segmentsForCell(6)

#print tm.connections.getSegment(1,0)

#tm.createSegment(6)

#print tm.connections.numSegments()
#print tm0.connections.numSegments()
#print tm.connections.numSegments(1)
#print tm.connections.numSegments(6)
#print tm.connections.numSegments(15)
#print tm.connections.numSegments(16)
#print tm.connections.numSegments(39)
#print tm.connections.numSegments(34)


'''

for column in Din0:
    print tm0.getMatchingSegments()
    tm0.burstColumn(column,tm0.getMatchingSegments(),tm0.getActiveCells(),tm0.getWinnerCells(), learn=True)

for column in xrange(tm.getColumnDimensions()[0]):
    connected = np.zeros((tm.getCellsPerColumn(),), dtype=int)
    tm.getConnectedPermanence(column,connected)
    print connected
'''