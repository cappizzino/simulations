# Software release for the paper:
# "A Sequence-Based Neuronal Model for Mobile Robot Localization"
# Peer Neubert, Subutai Ahmad, and Peter Protzel, Proc. of German 
# Conference on Artificial Intelligence, 2018 
#
# Simplified Higher Order Sequence Memory from HTM.
# No learning in spatial pooler. Starting with an empty set of minicolumns, 
# each time a feedforward pattern is seen, that is not similar to any 
# existing minicolumn, create a new one witch responds to thinsg similar to
# this pattern.
#
import sys
import numpy as np
import random
import time

class MCN:

    def __init__(self, name, params):
        self.name = name
        self.params = params

        # feedforward connections: FF(:,j) is the set of indices into the input
        # SDR of the j-th minicolumn. So the activation of all minicolumns can
        # be computed by sum(SDR(FF))
        self.FF = np.zeros((self.params.nInConPerCol,1), dtype=int)
        #self.FF = np.zeros((self.params.nInConPerCol,), dtype=int)
            
        # P(i,j) is the predicted-flag for the i-th cell in the j-th minicolumn
        self.P = np.zeros((self.params.nCellPerCol,1), dtype=int)
        self.prevP = np.zeros((self.params.nCellPerCol,1), dtype=int)

        # idx of all winning cells (in matrix of same size as P)
        self.winnerCells  = []
        self.prevWinnerCells  = []

        # idx of all active cells (in matrix of same size as P)
        self.activeCells  = np.array([], dtype=int)
        self.prevActiveCells  = np.array([], dtype=int)
        
        # flag if this minicolumn bursts
        self.burstedCol = np.zeros((1,), dtype=int)
        self.prevBurstedCol = np.zeros((1,), dtype=int)

        # datastructure that holds the connections for predictions
        self.predictionConnections = []
        self.predictionConnections_backward = []

        # current number of minicolumns in the network
        self.nCols = 0

        self.params.nReserveCols = 10000
        self.new = 0

    def __str__(self):
        return self.name

    # reset some internal variables
    def prepareNewIteration(self):
        self.prevWinnerCells = self.winnerCells
        self.winnerCells = []
        self.prevActiveCells = self.activeCells
        self.activeCells = []

        self.prevP = self.P
        self.prevBurstedCol = self.burstedCol

        if self.nCols > 0:
            self.P[:,:] = 0
            self.burstedCol[:,:] = 0

    # InputConnections ... idx in a potential input SDR ,stored in obj.FF
    # Return index of the ne column 
    def createNewColumn(self, inputConnections, nNewCols):

        for i in range(nNewCols):
            self.nCols = self.nCols + 1

            if self.nCols == self.FF.shape[1]:
                self.reserveStorage()

            indices_t = np.concatenate(inputConnections.transpose())
            self.FF[:,(self.nCols-1)] = np.random.choice(indices_t.transpose(), size = self.params.nInConPerCol, replace=True)
        
        newColIdx = np.arange(start=self.nCols-nNewCols, stop=self.nCols)

        return newColIdx

    def reserveStorage(self):
        k = self.params.nReserveCols - 1

        self.FF = np.c_[self.FF, np.zeros((self.params.nInConPerCol,k), dtype=int)]
        self.P = np.c_[self.P, np.zeros((self.params.nCellPerCol,k), dtype=int)]
        self.prevP = np.c_[self.prevP, np.zeros((self.params.nCellPerCol,k), dtype=int)]
        self.burstedCol = np.c_[self.burstedCol, np.zeros((1,k), dtype=int)]
        self.prevBurstedCol = np.c_[self.prevBurstedCol, np.zeros((1,k), dtype=int)]

        # datastructure that hold the connections for predictions. It is a
        # cell array of matrices.  predictionConnections{i,j} is a two row matrix
        # with each column [idx in P; permanence] <--- now: only first row yet
        '''
        for i in range(0, self.params.nCellPerCol, 1):
            linha = []
            for j in range(0, k+1, 1):
                linha.append([])
            self.predictionConnections.append(linha)

        for i in range(0, self.params.nCellPerCol, 1):
            linha = []
            for j in range(0, k+1, 1):
                linha.append([])
            self.predictionConnections_backward.append(linha)        
        '''
        for i in range(0, self.params.nCellPerCol, 1):
            linha = []
            for j in range(self.nCols-1, self.nCols-1+k+1, 1):
                linha.append([])
            self.predictionConnections.append(linha)

        for i in range(0, self.params.nCellPerCol, 1):
            linha = []
            for j in range(self.nCols-1, self.nCols-1+k+1, 1):
                linha.append([])
            self.predictionConnections_backward.append(linha)

    # Compare inputSDR to all minicolumns to find active minicolumns
    # Search for predicted cells in active minicolumns and activate their predictions
    def compute(self, inputSDR, supressLearningFlag):
        # flag to supress learning
        self.supressLearningFlag = supressLearningFlag

        # prepare new iteration
        self.prepareNewIteration()

        # How similar is the input SDR to the pattern of the minicolumns?
        if self.nCols != 0:
            start = 1
            columnActivity = self.computeColumnActivations(inputSDR)
            columnActivity_sorted = np.sort(columnActivity)[::-1]
            idx_sorted = np.argsort(columnActivity)[::-1]
        else:
            start = 0
            columnActivity = 0
            columnActivity_sorted = 0
            idx_sorted = 0

        if not self.supressLearningFlag:
            # Are there activities above threshold? If yes, activate the k most
            # active columns, otherwise create new ones and make these the active ones.
            if start != 0:
                cond1 = np.greater(columnActivity_sorted, self.params.minColumnActivity)
                cond3 = min(self.params.kActiveColumn,len(columnActivity_sorted))
                cond2 = np.greater_equal(columnActivity_sorted, columnActivity_sorted[cond3-1])
                activeCols = idx_sorted[cond1&cond2]
            else:
                activeCols = []
            
            sdrNonZeroIdx = np.argwhere(inputSDR == 1)
            cond1 = max(0, self.params.nColsPerPattern - len(activeCols))
            activeCols = np.concatenate([activeCols, self.createNewColumn(sdrNonZeroIdx, cond1)])
        else:
            # In non-learning mode, take the k most active columns
            # plus columns with same activity like kth-best column
            cond3 = min(self.params.kActiveColumn,len(columnActivity_sorted))
            cond2 = np.greater_equal(columnActivity_sorted, columnActivity_sorted[cond3-1])
            activeCols = idx_sorted[cond2]
        
        activeCols = activeCols.astype(int)

        # for each active column:
        # - mark all predicted cells as winnerCells
        # - if there was no predicted cell, chose one and activate all predictions
        # - activate predictions of winnerCells
        self.activeCells = []
        self.winnerCells = []

        for activeCol in activeCols:
            predictedIdx = np.argwhere(self.prevP[:,activeCol]>0)
 
            if predictedIdx.size == 0:
                # if there are no predicted: burst (predict from all cells
                # and choose one winner cell)
                winnerCell = self.burst(activeCol)
                index = np.ravel_multi_index([winnerCell,activeCol], self.P.shape, order='F')
                self.winnerCells.append(index)
            elif predictedIdx.size == 1:
                # if there is only one predicted cell, make this the winner cell
                winnerCell = np.asscalar(predictedIdx)
                self.activatePredictions(winnerCell, activeCol)
                index = np.ravel_multi_index([winnerCell,activeCol], self.P.shape, order='F')
                self.winnerCells.append(index)
            else:
                # if there are multiple predicted cells, make all winner cells
                for j in predictedIdx:
                    self.activatePredictions(np.asscalar(j),activeCol)
                    index = np.ravel_multi_index([np.asscalar(j),activeCol], self.P.shape, order='F')
                    self.winnerCells.append(index) 

        # learn predictions
        if not self.supressLearningFlag:
            self.learnPredictions()
        
            # also predict newly learned predictions
            for columnIdx in range(self.nCols):
                if self.burstedCol[0,columnIdx] == 1:
                    for i in range(self.P.shape[0]):
                        self.activatePredictions(i, columnIdx)

    # given the set of currently winning cells and the set of previously
    # winning cells, create prediction connection 
    def learnPredictions(self):
        for curIdx in self.winnerCells:
            [curCellIdx, curColIdx] = np.unravel_index(curIdx, self.P.shape, order='F')

            for prevIdx in self.prevWinnerCells:
                [prevCellIdx, prevColIdx] = np.unravel_index(prevIdx, self.P.shape, order='F')
                
                # check whether the previous column is already connected to the cell
                existingPredConFlag = self.checkExistingPredCon(prevColIdx, curIdx)
                if (not existingPredConFlag) or (random.random()<=self.params.probAdditionalCon):
                    self.predictionConnections_backward[curCellIdx][curColIdx].append(prevIdx)
                    self.predictionConnections[prevCellIdx][prevColIdx].append(curIdx)

            # ckeck for identical connections if random connections are enabled
            if self.params.probAdditionalCon != 0:
                if self.predictionConnections_backward[curCellIdx][curColIdx]!=[]:
                    self.predictionConnections_backward[curCellIdx][curColIdx] = self.unique(self.predictionConnections_backward[curCellIdx][curColIdx])

        
    # Check if there already is an predicting connection from the previous
    # column to this active cell. This is used during bursting to prevent
    # learning multiple connections from one column to a single cell. In
    # this case, the new connection should go to a new cell of the current
    # collumn, to indicate the different context.
    def checkExistingPredCon(self, prevColIdx, curCellIdx):
        [prevCellIdxs, prevColIdxs] = np.unravel_index(curCellIdx, self.P.shape, order='F')
        existingPredConFlag = np.any(prevColIdxs==prevColIdx)

        return existingPredConFlag
        
    # Activate the predictions of all cells and identify the cell with the
    # fewest forward predictions to be the winning cell. winnerCellIdx is the 
    # index of this cell in the minicolumn
    def burst(self, columnIdx):
        self.burstedCol[:,columnIdx] = 1

        for i in range(self.P.shape[0]):
            self.activatePredictions(i, columnIdx)
        
        # winnerCell: one of the cells with fewest existing forward predictions     
        nForwardPredictionsPerCell = []
        for i in range(self.params.nCellPerCol):
            nForwardPredictionsPerCell.append(len(self.predictionConnections[i][columnIdx]))

        if not self.supressLearningFlag:
            # (slightly) inhibit winning cells from the last iteration a little bit
            for i in range(len(self.prevWinnerCells)):
                if self.prevWinnerCells != []:
                    [cellIdx, colIdx] = np.unravel_index(self.prevWinnerCells[i], self.P.shape, order='F')
                    if colIdx == columnIdx:
                        nForwardPredictionsPerCell[cellIdx] = nForwardPredictionsPerCell[cellIdx] + self.params.nCellPerCol
                                    
            candidateIdx = []
            for i in range(len(nForwardPredictionsPerCell)):
                if nForwardPredictionsPerCell[i]==min(nForwardPredictionsPerCell):
                    candidateIdx.append(i)

            winnerCellIdx = self.resolveTie(candidateIdx)
            winnerCellIdx = winnerCellIdx[0]
        else:
            # in case of inference, return all used winner cells
            candidateIdx = []
            for i in range(len(nForwardPredictionsPerCell)):
                if nForwardPredictionsPerCell[i] > 0:
                    candidateIdx.append(i)

            if candidateIdx == []:
                winnerCellIdx = np.random.randint(0, self.params.nCellPerCol)
            else:
                winnerCellIdx = self.resolveTie(candidateIdx)
                winnerCellIdx = winnerCellIdx[0]

        return winnerCellIdx

    # randomly select one element to break a tie
    def resolveTie(obj, x):
        x_t = np.array(x)
        idx = np.random.choice(x_t, size = 1, replace=True)
        return idx

    # Increase number of predictions for all cells that are predicted from this cell
    def activatePredictions(self, cellIdx, colIdx):
        #print len(self.predictionConnections)
        #print len(self.predictionConnections[0])
        predIdx = self.predictionConnections[cellIdx][colIdx]
        if predIdx != []:
            [row, col] = np.unravel_index(predIdx, self.P.shape, order='F')
            self.P[row, col] = self.P[row, col] + 1

    # Evaluate all connections between the input space and columns
    def computeColumnActivations(self, inputSDR):
        columnActivity = inputSDR[self.FF[:,0:(self.nCols)]].sum(axis=0)
        return columnActivity

    def unique(self,list1): 
        # function to get unique values 
        unique_list = [] 
        # traverse for all elements 
        for x in list1: 
            # check if exists in unique_list or not 
            if x not in unique_list: 
                unique_list.append(x) 

        return unique_list