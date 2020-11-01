import numpy as np
import random

x = np.array([[1,2,3],[3,4,5]], dtype=int)
y = np.array ([5,1,3,4,2,20,50,5], dtype=int)
z = np.array ([6,2,6,2,6,2,60,5], dtype=int)

minColumnActivity = 3
k = 2

idx = np.random.choice(y, size = 1, replace=True)
print idx[0]

v = 0.75
g = 200
r = np.around(v*g)

print r

'''
FF = np.array(([],[]), dtype=int)
print FF.shape[1]

FF = np.zeros((3,1), dtype=int)
print FF

#indices_t = np.concatenate(x.transpose())
FF = x

FF = np.c_[FF, np.zeros((2,3), dtype=int)]
print FF

#indices_t = np.concatenate(x.transpose())
FF[:,3] = np.random.choice(y.transpose(), size = 2, replace=True)

print FF

test = []
columnActivity = y
columnActivity_sorted = np.sort(columnActivity)[::-1]
print columnActivity_sorted

ynew = np.sort(y)[::-1]
indx = np.argsort(y)[::-1]

#activeCols = idx_sorted(columnActivity_sorted>obj.params.minColumnActivity ...
#& columnActivity_sorted>=columnActivity_sorted(min(obj.params.kActiveColumn, length(columnActivity_sorted))));
cond1 = np.greater(ynew, minColumnActivity)
cond2 = np.greater_equal(ynew, ynew[min(k,len(ynew))-1])

activeCols = indx[cond2]
test = []

nForwardPredictionsPerCell = y
nCellPerCol = 32

winnerCellIdx = []
for i in range(len(nForwardPredictionsPerCell)):
    if nForwardPredictionsPerCell[i] > 60:
        winnerCellIdx.append(i)
if winnerCellIdx == []:
    winnerCellIdx = np.random.randint(0, nCellPerCol)

if nCellPerCol != 0:
    print "ok"

print winnerCellIdx
print len(x)
'''