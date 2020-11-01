import time
import sys
import os, os.path

import numpy as np
from numpy import linalg as LA

import random
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from nupic.algorithms.temporal_memory import TemporalMemory

import torch
import torchvision
import torch.nn as nn

from torchvision import transforms
from torchvision import models
from PIL import Image

from miniColumn2 import MCN

class Params:
    pass

def main():

    # Load CNN
    original_model = models.alexnet(pretrained=True)
    class AlexNetConv3(nn.Module):
                def __init__(self):
                    super(AlexNetConv3, self).__init__()
                    self.features = nn.Sequential(
                        # stop at conv3
                        *list(original_model.features.children())[:7]
                    )
                def forward(self, x):
                    x = self.features(x)
                    return x

    model = AlexNetConv3()
    model.eval()

    tm = TemporalMemory(
        # Must be the same dimensions as the SP
        columnDimensions=(2048,),
        # How many cells in each mini-column.
        cellsPerColumn=32,
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

    numberImages = 200
    features = []
    labels = []

    DIR = "/home/cappizzino/Documentos/doutorado/simulation/paper_01/dataset"

    path_im = [os.path.join(DIR,sp) for sp in [
        'fall/',
        'spring/',
        'summer/',
        'winter/']]

    # Seasons to compare.
    # First season is the input one. Second season is the reference season.
    # 0 = fall, 1 = spring, 2 = summer, 3 = winter.
    # simul 1 = 2 and 3
    # simul 2 = 1 and 0
    # simul 3 = 0 and 3
    reference_season = 1
    input_season = 3

    # Extract Features
    reference_features, reference_labels = extractFeatures(numberImages, reference_season, model,path_im)
    input_features, input_labels = extractFeatures(numberImages, input_season, model, path_im)

    #print len(input_features[0])
    #print input_labels[0]
    #print input_features

    # Experiments
    # Ground truth
    print 'Ground truth'
    GT = np.identity(numberImages, dtype = bool)
    for i in range(GT.shape[0]):
        for j in range(GT.shape[0]-1):
            if i==j:
                GT[i,j]=1

    # Pairwise (raw descriptors)
    print 'Pairwise descriptors'
    t = time.time()
    S_pairwise = cosine_similarity(reference_features[:numberImages], input_features[:numberImages])
    elapsed = time.time() - t
    print("Elapsed time: %f seconds\n" %elapsed)              

    # Dimension Reduction and binarizarion
    print 'Dimension Reduction'
    P = np.random.randn(len(input_features[0]), 1024)
    P = normc(P)

    # sLSBH (binarized descriptors)
    print 'sLSBH'
    t = time.time()
    D1_slsbh = getLSBH(reference_features[:numberImages],P,0.25)
    D2_slsbh = getLSBH(input_features[:numberImages],P,0.25)
    Sb_pairwise = pairwiseDescriptors(D1_slsbh[:numberImages], D2_slsbh[:numberImages])
    elapsed = time.time() - t
    print("Elapsed time: %f seconds\n" %elapsed)   
    #print len(np.nonzero(D1_slsbh[0])[0])

    D1_tm=[]
    D2_tm=[]
    id_max=[]
    id_max1=[]
    id_max2=[]

    print 'Temporal Pooler (1) descriptors'
    t = time.time()
    for i in range(numberImages):
        for _ in range(1):
            activeColumnIndices = np.nonzero(D1_slsbh[i,:])[0]
            tm.compute(activeColumnIndices, learn=True)
            activeCells = tm.getWinnerCells()
            D1_tm.append(activeCells)
            id_max1.append(max(activeCells))

    print 'Temporal Pooler (2) descriptors'
    for i in range(numberImages):
        activeColumnIndices = np.nonzero(D2_slsbh[i,:])[0]
        tm.compute(activeColumnIndices, learn=False)
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
    elapsed = time.time() - t
    print("Elapsed time: %f seconds\n" %elapsed)

    D1_mcn=[]
    D2_mcn=[]
    id_max=[]
    id_max1=[]
    id_max2=[]

    # Simple HTM parameters
    params = Params()
    params.probAdditionalCon = 0.05    # probability for random connection
    params.nCellPerCol = 32            # number of cells per minicolumn
    params.nInConPerCol = 200          # number of connections per minicolumn
    params.minColumnActivity = 0.75    # minicolumn activation threshold
    params.nColsPerPattern = 50        # minimum number of active minicolumns k_min
    params.kActiveColumn = 100         # maximum number of active minicolumns k_max

    # conversion of the parameter to a natural number that contains the
    # required number of 1s for activation
    params.minColumnActivity = np.round(params.minColumnActivity*params.nInConPerCol)

    htm = MCN('htm',params)

    nCols_MCN=[]
    nCols_HTM=[]

    print ('Simple HTM (1)')
    t = time.time()
    for i in range(numberImages):
        htm.compute(D1_slsbh[i,:],0)
        nCols_MCN.append(htm.nCols)
        nCols_HTM.append(tm.columnDimensions[0])
        id_max1.append(max(htm.winnerCells))
        D1_mcn.append(htm.winnerCells)

    print ('Simple HTM (2)')
    for i in range(numberImages):
        htm.compute(D2_slsbh[i,:],1)
        #nCols_MCN.append(htm.nCols)
        #nCols_HTM.append(tm.columnDimensions[0])
        id_max2.append(max(htm.winnerCells))
        D2_mcn.append(htm.winnerCells)

    id_max = max(max(id_max1),max(id_max2))

    D1_sparse = sparse.lil_matrix((numberImages, id_max+1), dtype='int8')
    for i in range(numberImages):
        D1_sparse[i,D1_mcn[i]] = 1

    D2_sparse = sparse.lil_matrix((numberImages, id_max+1), dtype='int8')
    for i in range(numberImages):
        D2_sparse[i,D2_mcn[i]] = 1

    S_MCN = pairwiseDescriptors(D1_sparse, D2_sparse)
    elapsed = time.time() - t
    print("Elapsed time: %f seconds\n" %elapsed)

    # Results
    print 'Results 1'
    fig, ax = plt.subplots()

    P, R = createPR(S_pairwise,GT)
    ax.plot(R, P, label='pairwise / raw (avgP=%f)' %np.trapz(P,R))

    P, R = createPR(S_MCN,GT)
    ax.plot(R, P, label='MCN (avgP=%f)' %np.trapz(P,R))

    P, R = createPR(Sb_pairwise,GT)
    ax.plot(R, P, label='sLSBH / raw (avgP=%f)' %np.trapz(P,R))

    P, R = createPR(S_TM,GT)
    ax.plot(R, P, label='HTM TM (avgP=%f)' %np.trapz(P,R))

    ax.legend()
    ax.grid(True)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    print 'Results 2'
    fig2, ax2 = plt.subplots()

    ax2.plot(nCols_MCN,'g',label='MCN = %i cols' %htm.nCols)
    ax2.plot(nCols_HTM,'b',label='HTM TM = %i cols' %tm.columnDimensions[0])

    ax2.legend()
    ax2.grid(True)
    plt.xlabel('Number of seen images')
    plt.ylabel('Number of MiniColumns')
    plt.show()

    print 'Results 3'
    fig3, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(9,4))

    P, R = createPR(S_pairwise,GT)
    ax3.plot(R, P, label='pairwise / raw (AUC=%f)' %np.trapz(P,R))

    P, R = createPR(S_MCN,GT)
    ax3.plot(R, P, label='MCN (AUC=%f)' %np.trapz(P,R))

    P, R = createPR(Sb_pairwise,GT)
    ax3.plot(R, P, label='sLSBH / raw (AUC=%f)' %np.trapz(P,R))

    P, R = createPR(S_TM,GT)
    ax3.plot(R, P, label='HTM TM (AUC=%f)' %np.trapz(P,R))

    ax3.grid(True)

    ax3.set_xlabel("Recall", fontsize = 12.0)
    ax3.set_ylabel("Precision", fontsize = 12.0)
    ax3.legend(fontsize=10)

    ax4.plot(nCols_MCN,'g',label='MCN = %i cols' %htm.nCols)
    ax4.plot(nCols_HTM,'b',label='HTM TM = %i cols' %tm.columnDimensions[0])

    ax4.grid(True)

    ax4.tick_params(axis='both', labelsize=6)
    ax4.set_xlabel('Number of seen images', fontsize = 12.0)
    ax4.set_ylabel('Number of MiniColumns', fontsize = 12.0)
    ax4.legend(fontsize=10)

    fig3.savefig('tes.eps')
    plt.show()

# Extract Features
def extractFeatures(n_images, season, net,path_im):
    features = []
    labels = []
    count = 0

    while count < n_images:
        # load image
        image, label = loadImage(count,season,path_im)
        #image.show()

        # preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0) 

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            net.to('cuda')

        with torch.no_grad():
            output = net(input_batch)

        outputFlatten = torch.flatten(output[0])
        outputFlatten = (outputFlatten.data).numpy()
        outputFlatten = outputFlatten / np.linalg.norm(outputFlatten)

        features.append(outputFlatten)
        labels.append(count)
        count = count + 1

    return np.asarray(features), np.asarray(labels)

# Given an index and a season, this method loads an image
def loadImage (image_index, season ,path_im):
    path = path_im[season]+str(image_index)+'.png'
    label = image_index
    image = Image.open(path)

    return image, label

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