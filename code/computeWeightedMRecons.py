#! /usr/bin/env python3
import numpy as np

def computeWeightedMRecons(NeighborValues,NeighborWeights,TrainingInfo):
    
    # Weighted Mode Computation
    if TrainingInfo.FeatReconMethod=='DWM':
        ClassLabels = np.unique(NeighborValues)
        ClassWeightSums = np.zeros((np.shape(NeighborWeights)[0],np.shape(ClassLabels)[0]))
        for i in range(0,np.shape(ClassLabels)[0]):
            TempFeats=np.zeros((np.shape(NeighborWeights)[0],np.shape(NeighborWeights)[1]))
            np.copyto(TempFeats,NeighborWeights)
            TempFeats[NeighborValues!=ClassLabels[i]]=0
            ClassWeightSums[:,i]=np.sum(TempFeats,axis=1)
        IdxOfMaxClass = np.argmax(ClassWeightSums,axis=1)
        ReconValues = ClassLabels[IdxOfMaxClass]

    # Weighted Mean Computation
    elif TrainingInfo.FeatReconMethod=='CWM':
        ReconValues=np.sum(NeighborValues*NeighborWeights,axis=1)

    return ReconValues
