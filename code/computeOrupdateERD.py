#! /usr/bin/env python3
import numpy as np
from sklearn.neighbors import NearestNeighbors
from computeNeighborWeights import computeNeighborWeights
from computeWeightedMRecons import computeWeightedMRecons
from computeFeatures import computeFeatures

def computeFullERD(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType):

    NeighborValues,NeighborWeights,NeighborDistances = FindNeighbors(TrainingInfo,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Resolution)

    ReconValues,ReconImage = ComputeRecons(TrainingInfo,NeighborValues,NeighborWeights,SizeImage,UnMeasuredIdxs,MeasuredIdxs,MeasuredValues)
    
    # Compute features
    PolyFeatures=computeFeatures(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,SizeImage,NeighborValues,NeighborWeights,NeighborDistances,TrainingInfo,ReconValues,ReconImage,Resolution,ImageType)
    
    # Compute ERD
    ERDValues = PolyFeatures.dot(Theta)
    
    return(ERDValues,ReconValues,ReconImage)

def updateERD(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NewIdxs,NumSamples,UpdateERDParams,ReconValues,ReconImage,ERDValues,MaxIdxsVect,BatchSamplingParams):

    ERDValues=np.delete(ERDValues,(MaxIdxsVect))
    ReconValues=np.delete(ReconValues,(MaxIdxsVect))
    SuggestedRadius = int(np.sqrt((1/np.pi)*(SizeImage[0]*SizeImage[1]*TrainingInfo.NumNbrs/NumSamples)))
    UpdateRadiusTemp=np.max([SuggestedRadius,UpdateERDParams.MinRadius]);
    UpdateRadius=int(np.min([UpdateERDParams.MaxRadius,UpdateRadiusTemp]));

    updateRadiusMat = np.zeros((SizeImage[0],SizeImage[1]))
    Done=0
    while(Done==0):
        if BatchSamplingParams.Do == 'N':
            updateRadiusMat[max(NewIdxs[0]-UpdateRadius,0):min(NewIdxs[0]+UpdateRadius,SizeImage[0])][:,max(NewIdxs[1]-UpdateRadius,0):min(NewIdxs[1]+UpdateRadius,SizeImage[1])]=1
        else:
            for b in range(0,BatchSamplingParams.NumSamplesPerIter):
                updateRadiusMat[max(NewIdxs[b][0]-UpdateRadius,0):min(NewIdxs[b][0]+UpdateRadius,SizeImage[0])][:,max(NewIdxs[b][1]-UpdateRadius,0):min(NewIdxs[b][1]+UpdateRadius,SizeImage[1])]=1
    
        updateIdxs = np.where(updateRadiusMat[Mask==0]==1)
        
        SmallUnMeasuredIdxs = np.transpose(np.where(np.logical_and(Mask==0,updateRadiusMat==1)))
        if SmallUnMeasuredIdxs.size==0:
            UpdateRadius=int(UpdateRadius*UpdateERDParams.IncreaseRadiusBy)
        else:
            Done=1

    
    # Find neighbors of unmeasured locations
    SmallNeighborValues,SmallNeighborWeights,SmallNeighborDistances = FindNeighbors(TrainingInfo,MeasuredIdxs,SmallUnMeasuredIdxs,MeasuredValues,Resolution)
    
    # Perform reconstruction
    SmallReconValues=computeWeightedMRecons(SmallNeighborValues,SmallNeighborWeights,TrainingInfo)
    
    ReconImage[(np.logical_and(Mask==0,updateRadiusMat==1))]=SmallReconValues
    ReconImage[MeasuredIdxs[:,0],MeasuredIdxs[:,1]]=MeasuredValues

    # Compute features
    SmallPolyFeatures=computeFeatures(MeasuredValues,MeasuredIdxs,SmallUnMeasuredIdxs,SizeImage,SmallNeighborValues,SmallNeighborWeights,SmallNeighborDistances,TrainingInfo,SmallReconValues,ReconImage,Resolution,ImageType)

    # Compute ERD
    SmallERDValues = SmallPolyFeatures.dot(Theta)

    ReconValues[updateIdxs] = SmallReconValues
    ERDValues[updateIdxs] = SmallERDValues

    return(ERDValues,ReconValues)


def FindNeighbors(TrainingInfo,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Resolution):

    # Find neighbors of unmeasured locations
    Neigh = NearestNeighbors(n_neighbors=TrainingInfo.NumNbrs)
    Neigh.fit(MeasuredIdxs)
    NeighborDistances, NeighborIndices = Neigh.kneighbors(UnMeasuredIdxs)
    NeighborDistances=NeighborDistances*Resolution
    NeighborValues=MeasuredValues[NeighborIndices]
    NeighborWeights=computeNeighborWeights(NeighborDistances,TrainingInfo)
    
    return(NeighborValues,NeighborWeights,NeighborDistances)

def ComputeRecons(TrainingInfo,NeighborValues,NeighborWeights,SizeImage,UnMeasuredIdxs,MeasuredIdxs,MeasuredValues):
    
    # Perform reconstruction
    ReconValues=computeWeightedMRecons(NeighborValues,NeighborWeights,TrainingInfo)
    ReconImage = np.zeros((SizeImage[0],SizeImage[1]))
    ReconImage[UnMeasuredIdxs[:,0],UnMeasuredIdxs[:,1]]=ReconValues
    ReconImage[MeasuredIdxs[:,0],MeasuredIdxs[:,1]]=MeasuredValues

    return(ReconValues,ReconImage)
