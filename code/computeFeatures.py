#! /usr/bin/env python3
import numpy as np
import numpy.matlib as matlib
from computeDifference import computeDifference


def computeFeatures(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,SizeImage,NeighborValues,NeighborWeights,NeighborDistances,TrainingInfo,ReconValues,ReconImage,Resolution,ImageType):
    Feature=np.zeros((np.shape(UnMeasuredIdxs)[0],6))

    # Compute st div features
    Feature[:,0],Feature[:,1]=computeStDivFeatures(NeighborValues,NeighborWeights,TrainingInfo,ReconValues,ImageType)
    
    # Compute distance/density features
    Feature[:,2],Feature[:,3]=computeDensityDistanceFeatures(NeighborDistances,NeighborWeights,SizeImage,TrainingInfo,ReconValues,ImageType)

    # Compute gradient features
    GradientImageX,GradientImageY=computeGradientFeatures(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,SizeImage,TrainingInfo,ReconImage,ImageType)
    Feature[:,4] = GradientImageY[UnMeasuredIdxs[:,0],UnMeasuredIdxs[:,1]]
    Feature[:,5] = GradientImageX[UnMeasuredIdxs[:,0],UnMeasuredIdxs[:,1]]


    PolyFeatures = computePolyFeatures(Feature)
    return PolyFeatures

def computeGradientFeatures(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,SizeImage,TrainingInfo,ReconImage,ImageType):
    GradientImageX,GradientImageY = np.gradient(ReconImage)
    if ImageType=='D':
        GradientImageX[GradientImageX!=0]=1
        GradientImageY[GradientImageY!=0]=1
    elif ImageType=='C':
        GradientImageX=abs(GradientImageX)
        GradientImageY=abs(GradientImageY)
    return(GradientImageX,GradientImageY)


def computeStDivFeatures(NeighborValues,NeighborWeights,TrainingInfo,ReconValues,ImageType):
    
    DiffVect = computeDifference(NeighborValues,np.transpose(matlib.repmat(ReconValues,np.shape(NeighborValues)[1],1)),ImageType)
    Feature_0 = np.sum(NeighborWeights*DiffVect,axis=1)
    Feature_1 = np.sqrt((1/TrainingInfo.NumNbrs)*np.sum(np.power(DiffVect,2),axis=1))
    return(Feature_0,Feature_1)


def computeDensityDistanceFeatures(NeighborDistances,NeighborWeights,SizeImage,TrainingInfo,ReconValues,ImageType):
    
    CutoffDist = np.ceil(np.sqrt((TrainingInfo.FeatDistCutoff/100)*(SizeImage[0]*SizeImage[1]/np.pi)))
    Feature_2 = NeighborDistances[:,0]
    NeighborsInCircle=np.sum(NeighborDistances<=CutoffDist,axis=1)
    Feature_3 = (1+(np.pi*(np.power(CutoffDist,2))))/(1+NeighborsInCircle)
    return(Feature_2,Feature_3)

def computePolyFeatures(Feature):
    
    PolyFeatures = np.hstack([np.ones((np.shape(Feature)[0],1)),Feature])
    for i in range(0,np.shape(Feature)[1]):
        for j in range(i,np.shape(Feature)[1]):
            Temp = Feature[:,i]*Feature[:,j]
            PolyFeatures = np.column_stack([PolyFeatures,Feature[:,i]*Feature[:,j]])

    return PolyFeatures

