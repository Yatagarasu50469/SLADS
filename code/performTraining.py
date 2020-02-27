#! /usr/bin/env python3

import os
import numpy as np
from scipy.io import loadmat
from sklearn import linear_model
from scipy import misc
import glob
import sys
import random

from computeOrupdateERD import FindNeighbors
from computeOrupdateERD import ComputeRecons
from computeFeatures import computeFeatures
from computeDifference import computeDifference

def performTraining(MeasurementPercentageVector,TrainingDataPath,ImageType,ImageExtension,SizeImage,TrainingInfo,Resolution,WindowSize,c_vec,PercOfRD):
    ImNum = 0
    
    loadPathImage = TrainingDataPath + 'Images' + os.path.sep   
    NumTrainingImages = np.size(glob.glob(loadPathImage + '*' + ImageExtension))
    for image_path in glob.glob(loadPathImage + '*' + ImageExtension):
        if ImageExtension=='.mat':
            ImgDat=loadmat(image_path)
            Img=ImgDat['img']
        else:
            Img = misc.imread(image_path)        
        if SizeImage[0]!=Img.shape[0] or SizeImage[1]!=Img.shape[1]:
            sys.exit('Error!!! The dimensions you entered in "SizeImage" do not match the dimensions of the training images')
        
        if not os.path.exists(TrainingDataPath + 'FeaturesRegressCoeffs'):
            os.makedirs(TrainingDataPath + 'FeaturesRegressCoeffs')

        for m in range(0,np.size(MeasurementPercentageVector)):

            SaveFolder = 'Image_' + str(ImNum+1) + '_Perc_' + str(MeasurementPercentageVector[m])
            SavePath = TrainingDataPath + 'FeaturesRegressCoeffs' + os.path.sep + SaveFolder
            if not os.path.exists(SavePath):
                os.makedirs(SavePath)

            
            Mask = np.zeros((SizeImage[0],SizeImage[1]))
            UnifMatrix = np.random.rand(SizeImage[0],SizeImage[1])
            Mask = UnifMatrix<(MeasurementPercentageVector[m]/100)
            
            MeasuredIdxs = np.transpose(np.where(Mask==1))
            UnMeasuredIdxs = np.transpose(np.where(Mask==0))            
            MeasuredValues = Img[Mask==1]

            NeighborValues,NeighborWeights,NeighborDistances = FindNeighbors(TrainingInfo,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Resolution)
            ReconValues,ReconImage = ComputeRecons(TrainingInfo,NeighborValues,NeighborWeights,SizeImage,UnMeasuredIdxs,MeasuredIdxs,MeasuredValues)
            AllPolyFeatures=computeFeatures(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,SizeImage,NeighborValues,NeighborWeights,NeighborDistances,TrainingInfo,ReconValues,ReconImage,Resolution,ImageType)
            NumRandChoices =  int((100-MeasurementPercentageVector[m])*PercOfRD*SizeImage[1]*SizeImage[0]/(100*100))
            print(NumRandChoices)
            OrderForRD = random.sample(range(0,UnMeasuredIdxs.shape[0]), NumRandChoices) 
            PolyFeatures = AllPolyFeatures[OrderForRD,:]
            RDPP = computeDifference(Img,ReconImage,ImageType)+0
            RDPP.astype(int)
            RDPPWithZeros = np.lib.pad(RDPP,(int(np.floor(WindowSize[0]/2)),int(np.floor(WindowSize[1]/2))),'constant',constant_values=0)
            ImgAsBlocks = im2col(RDPPWithZeros,WindowSize)
            MaskVect = np.ravel(Mask)
            ImgAsBlocksOnlyUnmeasured = ImgAsBlocks[:,np.logical_not(MaskVect)]
            temp = np.zeros((WindowSize[0]*WindowSize[1],NumRandChoices))
            for c in c_vec:
                sigma = NeighborDistances[:,0]/c
                cnt = 0;
                for l in OrderForRD:
                    Filter = generateGaussianKernel(sigma[l],WindowSize)
                    temp[:,cnt] = ImgAsBlocksOnlyUnmeasured[:,l]*Filter
                    cnt=cnt+1
                RD = np.sum(temp, axis=0)
                SavePath_c = SavePath + os.path.sep + 'c_' + str(c)

                if not os.path.exists(SavePath_c):
                    os.makedirs(SavePath_c)
                
                np.save(SavePath_c + os.path.sep + 'RD', RD)        
                np.save(SavePath_c + os.path.sep + 'OrderForRD', OrderForRD)   
            np.save(SavePath + os.path.sep + 'Mask', Mask)   
            np.save(SavePath + os.path.sep + 'ReconImage', ReconImage)
            np.save(SavePath + os.path.sep + 'PolyFeatures', PolyFeatures)
        if ImNum == 0:
            print('Feature Extraction Complete for ' + str(ImNum+1) + ' Image' )
        else:
            print('Feature Extraction Complete for ' + str(ImNum+1) + ' Images' )
        ImNum = ImNum + 1
        
    try:
        Img
    except NameError:
        sys.exit('Error!!! There are no images in ' + loadPathImage + ' that have the extention ' + ImageExtension)
        
    for c in c_vec:
        FirstLoop = 1
        for ImNum in range(0,NumTrainingImages):
            for m in range(0,np.size(MeasurementPercentageVector)):
                
                LoadFolder = 'Image_' + str(ImNum+1) + '_Perc_' + str(MeasurementPercentageVector[m])
                LoadPath = TrainingDataPath + 'FeaturesRegressCoeffs' + os.path.sep + LoadFolder
                PolyFeatures = np.load(LoadPath + os.path.sep + 'PolyFeatures.npy')
                LoadPath_c = LoadPath + os.path.sep + 'c_' + str(c)
                RD = np.load(LoadPath_c + os.path.sep + 'RD.npy')
                if ImageType=='D':
                    if FirstLoop==1:
                        BigPolyFeatures = np.column_stack((PolyFeatures[:,0:25],PolyFeatures[:,26]))
                        BigRD = RD
                        FirstLoop = 0                  
                    else:
                        TempPolyFeatures = np.column_stack((PolyFeatures[:,0:25],PolyFeatures[:,26]))                    
                        BigPolyFeatures = np.row_stack((BigPolyFeatures,TempPolyFeatures))
                        BigRD = np.append(BigRD,RD)
                else:
                    if FirstLoop==1:
                        BigPolyFeatures = PolyFeatures
                        BigRD = RD
                        FirstLoop = 0                  
                    else:
                        TempPolyFeatures = PolyFeatures               
                        BigPolyFeatures = np.row_stack((BigPolyFeatures,TempPolyFeatures))
                        BigRD = np.append(BigRD,RD)                    
                    
                       
        regr = linear_model.LinearRegression()
        regr.fit(BigPolyFeatures, BigRD)
        Theta = np.zeros((PolyFeatures.shape[1]))    
        if ImageType=='D':            
            Theta[0:24]=regr.coef_[0:24]
            Theta[26]=regr.coef_[25]
        else:
            Theta = regr.coef_
        SavePath_c = TrainingDataPath + os.path.sep + 'c_' + str(c)
        del BigRD,BigPolyFeatures

        if not os.path.exists(SavePath_c):
            os.makedirs(SavePath_c) 
        np.save(SavePath_c + os.path.sep + 'Theta', Theta)
        print("Regressions Complete for c = " + str(c))
            
def im2col(Matrix,WidowSize):
    M,N = Matrix.shape
    col_extent = N - WidowSize[1] + 1
    row_extent = M - WidowSize[0] + 1
    start_idx = np.arange(WidowSize[0])[:,None]*N + np.arange(WidowSize[1])
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
    out = np.take (Matrix,start_idx.ravel()[:,None] + offset_idx.ravel())

    return(out)
    # http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python

def generateGaussianKernel(sigma,WindowSize):
    FilterMat = np.ones((WindowSize[0],WindowSize[1]))
    for i in range(0,WindowSize[0]):
        for j in range(0,WindowSize[1]):
            FilterMat[i][j]=np.exp( -(1/(2*sigma**2)) * np.absolute( ( (i-np.floor(WindowSize[0]/2))**2 +  (j-np.floor(WindowSize[1]/2))**2 ) )  )
    FilterMat = FilterMat/np.amax(FilterMat)
    FilterMat = np.transpose(FilterMat)
    Filter=np.ravel(FilterMat)
    return Filter

