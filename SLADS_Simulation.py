#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#==================================================================
#Program: SLADS_TensorFlow_Simulation
#Author(s): David Helminiak
#Date Created: 13 February 2019
#Date Last Modified: March 2019
#Changelog: 0.1 - Combined Structure            - February 2019
#           0.2 - Combined Train/Test           - February 2019
#           0.3 - Gaussian CPU Multi-Threading  - February 2019
#           0.4 - Restructuring of dir vars     - March 2019
#           0.5 - Clearer code progress viz     - March 2019
#           0.6 - Plotting Statistics           - March 2019
#           0.7 - Line Scanning
#           0.8 - .RAW usage
#           0.9 - Continuous value prediction
#==================================================================
#==================================================================

#==================================================================
#ADDITIONAL NOTES:
#==================================================================
#Add Breakpoint anywhere in the program: 
#from IPython.core.debugger import Tracer; Tracer()() 
#==================================================================
#==================================================================

#==================================================================
#LIBRARY IMPORTS
#==================================================================
from __future__ import absolute_import, division, print_function
#import tensorflow as tf
#tf.enable_eager_execution() #Evaluate all operations without building graphs
import pandas as pd
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import multiprocessing
import os
import PIL
import math
import glob
import re
import random
import sys
import scipy
import shutil
import time
import warnings
warnings.filterwarnings("ignore")
from IPython import display
from joblib import Parallel, delayed
from matplotlib.pyplot import figure
from PIL import Image
from scipy import misc
from scipy.io import loadmat
from scipy.io import savemat
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors
from skimage.measure import compare_ssim
from tqdm.auto import tqdm
#from tensorflow import keras
#==================================================================


# In[ ]:


#==================================================================
#CLASS AND FUNCTION DEFINITIONS
#==================================================================
def plotErrorData(savePlotLocation, savePlotSuffix, trainingResultsAverageObject, xAxisValues, plotTitle, plotXLabel): #Plot and save the error data obtained during training

    #Format the base plot
    font = {'size' : 18}
    plt.rc('font', **font)

    #Add plot
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot()

    #Add error plots as training data is recorded
    if (len(trainingResultsAverageObject.mseAverageErrors) > 0):
        plt.plot(xAxisValues, trainingResultsAverageObject.mseAverageErrors, color='black', linestyle='--') 
        plt.scatter(xAxisValues, trainingResultsAverageObject.mseAverageErrors, color='black') 

    #Label plot
    plt.title(plotTitle)
    plt.xlabel(plotXLabel)
    plt.ylabel('Average MSE')

    #Export image
    plt.savefig(savePlotLocation + 'MSE' + savePlotSuffix + '.png')

    #Add plot
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot()

    #Add error plots as training data is recorded
    if (len(trainingResultsAverageObject.ssimAverageErrors) > 0):
        plt.plot(xAxisValues, trainingResultsAverageObject.ssimAverageErrors, color='black', linestyle='--') 
        plt.scatter(xAxisValues, trainingResultsAverageObject.ssimAverageErrors, color='black') 

    #Label plot
    plt.title(plotTitle)
    plt.xlabel(plotXLabel)
    plt.ylabel('Average SSIM')

    #Export image
    plt.savefig(savePlotLocation + 'SSIM' + savePlotSuffix + '.png')

    #Add plot
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot()

    #Add error plots as training data is recorded
    if (len(trainingResultsAverageObject.distortionAverageErrors) > 0):
        plt.plot(xAxisValues, trainingResultsAverageObject.distortionAverageErrors, color='black', linestyle='--') 
        plt.scatter(xAxisValues, trainingResultsAverageObject.distortionAverageErrors, color='black') 

    #Label plot
    plt.title(plotTitle)
    plt.xlabel(plotXLabel)
    plt.ylabel('Average % Total Distortion')

    #Export image
    plt.savefig(savePlotLocation + 'TotalDistortion' + savePlotSuffix + '.png')

def ttPlotAverageErrors(savePlotLocation, StopPercentageSLADSArr, StopPercentageTestingSLADSArr, trainTestAverageErrors):

    ttMSE = []
    ttSSIM = []
    ttTD = []

    for i in range(0, len(trainTestAverageErrors)):
        ttMSE.append(trainTestAverageErrors[i].mseAverageErrors)
        ttSSIM.append(trainTestAverageErrors[i].ssimAverageErrors)
        ttTD.append(trainTestAverageErrors[i].distortionAverageErrors)

    xAxisValues = StopPercentageTestingSLADSArr.tolist()
    labels = []
    for i in StopPercentageSLADSArr.tolist(): labels.append(str(i))
    linestyles = ['--', '-.', ':', '-']

    #TD
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot()
    if (len(ttMSE) > 0):
        for y, label, linestyle in zip(ttTD, labels, linestyles):
            plt.plot(xAxisValues, y, color='black', linestyle = linestyle, label = label) 
            plt.scatter(xAxisValues, y, color='black') 
    plt.xlabel('% Sampled')
    plt.ylabel('Average Total Distortion')
    if (len(labels) > 1):
        plt.legend(title='Training Sampling (%)')
    plt.savefig(savePlotLocation + 'ttPlotAverageTD' + '.png')

    #MSE
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot()
    if (len(ttMSE) > 0):
        for y, label, linestyle in zip(ttMSE, labels, linestyles):
            plt.plot(xAxisValues, y, color='black', linestyle = linestyle, label = label) 
            plt.scatter(xAxisValues, y, color='black') 
    plt.xlabel('% Sampled')
    plt.ylabel('Average MSE')
    if (len(labels) > 1):
        plt.legend(title='Training Sampling (%)')
    plt.savefig(savePlotLocation + 'ttPlotAverageMSE' + '.png')
    
    #SSIM
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot()
    if (len(ttMSE) > 0):
        for y, label, linestyle in zip(ttSSIM, labels, linestyles):
            plt.plot(xAxisValues, y, color='black', linestyle = linestyle, label = label) 
            plt.scatter(xAxisValues, y, color='black') 
    plt.xlabel('% Sampled')
    plt.ylabel('Average SSIM')
    if (len(labels) > 1):
        plt.legend(title='Training Sampling (%)')
    plt.savefig(savePlotLocation + 'ttPlotAverageSSIM' + '.png')
    
class simulationResults: #Object to hold local and global training information values for training convergence
    def initialize(self):
        self.mseAverageErrors = []
        self.ssimAverageErrors = []
        self.distortionAverageErrors = []
    def saveErrorData(self, mseValue, ssimValue, distortValue): #Store training error information
        self.mseError = mseValue
        self.ssimError = ssimValue
        self.totalDistortion = distortValue
    def saveAverageErrorData(self, mseAverageValue, ssimAverageValue, distortionAverageValue): #Store training error information
        self.mseAverageErrors.append(mseAverageValue)
        self.ssimAverageErrors.append(ssimAverageValue)
        self.distortionAverageErrors.append(distortionAverageValue)
        
#Storage location for training information definition
class TrainingInfo:
    def initialize(self,ReconMethod,FeatReconMethod, p, NumNbrs, FilterType, FilterC, FeatDistCutoff, MaxWindowForTraining,*args):
        self.ReconMethod = ReconMethod
        self.FeatReconMethod = FeatReconMethod
        self.p = p
        self.NumNbrs = NumNbrs        
        self.FilterType = FilterType
        self.FilterC = FilterC
        self.FeatDistCutoff = FeatDistCutoff
        self.MaxWindowForTraining=MaxWindowForTraining
        if args:
            self.PAP_Iter=args[0]
            self.PAP_Beta=args[1]
            self.PAP_InitType=args[2]
            self.PAP_ScaleMax=args[3]
        
class TestingInfo:
    def initialize(self):
        self.mseTestingError = [] #Storage location for MSE testing error data
        self.ssimTestingError = [] #Storage location for SSIM testing error data
    def saveTestingErrorData(self, mseValue, ssimValue): #Store training error information
        self.mseTestingError.append(mseValue)
        self.ssimTestingError.append(ssimValue)
    def plotTestingErrorData(self, savePlotLocation): #Plot and save the error data obtained during training
        x = 1
        #Export image
        plt.savefig(savePlotLocation + 'testingStatistics.png')       

        
#Storage location for the initial mask definition
class InitialMask:
    def initialize(self,RowSz,ColSz,MaskType,MaskNumber,Percentage):
        self.RowSz = RowSz
        self.ColSz = ColSz
        self.MaskType = MaskType
        self.MaskNumber = MaskNumber
        self.Percentage = Percentage

#Storage location for the training stopping parameters
class StopCondParams:
    def initialize(self,Beta,Threshold,JforGradient,MinPercentage,MaxPercentage):
        self.Beta = Beta
        self.Threshold = Threshold
        self.JforGradient = JforGradient
        self.MinPercentage = MinPercentage
        self.MaxPercentage = MaxPercentage

#Storage location for ERD parameters
class UpdateERDParams:
    def initialize(self,Do,MinRadius,MaxRadius,IncreaseRadiusBy):
        self.Do = Do
        self.MinRadius = MinRadius
        self.MaxRadius = MaxRadius
        self.IncreaseRadiusBy = IncreaseRadiusBy

#Storage location for batch sampling parameters
class BatchSamplingParams:
    def initialize(self,Do,NumSamplesPerIter):
        self.Do = Do
        self.NumSamplesPerIter = NumSamplesPerIter

#Storage object for image data
class imageData:
    def __init__(self, data, name):
        self.name =name
        self.data = data
        
#Convert an image into a series of columns
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

def computeFullERD(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType):

    NeighborValues,NeighborWeights,NeighborDistances = FindNeighbors(TrainingInfoObject,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Resolution)

    ReconValues,ReconImage = ComputeRecons(TrainingInfoObject,NeighborValues,NeighborWeights,SizeImage,UnMeasuredIdxs,MeasuredIdxs,MeasuredValues)
    
    # Compute features
    PolyFeatures=computeFeatures(UnMeasuredIdxs,SizeImage,NeighborValues,NeighborWeights,NeighborDistances,TrainingInfoObject,ReconValues,ReconImage,Resolution,ImageType)
    
    # Compute ERD
    ERDValues = PolyFeatures.dot(Theta)
    
    return(ERDValues,ReconValues,ReconImage)

def updateERD(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,NewIdxs,NumSamples,UpdateERDParamsObject,ReconValues,ReconImage,ERDValues,MaxIdxsVect,BatchSamplingParamsObject):

    ERDValues=np.delete(ERDValues,(MaxIdxsVect))
    ReconValues=np.delete(ReconValues,(MaxIdxsVect))
    SuggestedRadius = int(np.sqrt((1/np.pi)*(SizeImage[0]*SizeImage[1]*TrainingInfoObject.NumNbrs/NumSamples)))
    UpdateRadiusTemp=np.max([SuggestedRadius,UpdateERDParamsObject.MinRadius]);
    UpdateRadius=int(np.min([UpdateERDParamsObject.MaxRadius,UpdateRadiusTemp]));

    updateRadiusMat = np.zeros((SizeImage[0],SizeImage[1]))
    Done=0
    while(Done==0):
        if BatchSamplingParamsObject.Do == 'N':
            updateRadiusMat[max(NewIdxs[0]-UpdateRadius,0):min(NewIdxs[0]+UpdateRadius,SizeImage[0])][:,max(NewIdxs[1]-UpdateRadius,0):min(NewIdxs[1]+UpdateRadius,SizeImage[1])]=1
        else:
            for b in range(0,BatchSamplingParamsObject.NumSamplesPerIter):
                updateRadiusMat[max(NewIdxs[b][0]-UpdateRadius,0):min(NewIdxs[b][0]+UpdateRadius,SizeImage[0])][:,max(NewIdxs[b][1]-UpdateRadius,0):min(NewIdxs[b][1]+UpdateRadius,SizeImage[1])]=1
    
        updateIdxs = np.where(updateRadiusMat[Mask==0]==1)
        
        SmallUnMeasuredIdxs = np.transpose(np.where(np.logical_and(Mask==0,updateRadiusMat==1)))
        if SmallUnMeasuredIdxs.size==0:
            UpdateRadius=int(UpdateRadius*UpdateERDParamsObject.IncreaseRadiusBy)
        else:
            Done=1

    
    # Find neighbors of unmeasured locations
    SmallNeighborValues,SmallNeighborWeights,SmallNeighborDistances = FindNeighbors(TrainingInfoObject,MeasuredIdxs,SmallUnMeasuredIdxs,MeasuredValues,Resolution)
    
    # Perform reconstruction
    SmallReconValues=computeWeightedMRecons(SmallNeighborValues,SmallNeighborWeights,TrainingInfoObject)
    
    ReconImage[(np.logical_and(Mask==0,updateRadiusMat==1))]=SmallReconValues
    ReconImage[MeasuredIdxs[:,0],MeasuredIdxs[:,1]]=MeasuredValues

    # Compute features
    SmallPolyFeatures=computeFeatures(SmallUnMeasuredIdxs,SizeImage,SmallNeighborValues,SmallNeighborWeights,SmallNeighborDistances,TrainingInfoObject,SmallReconValues,ReconImage,Resolution,ImageType)

    # Compute ERD
    SmallERDValues = SmallPolyFeatures.dot(Theta)

    ReconValues[updateIdxs] = SmallReconValues
    ERDValues[updateIdxs] = SmallERDValues

    return(ERDValues,ReconValues)


def FindNeighbors(TrainingInfoObject,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Resolution):

    # Find neighbors of unmeasured locations
    Neigh = NearestNeighbors(n_neighbors=TrainingInfoObject.NumNbrs)
    Neigh.fit(MeasuredIdxs)
    NeighborDistances, NeighborIndices = Neigh.kneighbors(UnMeasuredIdxs)
    NeighborDistances=NeighborDistances*Resolution
    NeighborValues=MeasuredValues[NeighborIndices]
    NeighborWeights=computeNeighborWeights(NeighborDistances,TrainingInfoObject)
    
    return(NeighborValues,NeighborWeights,NeighborDistances) 

def ComputeRecons(TrainingInfoObject,NeighborValues,NeighborWeights,SizeImage,UnMeasuredIdxs,MeasuredIdxs,MeasuredValues):
    
    # Perform reconstruction
    ReconValues=computeWeightedMRecons(NeighborValues,NeighborWeights,TrainingInfoObject)
    ReconImage = np.zeros((SizeImage[0],SizeImage[1]))
    ReconImage[UnMeasuredIdxs[:,0],UnMeasuredIdxs[:,1]]=ReconValues
    ReconImage[MeasuredIdxs[:,0],MeasuredIdxs[:,1]]=MeasuredValues
    return(ReconValues,ReconImage)


def computeNeighborWeights(NeighborDistances,TrainingInfoObject):
    
    UnNormNeighborWeights=1/np.power(NeighborDistances,TrainingInfoObject.p)
    SumOverRow = (np.sum(UnNormNeighborWeights,axis=1))
    NeighborWeights=UnNormNeighborWeights/SumOverRow[:, np.newaxis]
    return NeighborWeights

def computeWeightedMRecons(NeighborValues,NeighborWeights,TrainingInfoObject):
    
    # Weighted Mode Computation
    if TrainingInfoObject.FeatReconMethod=='DWM':
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
    elif TrainingInfoObject.FeatReconMethod=='CWM':
        ReconValues=np.sum(NeighborValues*NeighborWeights,axis=1)

    return ReconValues


def computeDifference(array1,array2,type):
    if type == 'D':
        difference=array1!=array2
        difference = difference.astype(float)
    if type == 'C':
        difference=abs(array1-array2)


    return difference

def computeFeatures(UnMeasuredIdxs,SizeImage,NeighborValues,NeighborWeights,NeighborDistances,TrainingInfoObject,ReconValues,ReconImage,Resolution,ImageType):
    Feature=np.zeros((np.shape(UnMeasuredIdxs)[0],6))

    # Compute st div features
    Feature[:,0],Feature[:,1]=computeStDivFeatures(NeighborValues,NeighborWeights,TrainingInfoObject,ReconValues,ImageType)
    
    # Compute distance/density features
    Feature[:,2],Feature[:,3]=computeDensityDistanceFeatures(NeighborDistances,NeighborWeights,SizeImage,TrainingInfoObject,ReconValues,ImageType)

    # Compute gradient features
    GradientImageX,GradientImageY=computeGradientFeatures(ReconImage,ImageType)
    Feature[:,4] = GradientImageY[UnMeasuredIdxs[:,0],UnMeasuredIdxs[:,1]]
    Feature[:,5] = GradientImageX[UnMeasuredIdxs[:,0],UnMeasuredIdxs[:,1]]

    PolyFeatures = computePolyFeatures(Feature)
    return PolyFeatures

def computeGradientFeatures(ReconImage,ImageType):
    GradientImageX,GradientImageY = np.gradient(ReconImage)
    if ImageType=='D':
        GradientImageX[GradientImageX!=0]=1
        GradientImageY[GradientImageY!=0]=1
    elif ImageType=='C':
        GradientImageX=abs(GradientImageX)
        GradientImageY=abs(GradientImageY)
    return(GradientImageX,GradientImageY)


def computeStDivFeatures(NeighborValues,NeighborWeights,TrainingInfoObjecto,ReconValues,ImageType):
    DiffVect = computeDifference(NeighborValues,np.transpose(np.matlib.repmat(ReconValues,np.shape(NeighborValues)[1],1)),ImageType)
    Feature_0 = np.sum(NeighborWeights*DiffVect,axis=1)
    Feature_1 = np.sqrt((1/TrainingInfoObject.NumNbrs)*np.sum(np.power(DiffVect,2),axis=1))
    return(Feature_0,Feature_1)

def computeDensityDistanceFeatures(NeighborDistances,NeighborWeights,SizeImage,TrainingInfoObject,ReconValues,ImageType):
    CutoffDist = np.ceil(np.sqrt((TrainingInfoObject.FeatDistCutoff/100)*(SizeImage[0]*SizeImage[1]/np.pi)))
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

def performSLADStoFindC(codePath,trainingDataPath,ImageSet,ImageType,ImageExtension,TrainingInfoObject,SizeImage,StopPercentage,Resolution,c_vec,UpdateERDParamsObject,InitialMaskObject,MaskType,reconPercVector,Classify,directImagePath,consoleRows,cPlot,savePlotLocation):
    sys.path.append('code')
    SimulationRun = 0
    
    # Initialize stopping condition variable
    Beta = computeBeta(SizeImage)
    StopCondParamsObject = StopCondParams()
    StopCondParamsObject.initialize(Beta,0,50,2,StopPercentage)
    
    SavePathSLADS = trainingDataPath + 'SLADSResults' 
    PlotResult = 'N'

    # Batch Sampling
    PercOfSamplesPerIter = 0
    NumSamplesPerIter = int(PercOfSamplesPerIter*SizeImage[0]*SizeImage[1]/100)
    BatchSample = 'N'
    BatchSamplingParamsObject = BatchSamplingParams()
    if BatchSample=='N':
        BatchSamplingParamsObject.initialize(BatchSample,1)
    else:
        BatchSamplingParamsObject.initialize(BatchSample,NumSamplesPerIter)

    if not os.path.exists(SavePathSLADS):
        os.makedirs(SavePathSLADS)
        
    AreaUnderCurve = np.zeros(c_vec.shape[0])
    Idx_c = 0
    
    loadPathImage = trainingDataPath + 'Images' + os.path.sep
    loadPathInitialMask = resultsDataPath + 'InitialSamplingMasks' # Load initial measurement mask
    Mask = loadOrGenerateInitialMask(loadPathInitialMask,MaskType,InitialMaskObject,SizeImage)
    imageNames = glob.glob(loadPathImage + '*' + ImageExtension)
    NumImages = np.size(imageNames)
    AvgTDArr = []
    AvgMSEArr = []
    AvgSSIMArr = []
    for i in tqdm(range(0, len(c_vec)), desc = 'c Values', leave = True): #For each of the proposed c values
        c = c_vec[i]
        LoadPath_c = trainingDataPath + 'c_' + str(c)
        TrainingInfoObject.FilterC=c
        Theta = np.load(LoadPath_c + os.path.sep + 'Theta' + '_StopPerc_' + str(StopPercentageSLADS) + '.npy')
        ImageTDArr = []
        ImageMSEArr = []
        ImageSSIMArr = []
        for ImNum in tqdm(range(0, NumImages), desc = 'Images', leave = True): #For each of the images
            img = misc.imread(imageNames[ImNum])
            SavePathSLADS_c_ImNum = SavePathSLADS +  os.path.sep + 'Image_' + str(ImNum) + '_c_'+ str(c)
            if not os.path.exists(SavePathSLADS_c_ImNum):
                os.makedirs(SavePathSLADS_c_ImNum) 

            SavePath = SavePathSLADS + os.path.sep + 'Image_' + str(ImNum) + '_c_'+ str(c) + os.path.sep
            isRunningParallel = True
            runSLADSSimulationOnce(NumImages,Mask,codePath,ImageSet,SizeImage,StopCondParamsObject,Theta,TrainingInfoObject,TestingInfoObject,Resolution,ImageType,UpdateERDParamsObject,BatchSamplingParamsObject,SavePath,SimulationRun,ImNum,ImageExtension,PlotResult,Classify,directImagePath,falseFlag,isRunningParallel)
            
            MeasuredValuesFull=np.load(SavePath + 'MeasuredValues.npy')
            MeasuredIdxsFull=np.load(SavePath + 'MeasuredIdxs.npy')
            UnMeasuredIdxsFull=np.load(SavePath + 'UnMeasuredIdxs.npy')    
            Difference = np.zeros(reconPercVector.shape[0])
            idx=0
            
            for j in tqdm(range(0, len(reconPercVector)), desc = '% Reconstructed', leave = True): #For each of the reconstruction percentages
                p = reconPercVector[j]
                NumMeasurements = int(p*SizeImage[0]*SizeImage[1]/100)
                MeasuredValues = MeasuredValuesFull[0:NumMeasurements]
                MeasuredIdxs = MeasuredIdxsFull[0:NumMeasurements][:]
                temp1 = MeasuredIdxsFull[NumMeasurements+1:MeasuredValuesFull.shape[0]][:]
                temp2 = UnMeasuredIdxsFull
                UnMeasuredIdxs = np.concatenate((temp1, temp2), axis=0)
                Difference[idx], ReconImage = performReconOnce(SavePath,TrainingInfoObject,Resolution,SizeImage,ImageType,codePath,ImageSet,ImNum,ImageExtension,SimulationRun,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,directImagePath)
                idx = idx+1
            
            TD = Difference/(SizeImage[0]*SizeImage[1])
            MSE = (np.sum((ReconImage.astype("float") - img.astype("float")) ** 2))/(float(ReconImage.shape[0] * ReconImage.shape[1]))
            SSIM = compare_ssim(ReconImage.astype("float"), img.astype("float"))

            np.save(SavePath + 'TD', TD)
            np.save(SavePath + 'MSE', MSE)
            np.save(SavePath + 'SSIM', SSIM)
            
            ImageTDArr.append(TD)
            ImageMSEArr.append(MSE)
            ImageSSIMArr.append(SSIM)
            
            AreaUnderCurve[Idx_c]=AreaUnderCurve[Idx_c]+np.trapz(TD,x=reconPercVector)
            
        AvgTDArr.append(np.mean(ImageTDArr))
        AvgMSEArr.append(np.mean(ImageMSEArr))
        AvgSSIMArr.append(np.mean(ImageSSIMArr))
        
        Idx_c = Idx_c +1
        
    Best_c = c_vec[np.argmin(AreaUnderCurve)]
    if cPlot:
        #TD
        x = c_vec
        font = {'size' : 18}
        plt.rc('font', **font)
        f = plt.figure(figsize=(20,8))
        ax1 = f.add_subplot()
        plt.plot(c_vec, AvgTDArr, color='black', linestyle='--') 
        plt.scatter(c_vec, AvgTDArr, color='black') 
        plt.xlabel('c')
        plt.ylabel('Total Distortion')
        plt.savefig(savePlotLocation + 'cPlotTD.png')
        
        #MSE
        font = {'size' : 18}
        plt.rc('font', **font)
        f = plt.figure(figsize=(20,8))
        ax1 = f.add_subplot()
        plt.plot(c_vec, AvgMSEArr, color='black', linestyle='--') 
        plt.scatter(c_vec, AvgMSEArr, color='black') 
        plt.xlabel('c')
        plt.ylabel('MSE')
        plt.savefig(savePlotLocation + 'cPlotMSE.png')
        
        #SSIM
        font = {'size' : 18}
        plt.rc('font', **font)
        f = plt.figure(figsize=(20,8))
        ax1 = f.add_subplot()
        plt.plot(c_vec, AvgSSIMArr, color='black', linestyle='--') 
        plt.scatter(c_vec, AvgSSIMArr, color='black') 
        plt.xlabel('c')
        plt.ylabel('SSIM')
        plt.savefig(savePlotLocation + 'cPlotSSIM.png')
    
    sys.path.pop() #Hopefully remove the 'sys.path.append('code') flag at the top of this definitionr
    return Best_c, NumImages


def runSLADSSimulationOnce(NumImages,Mask,codePath,ImageSet,SizeImage,StopCondParamsObject,Theta,TrainingInfoObject,TestingInfoObject,Resolution,ImageType,UpdateERDParamsObject,BatchSamplingParamsObject,SavePath,SimulationRun,ImNum,ImageExtension,PlotResult,Classify,directImagePath,errorPlot,isRunningParallel):
    sys.path.append('code')
    MeasuredIdxs = np.transpose(np.where(Mask==1))
    UnMeasuredIdxs = np.transpose(np.where(Mask==0))

    ContinuousMeasuredValues = perfromInitialMeasurements(codePath,ImageSet,ImNum,ImageExtension,Mask,SimulationRun,directImagePath)
    if Classify=='2C':
        Threshold = filters.threshold_otsu(ContinuousMeasuredValues)
        print('Threhold found using the Otsu method for 2 Class classification = ' + str(Threshold))
        MeasuredValues = ContinuousMeasuredValues < Threshold
        MeasuredValues = MeasuredValues+0
#    elif Classify=='MC':
        #### Classification function to output NewValues ##################
        # NewValues is the vector of measured values post classification
    elif Classify=='N':
        MeasuredValues=ContinuousMeasuredValues

    # Perform SLADS
    IterNum=0
    NumSamples = np.shape(MeasuredValues)[0]
    StopCondFuncVal=np.zeros(( int((SizeImage[0]*SizeImage[1])*(StopCondParamsObject.MaxPercentage)/100)+10,2 ))
    pixelCount = SizeImage[0]*SizeImage[1]

    with tqdm(total = 100, desc = '% Sampled', leave = True, disable = isRunningParallel) as pbar:
        while (checkStopCondFuncThreshold(StopCondParamsObject,StopCondFuncVal,NumSamples,IterNum,SizeImage) != 1):
            if IterNum==0:
                Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationFirst(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,NumSamples,UpdateERDParamsObject,BatchSamplingParamsObject)           
            else:
                Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationAfter(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,UpdateERDParamsObject,BatchSamplingParamsObject,StopCondFuncVal,IterNum,NumSamples,NewIdxs,ReconValues,ReconImage,ERDValues,MaxIdxsVect)

            NewContinuousValues = performMeasurements(NewIdxs,codePath,ImageSet,ImNum,ImageExtension,MeasuredIdxs,BatchSamplingParamsObject,SimulationRun,directImagePath)
            ContinuousMeasuredValues = np.hstack((ContinuousMeasuredValues,NewContinuousValues))
            if Classify=='2C':           
                NewValues = NewContinuousValues > Threshold
                NewValues = NewValues+0
    #        elif Classify=='MC':
                #### Classification function to output NewValues ##################
                # NewValues is the vector of measured values post classification            
            elif Classify=='N':
                NewValues=NewContinuousValues    

            Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs = updateMeasurementArrays(NewIdxs,MaxIdxsVect,Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,NewValues,BatchSamplingParamsObject)

            NumSamples = np.shape(MeasuredValues)[0]

            StopCondFuncVal=computeStopCondFuncVal(ReconValues,MeasuredValues,StopCondParamsObject,ImageType,StopCondFuncVal,MaxIdxsVect,NumSamples,IterNum,BatchSamplingParamsObject)

            #if PlotResult=='Y' and np.remainder(NumSamples,round(0.01*SizeImage[0]*SizeImage[1])) ==0:
                #print(str(np.round(NumSamples*100/(SizeImage[0]*SizeImage[1]))) + ' Percent Sampled')
            IterNum += 1
            pbar.n = round((NumSamples/pixelCount)*100)
            pbar.refresh()
        np.save(SavePath + 'MeasuredValues', MeasuredValues)
        np.save(SavePath + 'MeasuredIdxs', MeasuredIdxs)
        np.save(SavePath + 'UnMeasuredIdxs', UnMeasuredIdxs)
        np.save(SavePath + 'StopCondFuncVal',StopCondFuncVal)
        np.save(SavePath + 'ContinuousMeasuredValues',ContinuousMeasuredValues)
        savemat(SavePath + 'MeasuredIdxs.mat',dict(MeasuredIdxs=MeasuredIdxs))
        savemat(SavePath + 'MeasuredValues.mat',dict(MeasuredValues=MeasuredValues))
        savemat(SavePath + 'UnMeasuredIdxs.mat',dict(UnMeasuredIdxs=UnMeasuredIdxs))
        savemat(SavePath + 'StopCondFuncVal.mat',dict(StopCondFuncVal=StopCondFuncVal))
        savemat(SavePath + 'ContinuousMeasuredValues.mat',dict(ContinuousMeasuredValues=ContinuousMeasuredValues))

        if errorPlot or (PlotResult=='Y'):
            percentSampled = np.round(NumSamples*100/(SizeImage[0]*SizeImage[1]))
            Difference,ReconImage = performReconOnce(SavePath,TrainingInfoObject,Resolution,SizeImage,ImageType,codePath,ImageSet,ImNum,ImageExtension,SimulationRun,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,directImagePath)
            TD = Difference/(SizeImage[0]*SizeImage[1])
            if (SimulationRun==1):
                img = misc.imread(directImagePath)
            else: 
                img = loadTestImage(codePath,ImageSet,ImNum,ImageExtension,SimulationRun)  
            #print('')
            #print('')
            #print('######################################')
            #print('Total Distortion = ' + str(TD))

            #Moved out plotting code into only function it is used
            MSE = (np.sum((ReconImage.astype("float") - img.astype("float")) ** 2))/(float(ReconImage.shape[0] * ReconImage.shape[1]))
            SSIM = compare_ssim(ReconImage.astype("float"), img.astype("float"))

            if errorPlot:
                resultObject = simulationResults()
                resultObject.saveErrorData(MSE, SSIM, TD) #Store resulting error information
 
            if PlotResult=='Y': 
                #Set plot formatting
                font = {'size' : 18}
                plt.rc('font', **font)
                f = plt.figure(figsize=(20,8))
                plt.suptitle("MSE: %.2f, SSIM: %.2f, TD: %.2f, Percent Sampled: %.2f" % (MSE, SSIM, TD, percentSampled), fontsize=20, fontweight='bold', y = 0.9)
                ax1 = f.add_subplot(131)
                ax1.imshow(Mask, cmap='gist_heat')
                ax1.set_title('Sampled Mask')
                ax2 = f.add_subplot(132)       
                ax2.imshow(ReconImage, cmap='gist_heat')
                ax2.set_title('Reconstructed Image')
                ax3 = f.add_subplot(133)
                ax3.imshow(img, cmap='gist_heat')
                ax3.set_title('Ground-truth Image')
                plt.savefig(SavePath + '.png')

            #pylab.show()
        sys.path.pop() #Hopefully remove the 'sys.path.append('code') flag at the top of this definition
        if errorPlot:
            return resultObject

def runSLADSOnce(Mask,codePath,SizeImage,StopCondParamsObject,Theta,TrainingInfoObject,Resolution,ImageType,UpdateERDParamsObject,BatchSamplingParamsObject,SavePath,SimulationRun,ImNum,PlotResult,Classify):
    sys.path.append('code')
    MeasuredIdxs = np.transpose(np.where(Mask==1))
    UnMeasuredIdxs = np.transpose(np.where(Mask==0))
    
    ##################################################################
    # CODE HERE
    # Plug in Your Measurement Routine
    # Please use 'MeasuredValues' as output variable
    # ContinuousMeasuredValues = performMeasurements(Mask)
    ##################################################################
    
    if Classify=='2C':
        Threshold = filters.threshold_otsu(ContinuousMeasuredValues)
        print('Threhold found using the Otsu method for 2 Class classification = ' + str(Threshold))
        MeasuredValues = ContinuousMeasuredValues < Threshold
        MeasuredValues = MeasuredValues+0
#    elif Classify=='MC':
        #### Classification function to output NewValues ##################
        # NewValues is the vector of measured values post classification
    elif Classify=='N':
        MeasuredValues=ContinuousMeasuredValues
    
    # Perform SLADS
    IterNum=0
    Stop=0
    NumSamples = np.shape(MeasuredValues)[0]
    StopCondFuncVal=np.zeros(( int((SizeImage[0]*SizeImage[1])*(StopCondParamsObject.MaxPercentage)/100)+10,2 ))
    while Stop !=1:
        
        if IterNum==0:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationFirst(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,NumSamples,UpdateERDParamsObject,BatchSamplingParamsObject)           
        else:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationAfter(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,UpdateERDParamsObject,BatchSamplingParamsObject,StopCondFuncVal,IterNum,NumSamples,NewIdxs,ReconValues,ReconImage,ERDValues,MaxIdxsVect)
        
        ##################################################################
        # CODE HERE
        # Plug in Your Measurement Routine
        # Please use 'NewContValues' as output variable
        # NewContinuousValues = performMeasurements(NewIdxs)
        ##################################################################    
        
        ContinuousMeasuredValues = np.hstack((ContinuousMeasuredValues,NewContinuousValues))
        if Classify=='2C':           
            NewValues = NewContinuousValues > Threshold
            NewValues = NewValues+0
#        elif Classify=='MC':
            #### Classification function to output NewValues ##################
            # NewValues is the vector of measured values post classification    
        elif Classify=='N':
            NewValues=NewContinuousValues    


        Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs = updateMeasurementArrays(NewIdxs,MaxIdxsVect,Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,NewValues,BatchSamplingParamsObject)
    
        NumSamples = np.shape(MeasuredValues)[0]
    
        StopCondFuncVal=computeStopCondFuncVal(ReconValues,MeasuredValues,StopCondParamsObject,ImageType,StopCondFuncVal,MaxIdxsVect,NumSamples,IterNum,BatchSamplingParamsObject)
            
        Stop = checkStopCondFuncThreshold(StopCondParamsObject,StopCondFuncVal,NumSamples,IterNum,SizeImage)
        if PlotResult=='Y' and np.remainder(NumSamples,round(0.01*SizeImage[0]*SizeImage[1])) ==0:
            print(str(np.round(NumSamples*100/(SizeImage[0]*SizeImage[1]))) + ' Percent Sampled')
        IterNum += 1
        
    
    np.save(SavePath + 'MeasuredValues', MeasuredValues)
    np.save(SavePath + 'MeasuredIdxs', MeasuredIdxs)
    np.save(SavePath + 'UnMeasuredIdxs', UnMeasuredIdxs)
    np.save(SavePath + 'StopCondFuncVal',StopCondFuncVal)
    np.save(SavePath + 'ContinuousMeasuredValues',ContinuousMeasuredValues)
    savemat(SavePath + 'MeasuredIdxs.mat',dict(MeasuredIdxs=MeasuredIdxs))
    savemat(SavePath + 'MeasuredValues.mat',dict(MeasuredValues=MeasuredValues))
    savemat(SavePath + 'UnMeasuredIdxs.mat',dict(UnMeasuredIdxs=UnMeasuredIdxs))
    savemat(SavePath + 'StopCondFuncVal.mat',dict(StopCondFuncVal=StopCondFuncVal))
    savemat(SavePath + 'ContinuousMeasuredValues.mat',dict(ContinuousMeasuredValues=ContinuousMeasuredValues))
    
    if PlotResult=='Y': 
        print(str(np.round(NumSamples*100/(SizeImage[0]*SizeImage[1]))) + ' Percent Sampled before stopping')
        plotFigure = plotAfterSLADS(Mask,ReconImage)
        #pylab.show()
    sys.path.pop() #Hopefully remove the 'sys.path.append('code') flag at the top of this definition


def perfromInitialMeasurements(codePath,ImageSet,ImNum,ImageExtension,Mask,SimulationRun,directImagePath):

    if (SimulationRun==1):
        Img = misc.imread(directImagePath)
    else: 
        Img = loadTestImage(codePath,ImageSet,ImNum,ImageExtension,SimulationRun)  
    if Mask.shape[0]!=Img.shape[0] or Mask.shape[1]!=Img.shape[1]:
        sys.exit('Error!!! The dimensions you entered in "SizeImage" do not match the dimensions of the testing image in ./ResultsAndData/TestingImages/TestingImageSet_' + ImageSet)
    MeasuredValues = Img[Mask==1]
    return(MeasuredValues)

def performMeasurements(NewIdxs,codePath,ImageSet,ImNum,ImageExtension,MeasuredIdxs,BatchSamplingParamsObject,SimulationRun,directImagePath):
    if (SimulationRun==1):
        Img=misc.imread(directImagePath)
    else: 
        Img = loadTestImage(codePath,ImageSet,ImNum,ImageExtension,SimulationRun)
    if BatchSamplingParamsObject.Do == 'N':
        NewValues = Img[NewIdxs[0],NewIdxs[1]]
    else:
        NewValues = Img[NewIdxs[:,0],NewIdxs[:,1]]
    return NewValues

def updateMeasurementArrays(NewIdxs,MaxIdxsVect,Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,NewValues,BatchSamplingParamsObject):
    
    if BatchSamplingParamsObject.Do == 'N':
        Mask[NewIdxs[0],NewIdxs[1]]=1
        MeasuredValues = np.hstack((MeasuredValues,NewValues))
        MeasuredIdxs = np.vstack((MeasuredIdxs,[NewIdxs[0],NewIdxs[1]]))
        UnMeasuredIdxs = np.delete(UnMeasuredIdxs,(MaxIdxsVect), axis=0)
    else:
        for i in range(0,BatchSamplingParamsObject.NumSamplesPerIter):
            Mask[NewIdxs[i,0],NewIdxs[i,1]]=1
        MeasuredValues = np.hstack((MeasuredValues,NewValues))
        MeasuredIdxs = np.vstack((MeasuredIdxs,NewIdxs))
        UnMeasuredIdxs = np.delete(UnMeasuredIdxs,(MaxIdxsVect), axis=0)
    return(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs)

def findNewMeasurementIdxs(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,NumSamples,UpdateERDParamsObject,ReconValues,ReconImage,ERDValues,ActualBatchSamplingParams):
    
    if ActualBatchSamplingParams.Do == 'N':
        MaxIdxsVect = np.argmax(ERDValues)
        NewIdxs = UnMeasuredIdxs[MaxIdxsVect,:]
    else:
        NewIdxs = np.zeros((ActualBatchSamplingParams.NumSamplesPerIter,2),dtype=np.int)
        MaxIdxsVect = np.zeros((ActualBatchSamplingParams.NumSamplesPerIter,1),dtype=np.int)
        MaxIdxsVect[0] = np.argmax(ERDValues)
        NewIdxs[0,:] = UnMeasuredIdxs[MaxIdxsVect[0],:]
        
        TempNewIdxs = np.zeros((ActualBatchSamplingParams.NumSamplesPerIter,2),dtype=np.int)
        TempMaxIdxsVect = np.zeros((ActualBatchSamplingParams.NumSamplesPerIter,1),dtype=np.int)
        TempMaxIdxsVect[0] = np.argmax(ERDValues)
        TempNewIdxs[0,:] = UnMeasuredIdxs[MaxIdxsVect[0],:]
        
        TempBatchSamplingParams = BatchSamplingParams()
        TempBatchSamplingParams.initialize('N',1)
        OrigUnMeasuredIdxs=np.zeros((np.shape(UnMeasuredIdxs)))
        np.copyto(OrigUnMeasuredIdxs,UnMeasuredIdxs)                
        for i in range(1,ActualBatchSamplingParams.NumSamplesPerIter):
            NewValues = ReconValues[TempMaxIdxsVect[i-1]]
            Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs = updateMeasurementArrays(TempNewIdxs[i-1,:],TempMaxIdxsVect[i-1],Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,NewValues,TempBatchSamplingParams)
            
            ERDValues,ReconValues=updateERD(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,TempNewIdxs[i-1,:],NumSamples,UpdateERDParamsObject,ReconValues,ReconImage,ERDValues,TempMaxIdxsVect[i-1],TempBatchSamplingParams)
            TempMaxIdxsVect[i] = np.argmax(ERDValues)
            TempNewIdxs[i,:] = UnMeasuredIdxs[TempMaxIdxsVect[i],:]
            NewIdxs[i,:] = TempNewIdxs[i,:]
            MaxIdxsVect[i]=np.where(np.all(OrigUnMeasuredIdxs==TempNewIdxs[i,:],axis=1))
    return(NewIdxs,MaxIdxsVect)

def loadTestImage(codePath,ImageSet,ImNum,ImageExtension,SimulationRun):
    
    if SimulationRun==1:
        sys.exit('ERROR!!! Direct loading of test image has been bypassed')
    else:
        #SEE IF THIS LINE IS EVER USED AS PLOTTING DURING TRAINING IS TURNED OFF
        loadPathImage = trainingDataPath + 'Images' + os.path.sep
        #ImagesToFindC Line
        #loadPathImage = resultsDataPath + 'InputData' + os.path.sep + 'TrainingDB_' + ImageSet + os.path.sep + 'ImagesToFindC' + os.path.sep     
    cnt = 0

    for image_path in glob.glob(loadPathImage + '*' + ImageExtension):
        if cnt == ImNum:
            if ImageExtension=='.mat':
                ImgDat=loadmat(image_path)
                Img=ImgDat['img']
            else:
                Img = misc.imread(image_path)
        cnt = cnt+1
    try:
        Img
    except NameError:
        sys.exit('Error!!! There are no images in ' + loadPathImage + ' that have the extention ' + ImageExtension)
    return Img

def loadOrGenerateInitialMask(loadPathInitialMask,MaskType,InitialMaskObject,SizeImage):
    if MaskType=='H':
        StartingMeasurementMask=InitialMaskObject.MaskType + '_' + str(InitialMaskObject.MaskNumber) + '_' + str(InitialMaskObject.RowSz) + '_' + str(InitialMaskObject.ColSz) + '_Percentage_' + str(InitialMaskObject.Percentage);
        loadPathInitialMask = loadPathInitialMask + os.path.sep + StartingMeasurementMask                                                               
        if not os.path.exists(loadPathInitialMask):                                                                                                                          
            sys.exit('Error!!! Check foder .ResultsAndData/InitialSamplingMasks/ for folder ' + loadPathInitialMask)                                                            
        Mask = np.load(loadPathInitialMask + os.path.sep + 'SampleMatrix.npy')
    else:
        Mask = generateInitialMask(InitialMaskObject,SizeImage)
    return Mask

def computeStopCondFuncVal(ReconValues,MeasuredValues,StopCondParamsObject,ImageType,StopCondFuncVal,MaxIdxsVect,NumSamples,IterNum,BatchSamplingParamsObject):
    
    if BatchSamplingParamsObject.Do=='N':
        Diff=computeDifference(ReconValues[MaxIdxsVect],MeasuredValues[NumSamples-1],ImageType)
        if IterNum == 0:
            StopCondFuncVal[IterNum,0] = StopCondParamsObject.Beta*Diff
        else:
            StopCondFuncVal[IterNum,0] = ((1-StopCondParamsObject.Beta)*StopCondFuncVal[IterNum-1,0] + StopCondParamsObject.Beta*Diff)
        StopCondFuncVal[IterNum,1] = NumSamples
    
    else:
        Diff=0
        for i in range(0,BatchSamplingParamsObject.NumSamplesPerIter):
            Diff=computeDifference(ReconValues[MaxIdxsVect[i]],MeasuredValues[NumSamples-1-(BatchSamplingParamsObject.NumSamplesPerIter-i-1)],ImageType)+Diff
        Diff = Diff/BatchSamplingParamsObject.NumSamplesPerIter
        if IterNum == 0:
            StopCondFuncVal[IterNum,0] = StopCondParamsObject.Beta*Diff
        else:
            StopCondFuncVal[IterNum,0] = ((1-StopCondParamsObject.Beta)*StopCondFuncVal[IterNum-1,0] + StopCondParamsObject.Beta*Diff)
        StopCondFuncVal[IterNum,1] = NumSamples
    return StopCondFuncVal

def checkStopCondFuncThreshold(StopCondParamsObject,StopCondFuncVal,NumSamples,IterNum,SizeImage):
    
    if StopCondParamsObject.Threshold==0:
        if NumSamples>SizeImage[0]*SizeImage[1]*StopCondParamsObject.MaxPercentage/100:
            Stop=1
        else:
            Stop=0

    else:
        if NumSamples>SizeImage[0]*SizeImage[1]*StopCondParamsObject.MaxPercentage/100:
            Stop=1
        else:
            if np.logical_and(((SizeImage[0]*SizeImage[1])*StopCondParamsObject.MinPercentage/100)<NumSamples,StopCondFuncVal[IterNum,0]<StopCondParamsObject.Threshold):
                Stop=0
                GradStopCondFunc =np.mean(StopCondFuncVal[IterNum,0]-StopCondFuncVal[IterNum-StopCondParamsObject.JforGradient:IterNum-1,0])
                if GradStopCondFunc<0:
                    Stop=1
            else:
                Stop=0
    return Stop

def computeBeta(SizeImage):
    import math
    if SizeImage[0]*SizeImage[1]<512**2+1:
        Beta = 0.001*(((18-math.log(SizeImage[0]*SizeImage[1],2))/2)+1)
    else:
        Beta = 0.001/(((math.log(SizeImage[0]*SizeImage[1],2)-18)/2)+1)
    return Beta     

def updateERDandFindNewLocationFirst(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,NumSamples,UpdateERDParamsObject,BatchSamplingParamsObject):

    ERDValues,ReconValues,ReconImage = computeFullERD(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType)

    NewIdxs,MaxIdxsVect = findNewMeasurementIdxs(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,NumSamples,UpdateERDParamsObject,ReconValues,ReconImage,ERDValues,BatchSamplingParamsObject)

    return(Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect)


def updateERDandFindNewLocationAfter(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,UpdateERDParamsObject,BatchSamplingParamsObject,StopCondFuncVal,IterNum,NumSamples,NewIdxs,ReconValues,ReconImage,ERDValues,MaxIdxsVect):
    
    if UpdateERDParamsObject.Do == 'N':
        ERDValues,ReconValues,ReconImage = computeFullERD(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType)
    else:
        ERDValues,ReconValues=updateERD(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,NewIdxs,NumSamples,UpdateERDParamsObject,ReconValues,ReconImage,ERDValues,MaxIdxsVect,BatchSamplingParamsObject)
    
    NewIdxs,MaxIdxsVect = findNewMeasurementIdxs(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfoObject,Resolution,ImageType,NumSamples,UpdateERDParamsObject,ReconValues,ReconImage,ERDValues,BatchSamplingParamsObject)

    return(Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect)

def performReconOnce(SavePath,TrainingInfoObject,Resolution,SizeImage,ImageType,codePath,ImageSet,ImNum,ImageExtension,SimulationRun,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,directImagePath):

    NeighborValues,NeighborWeights,NeighborDistances = FindNeighbors(TrainingInfoObject,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Resolution)
    ReconValues,ReconImage = ComputeRecons(TrainingInfoObject,NeighborValues,NeighborWeights,SizeImage,UnMeasuredIdxs,MeasuredIdxs,MeasuredValues)    
    
    if (SimulationRun==1):
        Img = misc.imread(directImagePath)
    else: 
        Img = loadTestImage(codePath,ImageSet,ImNum,ImageExtension,SimulationRun)
    
    Difference = np.sum(computeDifference(Img,ReconImage,ImageType))
    return(Difference,ReconImage)

def findStoppingThreshold(trainingDataPath,NumTrainingImages,Best_c,PercentageInitialMask,DesiredTD,reconPercVector,SizeImage):
    sys.path.append('code')
    SavePathSLADS = trainingDataPath + 'SLADSResults' 
    Thresh = np.zeros(NumTrainingImages)
    count=0
    for ImNum in range(0,NumTrainingImages): 
        LoadPath = SavePathSLADS + os.path.sep + 'Image_' + str(ImNum+1) + '_c_'+ str(Best_c) + os.path.sep
        StopCondFuncVal = np.load(LoadPath + 'StopCondFuncVal.npy')
        TD = np.load(LoadPath + 'TD.npy')
        found=0
        for i in range(0,TD.shape[0]):
            if TD[i]<DesiredTD and found==0 :
                Idx = int((reconPercVector[i]-PercentageInitialMask)*SizeImage[0]*SizeImage[1]/100)
                Thresh[ImNum]=StopCondFuncVal[Idx][0]
                count=count+1
                found=1
    Threshold = np.sum(Thresh)/count
    sys.path.pop()
    return Threshold

def generateInitialMask(InitialMaskObject,SizeImage):
    if InitialMaskObject.MaskType =='R':
        Mask = np.zeros((SizeImage[0],SizeImage[1]))
        UnifMatrix = np.random.rand(SizeImage[0],SizeImage[1])
        Mask = UnifMatrix<(InitialMaskObject.Percentage/100)
    elif InitialMaskObject.MaskType =='U':
        Mask = np.zeros((SizeImage[0],SizeImage[1]))
        ModVal = int(100/InitialMaskObject.Percentage)
        for r in range(0,SizeImage[0]):
            for s in range(0,SizeImage[1]): 
                LinIdx = r*SizeImage[1]+s
                if np.remainder(LinIdx,ModVal)==0:
                    Mask[r][s]=1
    return Mask
        
def plotImage(Image,Num):
    plt.imshow(Image)
    #pylab.show()

def plotAfterSLADS(Im1,Im2):
    plt.figure(1)                
    plt.subplot(121)          
    plt.imshow(Im1)
    plt.title('Sampled Mask')   
    plt.subplot(122)        
    plt.imshow(Im2)
    plt.title('Reconstructed Image')

def importImages(dirPath, inputExtension, SizeImage):
    if (inputExtension == ".tif"):
        dataFileNames = glob.glob(dirPath + "/*" + inputExtension) #Obtain filenames for each set
        zLen = len(dataFileNames) #Find total number of files imported
        if (zLen == 0):
            sys.exit("Error!!! There are no files with extension: " + inputExtension + " in the directory: " + dirPath)
        dataset = Image.open(dataFileNames[0]).convert('L') #Read in an image as grayscale
        dataset = np.asarray(dataset,dtype=np.float64).reshape((dataset.size[1],dataset.size[0])) #Flatten the image
        #firstX, firstY = dataset.shape[0], dataset.shape[1] #Obtain the set's dimensions
        firstX, firstY = SizeImage[0], SizeImage[1]
        datasets = [] #Create an empty array to hold instances of imageData objects
        counter = 0
        for file in dataFileNames: #For each of the filenames
            dataset = Image.open(file).convert('L') #Read in an image as grayscale
            datasetXLen, datasetYLen = dataset.size #Obtain the set's dimensions
            if ((SizeImage[0], SizeImage[1]) != (datasetXLen, datasetYLen)): #If any of the images differ in dimensions
                if(debugInfo): print("Warning!!! File:", dataFileNames[counter], "has dimensions X: ", datasetXLen, " Y: ", datasetYLen, " - Will resize") #Inform the user
                dataset = dataset.resize([SizeImage[0], SizeImage[1]]) #Resize the file to the dimensions specified
            dataset = np.asarray(dataset,dtype=np.float64).reshape((dataset.size[1],dataset.size[0])) #Flatten the image
            outputName = file
            outputName = outputName[outputName.startswith(dirPath) and len(dirPath):]
            outputName = re.sub('\.tif$', '', outputName)
            datasets.append(imageData(dataset, outputName))
            counter+=1
    return datasets


def gausKern_parhelper(sigmaVal, WindowSize, area): #Parallel loop for generating a Gaussian kernel
    return area*generateGaussianKernel(sigmaVal,WindowSize) #Calculate an "area" that the c value will capture based on a gaussian filter

def stats_parhelper(NumImages, Mask, codePath, TrainingImageSet, SizeImage, StopCondParamsObject, Theta, TrainingInfoObject, TestingInfoObject, Resolution, ImageType, UpdateERDParamsObject, BatchSamplingParamsObject, trainingPlotFeaturesPath, SimulationRun, ImNum, ImageExtension, PlotResult, Classify, directImagePath, errorPlot, isRunningParallel):    
    return runSLADSSimulationOnce(NumImages, Mask, codePath, TrainingImageSet, SizeImage, StopCondParamsObject, Theta, TrainingInfoObject, TestingInfoObject, Resolution, ImageType, UpdateERDParamsObject, BatchSamplingParamsObject, trainingPlotFeaturesPath, SimulationRun, ImNum, ImageExtension, PlotResult, Classify, directImagePath, errorPlot, isRunningParallel)

#Return the save path for output plots from testing of end models
def testingOutputName(testingFeaturesPath, dataFileName, StopPercentageSLADS, StopPercentageTestingSLADS):
    outputName = dataFileName
    outputName = outputName[outputName.startswith(dataPath) and len(dataPath):]
    outputName = re.sub('\.png', '', outputName)
    outputName = re.sub(testingDataImagesPath, '', outputName)
    outputName = outputName + '_c_'+ str(c) + '_StopTrainPerc_' + str(StopPercentageSLADS) + '_StopTestPerc_' + str(StopPercentageTestingSLADS) + '_'
    return testingFeaturesPath + outputName

def cls(): #Clear console screen
    os.system('cls' if os.name=='nt' else 'clear')   


# In[ ]:


#MAIN PROGRAM
#==================================================================
cls() #Clear the screen
jNotebook = False #Is the program being run in a jupyter-notebook; Program progress bars will not function correctly if True

#GENERAL PARAMETERS: L-01
#==================================================================

#Is training of a model to be performed
trainingModel = True

#Is testing of a model to be performed
testingModel = True

#Should the default multithreading be limited
overrideThreads = False

#If the default multithreading be limited how many threads should be used
num_threads = 1

#Should warning/debug information be displayed?
debugInfo = False

#Type of Image: D - for discrete (binary) image; C - for continuous
ImageType = 'C'

#What is the file extenstion of the data?
inputExtension = ".tif";

#What is the symmetric length all images should be resized to?
#(64x64), (128x128), (256x256), (512x512)
SizeImage = [64, 64]

#Find threshold for stopping function Y/N; If 'Y', set the DesiredTD in L-1
FindStopThresh = 'N'

#TRAINING MODEL PARAMETERS: L-02
#==================================================================

#Enter 'X' for TrainingInputDB_X
TrainingImageSet = '1'

#Should the input data be split into training/testing sets
#If true and customTesting is also true, then custom set will be added to the split of the trainingImageSet database
#If false and testingModel is true, then only custom set will be used
splitInputSet = True

#Should error data be plotted for the training data
trainingErrorPlot = True

#If trainingErrorPlot, should the plot use the testing data to test convergence
trainingErrorPlotwTesting = True

#Should c value vs distortion metric be plotted; (NOT WORKING CORRECTLY)
cPlot = False

#Stopping percentage for SLADS (to select C)
#Suggested: (64x64):50, (128x128):30, (256x256):20, (512x512):10
#Given ttPlotAverageErrors linestyle settings please enter no more than 4 values
StopPercentageSLADSArr = np.array([50])

#TESTING MODEL PARAMETERS: L-03
#==================================================================

#Is there a custom testing set of images to be used?
customTesting = False

#If customTesting set is true, enter 'X' for TestingInputDB_X
TestingImageSet = '1'

#If customTesting set is false, indicate % of data to use for training; test is 1-(trainSplit/100)
trainSplit = 80 

#Should error data be plotted for the testing data
testingErrorPlot = True

#Should the best c found during training be overidden?
overrideBestC = False

#If the best c should be overiden with a manual value what should that value be?
overrideBestCValue = 4

#Stopping percentage for SLADS
#Default uses the same value as the training model
#StopPercentageTestingSLADSArr = StopPercentageSLADSArr
StopPercentageTestingSLADSArr = np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80])

#PROGRAM PARAMETERS: L-1
#==================================================================

# Sweep range for c (to select best c for RD approximation)
c_vec = np.array([2,4,8,16,32])

# Sampling mask measurement percentages for training (best left unchanged)
MeasurementPercentageVector = np.array([5,10,20,40,80])

# Window size for approximate RD summation (best left unchanged)
WindowSize = [15,15]

# Update ERD or compute full ERD in SLADS (to find best c)
# with Update ERD, ERD only updated for a window surrounding new measurement
Update_ERD = 'Y' 

# Smallest ERD update window size permitted
MinWindSize = 3  

# Largest ERD update window size permitted  
MaxWindSize = 10       

# Initial Mask for SLADS (to find best c):
# Percentage of samples in initial mask
PercentageInitialMask = 1

# Type of initial mask   
# Choices: 
    # 'U': Uniform mask; can choose any percentage
    # 'R': Randomly distributed mask; can choose any percentage
    # 'H': low-dsicrepacy mask; can only choose 1% mas
MaskType = 'H'                   

# Desired total distortion (TD) value (to find threshold on stopping function)
# TD = D(X,\hat(X))/(Number of pixels in image)
# D(X,\hat(X)) is difference between actual image X and reconstructed image 
# \hat(X) (summed over all pixels)
# For ImageType 'D' in range [0-1] for ImageType 'C' in range [0-max value]
DesiredTD=0

#What is the file extention desired for the pre-processed data
ImageExtension = ".png"


# In[ ]:


#STATIC VARIABLE SETUP
#==================================================================

#Set the number of cpu threads to be used for generating Gaussian Kernals
if not overrideThreads:
    num_threads = multiprocessing.cpu_count() #Determine number of available threads

#Determine console size
if not jNotebook:
    consoleRows, consoleColumns = os.popen('stty size', 'r').read().split()
else:
    consoleRows = 40
    consoleColumns = 40

#Set global plot parameters
plt.rcParams["font.family"] = "Times New Roman"
    
#Convert types as needed
c_vec = c_vec.astype(float)
MeasurementPercentageVector = MeasurementPercentageVector.astype(float)
StopPercentageSLADSArr = StopPercentageSLADSArr.astype(float)
PercentageInitialMask = float(PercentageInitialMask)

NumReconsSLADS = 10 #UNKNOWN VARIABLE
PercOfRD = 50 #UNKNOWN VARIABLE
Classify = 'N' #UNKNOWN VARIABLE
Resolution = 1 #UNKNOWN VARIABLE

falseFlag = False #Hold a False variable for function calls to disable certain parameters

#Store training information as object
TrainingInfoObject = TrainingInfo()

#Initalize the training information as set through specifieed variables
if ImageType == 'D':
    TrainingInfoObject.initialize('DWM','DWM',2,10,'Gaussian',0,0.25,15)
elif ImageType == 'C':
    TrainingInfoObject.initialize('CWM','CWM',2,10,'Gaussian',0,0.25,15)

#Store testing information as object
TestingInfoObject = TestingInfo()
TestingInfoObject.initialize()
    
InitialMaskObject = InitialMask()
InitialMaskObject.initialize(SizeImage[0],SizeImage[1],MaskType,1,PercentageInitialMask)
    
#PATH/DIRECTORY SETUP
#==================================================================

#Set starting data path
codePath = '.' + os.path.sep

#Set path to Results and Data Folder
resultsDataPath = codePath + 'ResultsAndData' + os.path.sep

#Check directory
if not os.path.exists(resultsDataPath):                                                                                                                          
    sys.exit('Error!!! The folder ' + resultsDataPath + ' does not exist. ')

TrainingDBName = 'TrainingDB_' + TrainingImageSet

#If split is being conducted
if splitInputSet:
    dataPath = resultsDataPath + 'InputData' + os.path.sep + 'InputTrainingDB_' + str(TrainingImageSet) + os.path.sep

if trainingModel: #If training is to be performed, check the relevant directories
    
    #Path to the input training image database
    dataPath = resultsDataPath + 'InputData' + os.path.sep + 'InputTrainingDB_' + str(TrainingImageSet) + os.path.sep
    if not os.path.exists(dataPath):                                                                                                                          
        sys.exit('Error!!! The folder ' + dataPath + ' does not exist. Check entry for ' + TrainingImageSet)

    #Path to where training resources/features should be saved
    trainingFeaturesPath = resultsDataPath + 'TrainingSavedFeatures' + os.path.sep
    if os.path.exists(trainingFeaturesPath):
        shutil.rmtree(trainingFeaturesPath)
    os.makedirs(trainingFeaturesPath)
          
    if trainingErrorPlot:
        trainingPlotFeaturesPath = trainingFeaturesPath + 'Plot' + os.path.sep
        if os.path.exists(trainingPlotFeaturesPath):
            shutil.rmtree(trainingPlotFeaturesPath)
        os.makedirs(trainingPlotFeaturesPath)
    
    #Set path to where training data should be kept
    trainingDataPath = trainingFeaturesPath + 'TrainingDB_' + str(TrainingImageSet) + os.path.sep
    if not os.path.exists(trainingDataPath):                                                                                                                          
        os.makedirs(trainingDataPath)
    
    #Set path to where training data images should be kept
    trainingDataImagesPath = trainingDataPath + 'Images' + os.path.sep   
    if os.path.exists(trainingDataImagesPath):
        shutil.rmtree(trainingDataImagesPath)
    os.makedirs(trainingDataImagesPath)
        
    #Clear and setup training model path
    if os.path.exists(trainingDataPath + 'FeaturesRegressCoeffs'):
        shutil.rmtree(trainingDataPath + 'FeaturesRegressCoeffs')
    os.makedirs(trainingDataPath + 'FeaturesRegressCoeffs')
    
if testingModel: #If testing is to be performed, check the relevant directories
     
    #Path to where training resources/features should be saved
    testingFeaturesPath = resultsDataPath + 'TestingSavedFeatures' + os.path.sep
    if os.path.exists(testingFeaturesPath):
        shutil.rmtree(testingFeaturesPath)
    os.makedirs(testingFeaturesPath)
        
    #Set path to where testing data should be kept
    testingDataPath = testingFeaturesPath + 'TestingDB_' + str(TrainingImageSet) + os.path.sep
    if not os.path.exists(testingDataPath):                                                                                                                          
        os.makedirs(testingDataPath)
        
    #Set path to where initial input files are located
    if (customTesting): #If a custom training set is going to be used
        customTestingDataPath = resultsDataPath + 'InputData' + os.path.sep + 'InputTestingDB_' + str(TestingImageSet) + os.path.sep
        if not os.path.exists(customTestingDataPath):
            sys.exit('Error!!! The folder ' + customTestingDataPath + ' does not exist. Check entry for ' + customTestingDataPath + ' or set the customTesting flag to false.')
    
    #Set data path to where data should be split from
    if not customTesting and not trainingModel and splitInputSet:
        dataPath = resultsDataPath + 'InputData' + os.path.sep + 'InputTrainingDB_' + str(TrainingImageSet) + os.path.sep
        
    #Set path to where testing data images should be kept 
    testingDataImagesPath = testingDataPath + 'Images' + os.path.sep   
    if os.path.exists(testingDataImagesPath):
        shutil.rmtree(testingDataImagesPath)
    os.makedirs(testingDataImagesPath)
        
    #Clear and setup testing model path
    if os.path.exists(testingDataPath + 'FeaturesRegressCoeffs'):
        shutil.rmtree(testingDataPath + 'FeaturesRegressCoeffs')
    os.makedirs(testingDataPath + 'FeaturesRegressCoeffs')

    #Set path to save training results
    TrainingSavePath = resultsDataPath + 'SLADSSimulationResults' + os.path.sep + 'TrainingDB_' + TrainingImageSet + os.path.sep
    if os.path.exists(TrainingSavePath):
        shutil.rmtree(TrainingSavePath)
    os.makedirs(TrainingSavePath)
    
    #Set path to save testing results
    testingSavePath = resultsDataPath + 'SLADSSimulationResults' + os.path.sep + 'TestingDB_' + TestingImageSet + os.path.sep
    if os.path.exists(testingSavePath):
        shutil.rmtree(testingSavePath)
    os.makedirs(testingSavePath)
    
    #Set path for saving the resulting images from the testing results
    ImagesSavePath = testingSavePath + 'TestingImageResults' + os.path.sep    
    if os.path.exists(ImagesSavePath):
        shutil.rmtree(ImagesSavePath)  
    os.makedirs(ImagesSavePath)

    #Set path for saving the resulting training statistics
    trainingStatisticsSavePath = TrainingSavePath + 'Statistics' + os.path.sep    
    if os.path.exists(trainingStatisticsSavePath):
        shutil.rmtree(trainingStatisticsSavePath)  
    os.makedirs(trainingStatisticsSavePath)    

    #Set path for saving the resulting testing statistics
    testingStatisticsSavePath = testingSavePath + 'Statistics' + os.path.sep    
    if os.path.exists(testingStatisticsSavePath):
        shutil.rmtree(testingStatisticsSavePath)  
    os.makedirs(testingStatisticsSavePath)    
    
    #Set the path for where the initial mask locations are 
    loadPathInitialMask = resultsDataPath + 'InitialSamplingMasks' #Set initial mask path
    if not os.path.exists(loadPathInitialMask):                                                                                                                          
        sys.exit('Error!!! The folder ' + loadPathInitialMask + ' does not exist. Check entry for ' + loadPathInitialMask)


# In[ ]:


#DATA IMPORTATION
#==================================================================
print('#' * int(consoleColumns))
print('PRE-PROCESSING DATA')
print('#' * int(consoleColumns) + '\n')
#Check validity of training/testing parameters
if testingModel and not splitInputSet and not customTesting:
    sys.exit("ERROR!!! Testing enabled, but neither custom, nor split input sets enabled")

if splitInputSet and not testingModel:
    if(debugInfo): print("WARNING!!! Testing disabled, but splitInputSet enabled. Only the specified fraction of the training set will be used")

#If any training is to be performed
if trainingModel: 
    datasets = importImages(dataPath, inputExtension, SizeImage) #Import images and perform pre-processing
    datasets = shuffle(datasets) #Randomize dataset order
    if splitInputSet: #If data should be split
        numTrain = int(len(datasets)*(trainSplit/100)) #Find number of training examples; round as int for indexing
        trainingData = datasets[0:numTrain] #Split off the training dataset
    else: #The data should not be split
        trainingData = datasets #Simply pass along the data as the training set
        
    #Save pre-processed training images   
    for i in range(0,len(trainingData)): #For each dataset in the training data
        dataset = trainingData[i]
        grayImage=Image.fromarray(dataset.data).convert("L") #Reset the values for the image such that they are visible
        grayImage.save(trainingDataImagesPath+dataset.name+ImageExtension) #Save the output image
        trainingData[i] = misc.imread(trainingDataImagesPath+dataset.name+ImageExtension)
        
if testingModel:
    if splitInputSet and not trainingModel:
        datasets = importImages(dataPath, inputExtension, SizeImage) #Import images and perform pre-processing
        datasets = shuffle(datasets) #Randomize dataset order
        numTrain = int(len(datasets)*(trainSplit/100)) #Find number of training examples; round as int for indexing
    
    #If only split data from the training set is to be used for testing
    if not customTesting and splitInputSet:
        testingData = datasets[numTrain:(len(datasets))]

    #If split data and custom data should be used for testing
    if customTesting and splitInputSet:
        testingData = datasets[numTrain:(len(datasets))] + importImages(customTestingDataPath, inputExtension, SizeImage)

    #If only custom data should be used for testing
    if customTesting and not splitInputSet:
        testingData = importImages(customTestingDataPath, inputExtension, SizeImage)
    
    #Save pre-processed testingData
    for i in range(0, len(testingData)): 
        dataset = testingData[i]
        grayImage=Image.fromarray(dataset.data).convert("L") #Reset the values for the image such that they are visible
        grayImage.save(testingDataImagesPath+dataset.name+ImageExtension) #Save the output image
        testingData[i] = misc.imread(testingDataImagesPath+dataset.name+ImageExtension) #Import into final testing dataset


# In[ ]:


#RUN TRAINING
#==================================================================  
if trainingModel:
    print(' \r') #Reset cursor
    print('#' * int(consoleColumns))
    print('TRAINING MODEL')
    print('#' * int(consoleColumns) + '\n')
    
    NumTrainingImages = len(trainingData)
    imagePercent = (1/NumTrainingImages)*100
    isRunningParallel = True
    if trainingErrorPlotwTesting:
            if not testingModel:
                sys.exit('Error!!! trainningErrorPlotwTesting enabled, but testingModel is not')
            else: 
                testingImageFiles = glob.glob(testingDataImagesPath + "/*" + ImageExtension) #Obtain filenames for each set

    #For each of the possible sampling percentages
    for p in tqdm(range(0,len(StopPercentageSLADSArr)), desc = 'Sampling Percentages', leave = True):
        
        StopPercentageSLADS = StopPercentageSLADSArr[p]
        #Vector of values to determine for the reconstruction

        reconPercVector = np.linspace(PercentageInitialMask, StopPercentageSLADS, num=NumReconsSLADS*(StopPercentageSLADS-PercentageInitialMask), endpoint=False)

        #For each of the Sampling mask measurement percentages for training
        #For each of the images want to maximize the reduction in distortion 
        #Train regression formula on known images with randomly selected sampling densities
        #Estimation of distortion reduction based on a provided weight parameter and the distance between the previous and proposed pixel location
        samplingPercent = (imagePercent/np.size(MeasurementPercentageVector))
        cPercent = samplingPercent/len(c_vec)

        for ImNum in tqdm(range(0,len(trainingData)), desc = 'Training Images', leave = True):
            Img = trainingData[ImNum] #Retreive a training image
            try:
                Img
            except NameError:
                sys.exit('Error!!! There are no images in the training set!')

            for m in tqdm(range(0,np.size(MeasurementPercentageVector)), desc = 'Sampling Densities', leave = True): #For each of the proposed sampling densities
                #Create a save folder for the produced information
                SaveFolder = 'Image_' + str(ImNum+1) + '_SampPerc_' + str(MeasurementPercentageVector[m])
                trainingFeaturesSavePath = trainingDataPath + 'FeaturesRegressCoeffs' + os.path.sep + SaveFolder
                if not os.path.exists(trainingFeaturesSavePath):
                    os.makedirs(trainingFeaturesSavePath)

                #Create a random uniform, boolean mask at the sampling density specified and apply it to the image
                Mask = np.zeros((SizeImage[0],SizeImage[1]))
                UnifMatrix = np.random.rand(SizeImage[0],SizeImage[1])
                Mask = UnifMatrix<(MeasurementPercentageVector[m]/100)
                MeasuredIdxs = np.transpose(np.where(Mask==1))
                UnMeasuredIdxs = np.transpose(np.where(Mask==0))            
                MeasuredValues = Img[Mask==1]

                # Find neighbors
                NeighborValues,NeighborWeights,NeighborDistances = FindNeighbors(TrainingInfoObject,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Resolution)

                #Form the reconstructed image
                ReconValues,ReconImage = ComputeRecons(TrainingInfoObject,NeighborValues,NeighborWeights,SizeImage,UnMeasuredIdxs,MeasuredIdxs,MeasuredValues)

                #Compute the image features
                AllPolyFeatures=computeFeatures(UnMeasuredIdxs,SizeImage,NeighborValues,NeighborWeights,NeighborDistances,TrainingInfoObject,ReconValues,ReconImage,Resolution,ImageType)

                #Determine a number of points 
                #unmasked percent * percentage of distortion reduction * imageArea / (100*100)
                NumRandChoices = int((100-MeasurementPercentageVector[m])*PercOfRD*SizeImage[1]*SizeImage[0]/(100*100))

                #Create a sample based on a number of previously unmeasured points
                OrderForRD = random.sample(range(0,UnMeasuredIdxs.shape[0]), NumRandChoices) 

                #Extract the image features of interest from the randomly selected unmeasured points
                PolyFeatures = AllPolyFeatures[OrderForRD,:]

                #Compute the differences between the original and reconstructed images
                RDPP = computeDifference(Img,ReconImage,ImageType)

                #Round differences to nearest integer
                RDPP.astype(int)

                #Pad the differences
                RDPPWithZeros = np.lib.pad(RDPP,(int(np.floor(WindowSize[0]/2)),int(np.floor(WindowSize[1]/2))),'constant',constant_values=0)

                #Convert image to an array
                ImgAsBlocks = im2col(RDPPWithZeros,WindowSize)

                #Flatten 2D mask array to 1-D
                MaskVect = np.ravel(Mask)

                #Identify the pixels that have not yet been measured
                ImgAsBlocksOnlyUnmeasured = ImgAsBlocks[:,np.logical_not(MaskVect)]

                temp = np.zeros((WindowSize[0]*WindowSize[1],NumRandChoices))
                for n in tqdm(range(0,len(c_vec)), desc = 'c Values', leave = True): #For each of the possible c values
                    c = c_vec[n]
                    sigma = NeighborDistances[:,0]/c
                    cnt = 0;
                    
                    #parallize runSLADSSimulationOnce
                    area = Parallel(n_jobs=num_threads)(delayed(gausKern_parhelper)(sigma[OrderForRD[index]], WindowSize, ImgAsBlocksOnlyUnmeasured[:,OrderForRD[index]]) for index in tqdm(range(0,len(OrderForRD)), desc = 'Gaussian', leave = True)) #Perform task in parallel
                    for i in range (0,len(OrderForRD)): temp[:,i] = area[i]
                    RD = np.sum(temp, axis=0) #Determine how much "area of uncertainty" is possibly removed for a c value

                    #Save everything
                    SavePath_c = trainingFeaturesSavePath + os.path.sep + 'c_' + str(c)
                    if not os.path.exists(SavePath_c):
                        os.makedirs(SavePath_c)
                    np.save(SavePath_c + os.path.sep + 'RD' + '_StopPerc_' + str(StopPercentageSLADS), RD)        
                    np.save(SavePath_c + os.path.sep + 'OrderForRD' + '_StopPerc_' + str(StopPercentageSLADS), OrderForRD)

                np.save(trainingFeaturesSavePath + os.path.sep + 'Mask' + '_StopPerc_'+ str(StopPercentageSLADS), Mask)   
                np.save(trainingFeaturesSavePath + os.path.sep + 'ReconImage' + '_StopPerc_' + str(StopPercentageSLADS), ReconImage)
                np.save(trainingFeaturesSavePath + os.path.sep + 'PolyFeatures' + '_StopPerc_' + str(StopPercentageSLADS), PolyFeatures)
 
    print('\n\n\n\n\n' + ('-' * int(consoleColumns)))
    print('PERFORM TRAINING')
    print('-' * int(consoleColumns) + '\n')
    
    #For each of the possible sampling percentages
    for p in tqdm(range(0,len(StopPercentageSLADSArr)), desc = 'Sampling Percentages', leave = True):
        StopPercentageSLADS = StopPercentageSLADSArr[p]
        
        #Append the observed polyfeatures for each sampling density and image to a single big array
        #NOTE: WHAT ABOUT LEAVING THESE IN MEMORY TO REDUCE TIME?
        for i in tqdm(range(0,len(c_vec)), desc = 'c Values', leave = True): #For each of the proposed c values
            c = c_vec[i]
            FirstLoop = 1
            for ImNum in tqdm(range(0,NumTrainingImages), desc = 'Training Images', leave = True): #For each of the possible training images
                
                for m in tqdm(range(0,np.size(MeasurementPercentageVector)), desc = 'Sampling Densities', leave = True): #For each of the proposed sampling densities
                    #Set loading paths
                    LoadPath = trainingDataPath + 'FeaturesRegressCoeffs' + os.path.sep + 'Image_' + str(ImNum+1) + '_SampPerc_' + str(MeasurementPercentageVector[m])
                    LoadPath_c = LoadPath + os.path.sep + 'c_' + str(c)

                    #Load the image polynomial feature 
                    PolyFeatures = np.load(LoadPath + os.path.sep + 'PolyFeatures' + '_StopPerc_' + str(StopPercentageSLADS)+ '.npy')

                    #Load the possible reduction in distortion value
                    RD = np.load(LoadPath_c + os.path.sep + 'RD' + '_StopPerc_' + str(StopPercentageSLADS)+ '.npy')

                    if ImageType=='D':
                        if FirstLoop==1:
                            BigPolyFeatures = np.column_stack((PolyFeatures[:,0:25],PolyFeatures[:,26]))
                            BigRD = RD
                            FirstLoop = 0                  
                        else:
                            TempPolyFeatures = np.column_stack((PolyFeatures[:,0:25],PolyFeatures[:,26]))                    
                            BigPolyFeatures = np.row_stack((BigPolyFeatures,TempPolyFeatures))
                            BigRD = np.append(BigRD,RD)

                    else: #If image type is continuous
                        if FirstLoop==1: #If for the first possible c
                            BigPolyFeatures = PolyFeatures
                            BigRD = RD
                            FirstLoop = 0                  
                        else:
                            TempPolyFeatures = PolyFeatures               
                            BigPolyFeatures = np.row_stack((BigPolyFeatures,TempPolyFeatures))
                            BigRD = np.append(BigRD,RD)                    

            regr = linear_model.LinearRegression()

            #Perform the regression, fitting the observed polynomial fetures to the expected reduction in distortion
            regr.fit(BigPolyFeatures, BigRD)

            Theta = np.zeros((PolyFeatures.shape[1]))    

            if ImageType=='D':            
                Theta[0:24]=regr.coef_[0:24]
                Theta[26]=regr.coef_[25]
            else: #If image type is continuous
                Theta = regr.coef_

            del BigRD,BigPolyFeatures

            SavePath_c = trainingDataPath + 'c_' + str(c)
            if not os.path.exists(SavePath_c):
                os.makedirs(SavePath_c) 
            np.save(SavePath_c + os.path.sep + 'Theta' + '_StopPerc_' + str(StopPercentageSLADS), Theta)

    print('\n\n\n\n' + ('-' * int(consoleColumns)))
    print('DETERMINING BEST C')
    print('-' * int(consoleColumns) + '\n')            
            
    #For each of the possible sampling percentages
    for p in tqdm(range(0,len(StopPercentageSLADSArr)), desc = 'Sampling Percentages', leave = True):
        StopPercentageSLADS = StopPercentageSLADSArr[p]
        
        UpdateERDParamsObject = UpdateERDParams()
        UpdateERDParamsObject.initialize(Update_ERD,MinWindSize,MaxWindSize,1.5)

        # Find the best value of c
        directImagePath = '' #Direct image support not yet supported for training sets
        Best_c,NumImagesForSLADS = performSLADStoFindC(codePath,trainingDataPath,TrainingImageSet,ImageType,ImageExtension,TrainingInfoObject,SizeImage,StopPercentageSLADS,Resolution,c_vec,UpdateERDParamsObject,InitialMaskObject,MaskType,reconPercVector,Classify,directImagePath,consoleRows,cPlot,trainingStatisticsSavePath)
        
        #Set path to where best c value should be kept
        SavePath_bestc = trainingDataPath + 'best_c' + '_StopPerc_' + str(StopPercentageSLADS) + '.npy'
        np.save(SavePath_bestc, np.array([Best_c]))

        #Directory checking
        ThetaSavePath = trainingFeaturesPath + TrainingDBName + os.path.sep + 'c_' + str(Best_c) + os.path.sep
        ThetaLoadPath = trainingFeaturesPath + TrainingDBName + os.path.sep + 'c_' + str(Best_c) + os.path.sep
        if not os.path.exists(ThetaSavePath):
            os.makedirs(ThetaSavePath)

        np.save(ThetaSavePath + 'Theta' + '_StopPerc_'+ str(StopPercentageSLADS), Theta)

        # Find the Threshold on stopping condition that corresponds to the desired total distortion (TD) value set above
        if FindStopThresh=='Y':   
            Threshold=findStoppingThreshold(trainingDataPath,NumImagesForSLADS,Best_c,PercentageInitialMask,DesiredTD,reconPercVector,SizeImage)
            #print('For a TD of '+ str(DesiredTD) + ' set stopping function threshold to: ' + str(Threshold))
            #print('**** Make sure to enter this value in runSimulations.py and in runSLADS.py')
            #print('The threshold value is saved in:  ' + ThetaSavePath + ' as Threshold.npy')
            np.save(ThetaSavePath + 'Threshold' + '_StopPerc_'+ str(StopPercentageSLADS), Threshold) 


    #Now that the optimal C value has been found for reconstruction
    #Train the model 1 image at a time, perform a reconstruction and watch statistics over all training images
    if (trainingErrorPlot):
        print('\n\n\n\n' + ('-' * int(consoleColumns)))
        print('PLOTTING MODEL TRAINING CONVERGENCE')
        print(('-' * int(consoleColumns)) + '\n')   
        plotXLabel = '# Training Samples' #x label for model training convergence plot
        errorPlot = True
        
        #For each of the possible sampling percentages
        for p in tqdm(range(0,len(StopPercentageSLADSArr)), desc = 'Sampling Percentages', leave = True):
            StopPercentageSLADS = StopPercentageSLADSArr[p]
            Beta = computeBeta(SizeImage)
            StopCondParamsObject = StopCondParams()
            StopCondParamsObject.initialize(Beta,0,50,2,StopPercentageSLADS)
            UpdateERDParamsObject = UpdateERDParams()
            UpdateERDParamsObject.initialize(Update_ERD,MinWindSize,MaxWindSize,1.5)
            BatchSample = 'N'
            BatchSamplingParamsObject = BatchSamplingParams()
            if BatchSample=='N':
                BatchSamplingParamsObject.initialize(BatchSample,1)
            else:
                BatchSamplingParamsObject.initialize(BatchSample,NumSamplesPerIter)

            SimulationRun = 1
            PlotResult = 'N'
            regr = linear_model.LinearRegression() #Construct a regression model
            loadPathInitialMask = resultsDataPath + 'InitialSamplingMasks' # Load initial measurement mask
            FirstLoop = 1
            trainingStatisticsSavePathSuffix = '_StopPerc_' + str(StopPercentageSLADS)
            

            if trainingErrorPlotwTesting:
                NumImagesPlot = len(testingImageFiles)
                plottingFileNames = testingImageFiles
                trainingPlotTitle = 'Averaged Total Testing Image Results'
            else: #Using the training data for watching development of model with respect to # training samples
                NumImagesPlot = len(trainingImageFiles)
                plottingFileNames = trainingImageFiles
                trainingPlotTitle = 'Averaged Total Training Image Results'

            #Create an object to hold progressive development of model
            trainingResultsAverageObject = simulationResults() 
            trainingResultsAverageObject.initialize()
            #For each of the training images add the polynomial features determined for the best c to the model and check reconstruction capability
            for ImNum in tqdm(range(0,NumTrainingImages), desc = 'Training Images', leave = True): 
                #Aggregate together the polynomial features determined for the best c at each of the possible sampling densities
                for m in tqdm(range(0,np.size(MeasurementPercentageVector)), desc = 'Sampling Densities', leave = True): #For each of the proposed sampling densities

                    LoadPath = trainingDataPath + 'FeaturesRegressCoeffs' + os.path.sep + 'Image_' + str(ImNum+1) + '_SampPerc_' + str(MeasurementPercentageVector[m])
                    LoadPath_c = LoadPath + os.path.sep + 'c_' + str(Best_c) #Only using the best c found during the original training
                    RD = np.load(LoadPath_c + os.path.sep + 'RD'+ '_StopPerc_'+ str(StopPercentageSLADS)+ '.npy')  #Load the possible reduction in distortion value
                    PolyFeatures = np.load(LoadPath + os.path.sep + 'PolyFeatures' + '_StopPerc_'+ str(StopPercentageSLADS)+ '.npy')
                    if ImageType=='D':
                        if FirstLoop==1:
                            BigPolyFeatures = np.column_stack((PolyFeatures[:,0:25],PolyFeatures[:,26]))
                            BigRD = RD
                            FirstLoop = 0                  
                        else:
                            TempPolyFeatures = np.column_stack((PolyFeatures[:,0:25],PolyFeatures[:,26]))                    
                            BigPolyFeatures = np.row_stack((BigPolyFeatures,TempPolyFeatures))
                            BigRD = np.append(BigRD,RD)
                    else: #If image type is continuous
                        if FirstLoop==1: #If initialization
                            BigPolyFeatures = PolyFeatures
                            BigRD = RD
                            FirstLoop = 0                  
                        else:
                            TempPolyFeatures = PolyFeatures               
                            BigPolyFeatures = np.row_stack((BigPolyFeatures,TempPolyFeatures))
                            BigRD = np.append(BigRD,RD)                    

                #Perform the regression, fitting the observed polynomial fetures to the expected reduction in distortion
                regr = linear_model.LinearRegression() #Construct a new regression model
                regr.fit(BigPolyFeatures, BigRD)
                Theta = np.zeros((PolyFeatures.shape[1]))    
                if ImageType=='D':            
                    Theta[0:24]=regr.coef_[0:24]
                    Theta[26]=regr.coef_[25]
                else: #If image type is continuous
                    Theta = regr.coef_               

                #Perform SLADS on all of the images, saving statistics of interest in parallel
                trainingResultObject = Parallel(n_jobs=num_threads)(delayed(stats_parhelper)(NumImagesPlot, loadOrGenerateInitialMask(loadPathInitialMask,MaskType,InitialMaskObject,SizeImage), codePath, TestingImageSet, SizeImage, StopCondParamsObject, Theta, TrainingInfoObject, TestingInfoObject, Resolution, ImageType, UpdateERDParamsObject, BatchSamplingParamsObject, trainingPlotFeaturesPath, SimulationRun, i, ImageExtension, PlotResult, Classify, plottingFileNames[i], errorPlot, isRunningParallel) for i in tqdm(range(0,NumImagesPlot), desc = 'Avg. Stats.', leave = False))

                mseTrainingResults = []
                ssimTrainingResults = []
                distortTrainingResults = []
                for result in trainingResultObject: 
                    mseTrainingResults.append(result.mseError)
                    ssimTrainingResults.append(result.ssimError)
                    distortTrainingResults.append(result.totalDistortion)
                trainingResultsAverageObject.saveAverageErrorData(np.mean(mseTrainingResults), np.mean(ssimTrainingResults), np.mean(distortTrainingResults))
            xAxisValues = np.linspace(1, len(trainingResultsAverageObject.ssimAverageErrors), len(trainingResultsAverageObject.ssimAverageErrors))         
            trainingSpecificStatisticsSavePath = trainingStatisticsSavePath + 'StopTrainPerc_' + str(StopPercentageSLADS) + os.path.sep    
            
            #Directory setup for specific training statistics
            if os.path.exists(trainingSpecificStatisticsSavePath): 
                shutil.rmtree(trainingSpecificStatisticsSavePath)
            os.makedirs(trainingSpecificStatisticsSavePath)
            
            plotErrorData(trainingSpecificStatisticsSavePath, trainingStatisticsSavePathSuffix, trainingResultsAverageObject, xAxisValues, trainingPlotTitle, plotXLabel) #Plot and save the error data obtained during training
            del BigRD,BigPolyFeatures #Clean up workspace a bit


# In[ ]:


#RUN TESTING
#==================================================================  
if testingModel:
    print('\n\n' + ('#' * int(consoleColumns)))
    print('TESTING MODEL')
    print(('#' * int(consoleColumns)) + '\n')
            
    #Should the testing results be plotted
    PlotResult='Y'

    # If you want to use stopping function used, enter threshold (from Training), else leave at 0      
    StoppingThrehsold = 0

    #Static variables for sladssimulationonce call
    isRunningParallel = True
    SimulationRun = 1
    errorPlot = True
    
    #Directory setup for training database
    TrainingDBPath = trainingFeaturesPath + TrainingDBName
    if not os.path.exists(TrainingDBPath): 
        sys.exit('Error!!! The folder ' + TrainingDBPath + ' does not exist. Check entry for ' + TrainingImageSet)

    # Batch Sampling; If 'Y' set number of samples in each step in L-1 (NumSamplesPerIter)
    BatchSample = 'N'
    BatchSamplingParamsObject = BatchSamplingParams()
    if BatchSample=='N':
        BatchSamplingParamsObject.initialize(BatchSample,1)
    else:
        BatchSamplingParamsObject.initialize(BatchSample,NumSamplesPerIter)

    UpdateERDParamsObject = UpdateERDParams()
    UpdateERDParamsObject.initialize(Update_ERD,MinWindSize,MaxWindSize,1.5)

    PercentageInitialMask = float(PercentageInitialMask)

    Mask = loadOrGenerateInitialMask(loadPathInitialMask,MaskType,InitialMaskObject,SizeImage)
    np.save(testingFeaturesPath + 'InitialMask', Mask)
    savemat(testingFeaturesPath + 'InitialMask.mat', dict(Mask=Mask))
    
    testingPlotTitle = 'Averaged Total Testing Image Results'
    plotXLabel = '% Sampled'
    
    #Create a list for holding Average results for each training sample percentage
    trainTestAverageErrors = []
    #For each of the possible training sampling percentages (each with their own best c)
    for p in tqdm(range(0,len(StopPercentageSLADSArr)), desc = 'Training Sampling Percentages', leave = True):
        StopPercentageSLADS = float(StopPercentageSLADSArr[p])
        
        #Import the best c and theta values; Training for database must have been performed first
        if not overrideBestC: #If automatic best c selection
            LoadPath_bestc = trainingFeaturesPath + 'TrainingDB_' + str(TestingImageSet) + os.path.sep + 'best_c' + '_StopPerc_' + str(StopPercentageSLADS) + '.npy'
            if not os.path.exists(LoadPath_bestc):
                sys.exit('Error!!! The best c file ' + SavePath_bestc + ' does not exist. Check entry for ' + SavePath_bestc)
            c = np.load(LoadPath_bestc)[0].astype(float)
        else: #If manual best c selection
            c = overrideBestCValue.astype(float)

        #Directory setup for Theta
        ThetaLoadPath = trainingFeaturesPath + TrainingDBName + os.path.sep + 'c_' + str(c) + os.path.sep
        if not os.path.exists(ThetaLoadPath):                                                                                                                          
            sys.exit('Error!!! Check folder ./ResultsAndData/TrainingSavedFeatures/TrainingDB_' + TrainingImageSet + ' for folder c_' + str(c))
        
        #Load Theta
        Theta=np.transpose(np.load(ThetaLoadPath +'Theta' + '_StopPerc_' + str(StopPercentageSLADS) + '.npy'))

        Beta = computeBeta(SizeImage)
        StopCondParamsObject=StopCondParams()

        
        #Create an object to hold progressive development of model
        testingResultsAverageObject = simulationResults() 
        testingResultsAverageObject.initialize()
        
        #For each of the possible testing sampling percentages
        for q in tqdm(range(0,len(StopPercentageTestingSLADSArr)), desc = 'Testing Sampling Percentages', leave = True):
            StopPercentageTestingSLADS = float(StopPercentageTestingSLADSArr[q])
            StopCondParamsObject.initialize(Beta,StoppingThrehsold,50,2,StopPercentageTestingSLADS)
            
            # Run SLADS simulations
            dataFileNames = glob.glob(testingDataImagesPath + "/*" + ImageExtension) #Obtain filenames for each set
            numberTestFiles = len(dataFileNames)

            #Perform SLADS on all of the images, saving statistics of interest in parallel
            testingResultObject = Parallel(n_jobs=num_threads)(delayed(stats_parhelper)(numberTestFiles, loadOrGenerateInitialMask(loadPathInitialMask,MaskType,InitialMaskObject,SizeImage), codePath, TestingImageSet, SizeImage, StopCondParamsObject, Theta, TrainingInfoObject, TestingInfoObject, Resolution, ImageType, UpdateERDParamsObject, BatchSamplingParamsObject, testingOutputName(testingFeaturesPath, dataFileNames[i], StopPercentageSLADS, StopPercentageTestingSLADS),SimulationRun, i, ImageExtension, PlotResult, Classify, dataFileNames[i], errorPlot, isRunningParallel) for i in tqdm(range(0,numberTestFiles), desc = 'Testing Images', leave = True))
            
            #Create holding arrays for the results
            mseTestingResults = []
            ssimTestingResults = []
            distortTestingResults = []
            
            #Extract results from returned object
            for result in testingResultObject: 
                mseTestingResults.append(result.mseError)
                ssimTestingResults.append(result.ssimError)
                distortTestingResults.append(result.totalDistortion)
                
            #Store the Average DMs for all images for a particular testing sampling percentage
            testingResultsAverageObject.saveAverageErrorData(np.mean(mseTestingResults), np.mean(ssimTestingResults), np.mean(distortTestingResults))

            #Directory setup for individual images final save path
            ImagesFinalSavePath = ImagesSavePath + 'StopTrainPerc_' + str(StopPercentageSLADS) + '_StopTestPerc_' + str(StopPercentageTestingSLADS) + os.path.sep
            if os.path.exists(ImagesFinalSavePath): 
                shutil.rmtree(ImagesFinalSavePath)
            os.makedirs(ImagesFinalSavePath)

            #Obtain filenames for each of the images that were made
            dataFileNames = glob.glob(testingFeaturesPath + "/*"+ImageExtension) 
            for j in range(0, len(dataFileNames)): #For each of the files
                outputName = dataFileNames[j] #Grab the filename
                outputName = re.sub(testingFeaturesPath, '', outputName) #Remove the directory prefix
                shutil.move(dataFileNames[j], ImagesFinalSavePath+outputName) #Move them to an Images folder

        #Plot the average DMs across the different testing sampling percentages given a particular training sampling percentage
        testingSpecificStatisticsSavePathSuffix ='_StopTrainPerc_' + str(StopPercentageSLADS)
        testingSpecificStatisticsSavePath = testingStatisticsSavePath + 'StopTrainPerc_' + str(StopPercentageSLADS) + os.path.sep    
        
        #Directory setup for specific testing statistics
        if os.path.exists(testingSpecificStatisticsSavePath): 
            shutil.rmtree(testingSpecificStatisticsSavePath)
        os.makedirs(testingSpecificStatisticsSavePath)      
        
        plotErrorData(testingSpecificStatisticsSavePath, testingSpecificStatisticsSavePathSuffix, testingResultsAverageObject, StopPercentageTestingSLADSArr.tolist(), testingPlotTitle, plotXLabel) #Plot and save the error data obtained during 
        trainTestAverageErrors.append(testingResultsAverageObject)
        
    #Plot the average DMS together in a single plot for all of the training sampling percentages
    ttPlotAverageErrors(testingStatisticsSavePath, StopPercentageSLADSArr, StopPercentageTestingSLADSArr, trainTestAverageErrors)

#AFTER INTENDED PROCEDURES (TRAINING/TESTING) HAVE BEEN PERFORMED
print('\n\n\n' + ('#' * int(consoleColumns)))
print('PROGRAM COMPLETE')
print('#' * int(consoleColumns) + '\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




