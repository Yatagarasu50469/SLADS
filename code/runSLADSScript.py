#! /usr/bin/env python3
import sys
import os
import numpy as np
from scipy.io import savemat


def runSLADSScript(FolderName,ImageType,TrainingImageSet,SizeImage,c,StoppingPercentage,StoppingThrehsold,Classify,PercentageInitialMask,MaskType,BatchSample,PlotResult,NumSamplesPerIter,Update_ERD,MinWindSize,MaxWindSize):

    from variableDefinitions import TrainingInfo
    from variableDefinitions import InitialMask
    from variableDefinitions import StopCondParams
    from variableDefinitions import UpdateERDParams
    from variableDefinitions import BatchSamplingParams
    from runSLADSOnce import runSLADSOnce     
    from loader import loadOrGenerateInitialMask
    from computeStopCondRelated import computeBeta
    
    c = float(c);StoppingPercentage = float(StoppingPercentage);PercentageInitialMask = float(PercentageInitialMask)
             
    Resolution = 1
    CodePath = '.' + os.path.sep
    SavePath = CodePath + 'ResultsAndData' + os.path.sep + 'SLADSSimulationResults' + os.path.sep + FolderName + os.path.sep
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)                          
    InitialMask=InitialMask()
    InitialMask.initialize(SizeImage[0],SizeImage[1],MaskType,1,PercentageInitialMask)
    
    Beta = computeBeta(SizeImage)
    StopCondParams=StopCondParams()
    StopCondParams.initialize(Beta,StoppingThrehsold,50,2,StoppingPercentage)
    
    UpdateERDParams = UpdateERDParams()
    UpdateERDParams.initialize(Update_ERD,MinWindSize,MaxWindSize,1.5)
    
    BatchSamplingParams = BatchSamplingParams()
    if BatchSample=='N':
        BatchSamplingParams.initialize(BatchSample,1)
    else:
        BatchSamplingParams.initialize(BatchSample,NumSamplesPerIter)
    
    # Training image information to load Theta
    TrainingInfo = TrainingInfo()
    if ImageType == 'D':
        TrainingInfo.initialize('DWM','DWM',2,10,'Gaussian',c,0.25,15)
    elif ImageType == 'C':
        TrainingInfo.initialize('CWM','CWM',2,10,'Gaussian',c,0.25,15)
    
    TrainingDBName = 'TrainingDB_' + TrainingImageSet
    TrainingDBPath = 'ResultsAndData' + os.path.sep + 'TrainingSavedFeatures' + os.path.sep + TrainingDBName
    if not os.path.exists(TrainingDBPath):                                                                                                                          
        sys.exit('Error!!! The folder ' + TrainingDBPath + ' does not exist. Check entry for ' + TrainingImageSet)
    
    #Load Theta
    ThetaLoadPath = 'ResultsAndData' + os.path.sep + 'TrainingSavedFeatures' + os.path.sep + TrainingDBName + os.path.sep + 'c_' + str(c) + os.path.sep
    if not os.path.exists(ThetaLoadPath):                                                                                                                          
        sys.exit('Error!!! Check folder ./ResultsAndData/TrainingSavedFeatures/TrainingDB_' + TrainingImageSet + ' for folder c_' + str(c))                                                                                                                                  
    Theta=np.transpose(np.load(ThetaLoadPath +'Theta.npy'))
    
    # Load initial measurement mask
    loadPathInitialMask = CodePath + 'ResultsAndData' + os.path.sep + 'InitialSamplingMasks'
    Mask = loadOrGenerateInitialMask(loadPathInitialMask,MaskType,InitialMask,SizeImage)
    np.save(SavePath + 'InitialMask', Mask)
    savemat(SavePath + 'InitialMask.mat',dict(Mask=Mask))
    
    # Run SLADS simulation once
    SimulationRun = 1
    ImNum = 1
    runSLADSOnce(Mask,CodePath,SizeImage,StopCondParams,Theta,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,SavePath,SimulationRun,ImNum,PlotResult,Classify)                               
