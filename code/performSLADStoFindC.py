#! /usr/bin/env python3
import sys
import os
sys.path.append('code')
import numpy as np
import glob
from runSLADSOnce import runSLADSSimulationOnce
from performReconOnce import performReconOnce
from loader import loadOrGenerateInitialMask

def performSLADStoFindC(CodePath,TrainingDataPath,ImageSet,ImageType,ImageExtension,TrainingInfo,SizeImage,StopPercentage,Resolution,c_vec,UpdateERDParams,InitialMask,MaskType,reconPercVector,Classify):
    SimulationRun = 0   
    
    from variableDefinitions import StopCondParams
    from variableDefinitions import BatchSamplingParams
    from computeStopCondRelated import computeBeta
    # Initialize stopping condition variable

    Beta = computeBeta(SizeImage)
    StopCondParams=StopCondParams()
    StopCondParams.initialize(Beta,0,50,2,StopPercentage)
    
    SavePathSLADS = TrainingDataPath + 'SLADSResults' 
    PlotResult = 'N'

    # Batch Sampling
    PercOfSamplesPerIter = 0
    NumSamplesPerIter = int(PercOfSamplesPerIter*SizeImage[0]*SizeImage[1]/100)
    BatchSample = 'N'
    BatchSamplingParams = BatchSamplingParams()
    if BatchSample=='N':
        BatchSamplingParams.initialize(BatchSample,1)
    else:
        BatchSamplingParams.initialize(BatchSample,NumSamplesPerIter)

    
    
    if not os.path.exists(SavePathSLADS):
        os.makedirs(SavePathSLADS)
        
    AreaUnderCurve = np.zeros(c_vec.shape[0])
    Idx_c = 0
    
    for c in c_vec:
        LoadPath_c = TrainingDataPath + os.path.sep + 'c_' + str(c)
        TrainingInfo.FilterC=c
        Theta = np.load(LoadPath_c + os.path.sep + 'Theta.npy')
        
        
        loadPathImage = TrainingDataPath + 'ImagesToFindC' + os.path.sep   
        NumImages = np.size(glob.glob(loadPathImage + '*' + ImageExtension))
        for ImNum in range(1,NumImages+1):
        
            SavePathSLADS_c_ImNum = SavePathSLADS +  os.path.sep + 'Image_' + str(ImNum) + '_c_'+ str(c)
            if not os.path.exists(SavePathSLADS_c_ImNum):
                os.makedirs(SavePathSLADS_c_ImNum) 
            # Load initial measurement mask
            loadPathInitialMask = CodePath + 'ResultsAndData' + os.path.sep + 'InitialSamplingMasks'
            Mask = loadOrGenerateInitialMask(loadPathInitialMask,MaskType,InitialMask,SizeImage)
            SavePath = SavePathSLADS + os.path.sep + 'Image_' + str(ImNum) + '_c_'+ str(c) + os.path.sep
                                    
            runSLADSSimulationOnce(Mask,CodePath,ImageSet,SizeImage,StopCondParams,Theta,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,SavePath,SimulationRun,ImNum,ImageExtension,PlotResult,Classify)
            MeasuredValuesFull=np.load(SavePath + 'MeasuredValues.npy')
            MeasuredIdxsFull=np.load(SavePath + 'MeasuredIdxs.npy')
            UnMeasuredIdxsFull=np.load(SavePath + 'UnMeasuredIdxs.npy')    
            Difference = np.zeros(reconPercVector.shape[0])
            idx=0
            for p in reconPercVector:
                NumMeasurements = int(p*SizeImage[0]*SizeImage[1]/100)
                MeasuredValues = MeasuredValuesFull[0:NumMeasurements]
                MeasuredIdxs = MeasuredIdxsFull[0:NumMeasurements][:]
                
                temp1 = MeasuredIdxsFull[NumMeasurements+1:MeasuredValuesFull.shape[0]][:]
                temp2 = UnMeasuredIdxsFull
                UnMeasuredIdxs = np.concatenate((temp1, temp2), axis=0)

                Difference[idx],ReconImage = performReconOnce(SavePath,TrainingInfo,Resolution,SizeImage,ImageType,CodePath,ImageSet,ImNum,ImageExtension,SimulationRun,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues)
                idx = idx+1
            TD = Difference/(SizeImage[0]*SizeImage[1])
            np.save(SavePath + 'TD', TD)
            AreaUnderCurve[Idx_c]=AreaUnderCurve[Idx_c]+np.trapz(TD,x=reconPercVector)
        print('SLADS complete for c = ' + str(c))
        
        Idx_c = Idx_c +1
        
    Best_c = c_vec[np.argmin(AreaUnderCurve)]
    return Best_c,NumImages
      
            