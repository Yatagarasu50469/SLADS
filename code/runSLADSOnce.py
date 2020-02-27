#! /usr/bin/env python3

import sys
sys.path.append('code')
import numpy as np
from scipy.io import savemat
from skimage import filters
import pylab


from performMeasurements import perfromMeasurements
from performMeasurements import perfromInitialMeasurements
from updateERDandFindNewLocation import updateERDandFindNewLocationFirst
from updateERDandFindNewLocation import updateERDandFindNewLocationAfter
from computeStopCondRelated import computeStopCondFuncVal
from computeStopCondRelated import checkStopCondFuncThreshold
from performMeasurements import updateMeasurementArrays
from performReconOnce import performReconOnce
from loader import loadTestImage

def runSLADSSimulationOnce(Mask,CodePath,ImageSet,SizeImage,StopCondParams,Theta,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,SavePath,SimulationRun,ImNum,ImageExtension,PlotResult,Classify):
  
    MeasuredIdxs = np.transpose(np.where(Mask==1))
    UnMeasuredIdxs = np.transpose(np.where(Mask==0))
    
    ContinuousMeasuredValues = perfromInitialMeasurements(CodePath,ImageSet,ImNum,ImageExtension,Mask,SimulationRun)
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
    StopCondFuncVal=np.zeros(( int((SizeImage[0]*SizeImage[1])*(StopCondParams.MaxPercentage)/100)+10,2 ))
    while Stop !=1:
        
        if IterNum==0:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationFirst(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NumSamples,UpdateERDParams,BatchSamplingParams)           
        else:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationAfter(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,StopCondFuncVal,IterNum,NumSamples,NewIdxs,ReconValues,ReconImage,ERDValues,MaxIdxsVect)
    
        NewContinuousValues = perfromMeasurements(NewIdxs,CodePath,ImageSet,ImNum,ImageExtension,MeasuredIdxs,BatchSamplingParams,SimulationRun)
        ContinuousMeasuredValues = np.hstack((ContinuousMeasuredValues,NewContinuousValues))
        if Classify=='2C':           
            NewValues = NewContinuousValues > Threshold
            NewValues = NewValues+0
#        elif Classify=='MC':
            #### Classification function to output NewValues ##################
            # NewValues is the vector of measured values post classification            
        elif Classify=='N':
            NewValues=NewContinuousValues    


        Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs = updateMeasurementArrays(NewIdxs,MaxIdxsVect,Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,NewValues,BatchSamplingParams)
    
        NumSamples = np.shape(MeasuredValues)[0]
    
        StopCondFuncVal=computeStopCondFuncVal(ReconValues,MeasuredValues,StopCondParams,ImageType,StopCondFuncVal,MaxIdxsVect,NumSamples,IterNum,BatchSamplingParams)
            
        Stop = checkStopCondFuncThreshold(StopCondParams,StopCondFuncVal,NumSamples,IterNum,SizeImage)
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
        Difference,ReconImage = performReconOnce(SavePath,TrainingInfo,Resolution,SizeImage,ImageType,CodePath,ImageSet,ImNum,ImageExtension,SimulationRun,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues)
        TD = Difference/(SizeImage[0]*SizeImage[1])
        img=loadTestImage(CodePath,ImageSet,ImNum,ImageExtension,SimulationRun)  
        print('')
        print('')
        print('######################################')
        print('Total Distortion = ' + str(TD))
        
        from plotter import plotAfterSLADSSimulation  
        plotAfterSLADSSimulation(Mask,ReconImage,img)
        pylab.show()

        
def runSLADSOnce(Mask,CodePath,SizeImage,StopCondParams,Theta,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,SavePath,SimulationRun,ImNum,PlotResult,Classify):
  
    MeasuredIdxs = np.transpose(np.where(Mask==1))
    UnMeasuredIdxs = np.transpose(np.where(Mask==0))
    
    ##################################################################
    # CODE HERE
    # Plug in Your Measurement Routine
    # Please use 'MeasuredValues' as output variable
    # ContinuousMeasuredValues = perfromMeasurements(Mask)
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
    StopCondFuncVal=np.zeros(( int((SizeImage[0]*SizeImage[1])*(StopCondParams.MaxPercentage)/100)+10,2 ))
    while Stop !=1:
        
        if IterNum==0:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationFirst(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NumSamples,UpdateERDParams,BatchSamplingParams)           
        else:
            Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect=updateERDandFindNewLocationAfter(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,StopCondFuncVal,IterNum,NumSamples,NewIdxs,ReconValues,ReconImage,ERDValues,MaxIdxsVect)
        
        ##################################################################
        # CODE HERE
        # Plug in Your Measurement Routine
        # Please use 'NewContValues' as output variable
        # NewContinuousValues = perfromMeasurements(NewIdxs)
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


        Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs = updateMeasurementArrays(NewIdxs,MaxIdxsVect,Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,NewValues,BatchSamplingParams)
    
        NumSamples = np.shape(MeasuredValues)[0]
    
        StopCondFuncVal=computeStopCondFuncVal(ReconValues,MeasuredValues,StopCondParams,ImageType,StopCondFuncVal,MaxIdxsVect,NumSamples,IterNum,BatchSamplingParams)
            
        Stop = checkStopCondFuncThreshold(StopCondParams,StopCondFuncVal,NumSamples,IterNum,SizeImage)
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
        
        from plotter import plotAfterSLADS  
        plotAfterSLADS(Mask,ReconImage)
        pylab.show()
        