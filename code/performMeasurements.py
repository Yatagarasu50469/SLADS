#! /usr/bin/env python3
import sys
import numpy as np
from loader import loadTestImage
from computeOrupdateERD import updateERD
from variableDefinitions import BatchSamplingParams


def perfromInitialMeasurements(CodePath,TestingImageSet,ImNum,ImageExtension,Mask,SimulationRun):
    
    Img = loadTestImage(CodePath,TestingImageSet,ImNum,ImageExtension,SimulationRun)
    if Mask.shape[0]!=Img.shape[0] or Mask.shape[1]!=Img.shape[1]:
        sys.exit('Error!!! The dimensions you entered in "SizeImage" do not match the dimensions of the testing image in ./ResultsAndData/TestingImages/TestingImageSet_' + TestingImageSet)
    MeasuredValues = Img[Mask==1]
    return(MeasuredValues)



def perfromMeasurements(NewIdxs,CodePath,TestingImageSet,ImNum,ImageExtension,MeasuredIdxs,BatchSamplingParams,SimulationRun):
    Img = loadTestImage(CodePath,TestingImageSet,ImNum,ImageExtension,SimulationRun)
    if BatchSamplingParams.Do == 'N':
        NewValues = Img[NewIdxs[0],NewIdxs[1]]
    else:
        NewValues = Img[NewIdxs[:,0],NewIdxs[:,1]]
    return NewValues

def updateMeasurementArrays(NewIdxs,MaxIdxsVect,Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,NewValues,BatchSamplingParams):
    
    if BatchSamplingParams.Do == 'N':
        Mask[NewIdxs[0],NewIdxs[1]]=1
        MeasuredValues = np.hstack((MeasuredValues,NewValues))
        MeasuredIdxs = np.vstack((MeasuredIdxs,[NewIdxs[0],NewIdxs[1]]))
        UnMeasuredIdxs = np.delete(UnMeasuredIdxs,(MaxIdxsVect), axis=0)
    else:
        for i in range(0,BatchSamplingParams.NumSamplesPerIter):
            Mask[NewIdxs[i,0],NewIdxs[i,1]]=1
        MeasuredValues = np.hstack((MeasuredValues,NewValues))
        MeasuredIdxs = np.vstack((MeasuredIdxs,NewIdxs))
        UnMeasuredIdxs = np.delete(UnMeasuredIdxs,(MaxIdxsVect), axis=0)
    return(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs)



def findNewMeasurementIdxs(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NumSamples,UpdateERDParams,ReconValues,ReconImage,ERDValues,ActualBatchSamplingParams):
    
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
            
            ERDValues,ReconValues=updateERD(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfo,Resolution,ImageType,TempNewIdxs[i-1,:],NumSamples,UpdateERDParams,ReconValues,ReconImage,ERDValues,TempMaxIdxsVect[i-1],TempBatchSamplingParams)
            TempMaxIdxsVect[i] = np.argmax(ERDValues)
            TempNewIdxs[i,:] = UnMeasuredIdxs[TempMaxIdxsVect[i],:]
            NewIdxs[i,:] = TempNewIdxs[i,:]
            MaxIdxsVect[i]=np.where(np.all(OrigUnMeasuredIdxs==TempNewIdxs[i,:],axis=1))
    return(NewIdxs,MaxIdxsVect)