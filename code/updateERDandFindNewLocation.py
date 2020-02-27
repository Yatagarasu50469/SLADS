#! /usr/bin/env python3
import numpy as np
from computeOrupdateERD import computeFullERD
from computeOrupdateERD import updateERD
from performMeasurements import findNewMeasurementIdxs



def updateERDandFindNewLocationFirst(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NumSamples,UpdateERDParams,BatchSamplingParams):

    ERDValues,ReconValues,ReconImage = computeFullERD(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType)

    NewIdxs,MaxIdxsVect = findNewMeasurementIdxs(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NumSamples,UpdateERDParams,ReconValues,ReconImage,ERDValues,BatchSamplingParams)

    return(Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect)



def updateERDandFindNewLocationAfter(Mask,MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType,UpdateERDParams,BatchSamplingParams,StopCondFuncVal,IterNum,NumSamples,NewIdxs,ReconValues,ReconImage,ERDValues,MaxIdxsVect):
    
    if UpdateERDParams.Do == 'N':
        ERDValues,ReconValues,ReconImage = computeFullERD(MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,Theta,SizeImage,TrainingInfo,Resolution,ImageType)
    else:
        ERDValues,ReconValues=updateERD(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NewIdxs,NumSamples,UpdateERDParams,ReconValues,ReconImage,ERDValues,MaxIdxsVect,BatchSamplingParams)
    
    NewIdxs,MaxIdxsVect = findNewMeasurementIdxs(Mask,MeasuredIdxs,UnMeasuredIdxs,MeasuredValues,Theta,SizeImage,TrainingInfo,Resolution,ImageType,NumSamples,UpdateERDParams,ReconValues,ReconImage,ERDValues,BatchSamplingParams)

    return(Mask,MeasuredValues,ERDValues,ReconValues,ReconImage,NewIdxs,MaxIdxsVect)