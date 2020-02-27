#! /usr/bin/env python3
import numpy as np
from computeDifference import computeDifference

def computeStopCondFuncVal(ReconValues,MeasuredValues,StopCondParams,ImageType,StopCondFuncVal,MaxIdxsVect,NumSamples,IterNum,BatchSamplingParams):
    
    if BatchSamplingParams.Do=='N':
        Diff=computeDifference(ReconValues[MaxIdxsVect],MeasuredValues[NumSamples-1],ImageType)
        if IterNum == 0:
            StopCondFuncVal[IterNum,0] = StopCondParams.Beta*Diff
        else:
            StopCondFuncVal[IterNum,0] = ((1-StopCondParams.Beta)*StopCondFuncVal[IterNum-1,0] + StopCondParams.Beta*Diff)
        StopCondFuncVal[IterNum,1] = NumSamples
    
    else:
        Diff=0
        for i in range(0,BatchSamplingParams.NumSamplesPerIter):
            Diff=computeDifference(ReconValues[MaxIdxsVect[i]],MeasuredValues[NumSamples-1-(BatchSamplingParams.NumSamplesPerIter-i-1)],ImageType)+Diff
        Diff = Diff/BatchSamplingParams.NumSamplesPerIter
        if IterNum == 0:
            StopCondFuncVal[IterNum,0] = StopCondParams.Beta*Diff
        else:
            StopCondFuncVal[IterNum,0] = ((1-StopCondParams.Beta)*StopCondFuncVal[IterNum-1,0] + StopCondParams.Beta*Diff)
        StopCondFuncVal[IterNum,1] = NumSamples
    return StopCondFuncVal

def checkStopCondFuncThreshold(StopCondParams,StopCondFuncVal,NumSamples,IterNum,SizeImage):
    
    if StopCondParams.Threshold==0:
        if NumSamples>SizeImage[0]*SizeImage[1]*StopCondParams.MaxPercentage/100:
            Stop=1
        else:
            Stop=0

    else:
        if NumSamples>SizeImage[0]*SizeImage[1]*StopCondParams.MaxPercentage/100:
            Stop=1
        else:
            if np.logical_and(((SizeImage[0]*SizeImage[1])*StopCondParams.MinPercentage/100)<NumSamples,StopCondFuncVal[IterNum,0]<StopCondParams.Threshold):
                Stop=0
                GradStopCondFunc =np.mean(StopCondFuncVal[IterNum,0]-StopCondFuncVal[IterNum-StopCondParams.JforGradient:IterNum-1,0])
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