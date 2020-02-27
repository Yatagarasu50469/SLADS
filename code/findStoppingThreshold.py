#! /usr/bin/env python3
import sys
import os
sys.path.append('code')
import numpy as np

def findStoppingThreshold(TrainingDataPath,NumTrainingImages,Best_c,PercentageInitialMask,DesiredTD,reconPercVector,SizeImage):
    SavePathSLADS = TrainingDataPath + 'SLADSResults' 
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
    
    return Threshold
                

