#! /usr/bin/env python3
import os
import numpy as np
import sys
def runTrainingScript(ImageType,ImageExtension,TrainingImageSet,SizeImage,c_vec,StopPercentageSLADS,FindStopThresh,MeasurementPercentageVector,WindowSize,Update_ERD,MinWindSize,MaxWindSize,PercentageInitialMask,MaskType,DesiredTD,NumReconsSLADS,PercOfRD):
    
    from variableDefinitions import TrainingInfo
    from variableDefinitions import UpdateERDParams
    from variableDefinitions import InitialMask
    from performTraining import performTraining
    from performSLADStoFindC import performSLADStoFindC
    from findStoppingThreshold import findStoppingThreshold
    Classify = 'N'
    Resolution = 1
    FolderName = 'TrainingDB_'
    c_vec=c_vec.astype(float);StopPercentageSLADS = float(StopPercentageSLADS);PercentageInitialMask = float(PercentageInitialMask);MeasurementPercentageVector=MeasurementPercentageVector.astype(float)    
    reconPercVector = np.linspace(PercentageInitialMask, StopPercentageSLADS, num=NumReconsSLADS*(StopPercentageSLADS-PercentageInitialMask), endpoint=False)
    CodePath = '.' + os.path.sep
    TrainingDataPath = CodePath + 'ResultsAndData' + os.path.sep + 'TrainingData' + os.path.sep + FolderName + str(TrainingImageSet) + os.path.sep
    if not os.path.exists(TrainingDataPath):                                                                                                                          
        sys.exit('Error!!! The folder ' + TrainingDataPath + ' does not exist. Check entry for ' + TrainingImageSet)
    if not os.path.exists(TrainingDataPath + os.path.sep + 'Images'):                                                                                                                      
        sys.exit('Error!!! The folder ' + TrainingDataPath + os.path.sep + 'Images does not exist')      
    if not os.path.exists(TrainingDataPath + os.path.sep + 'ImagesToFindC'):                                                                                                                          
        sys.exit('Error!!! The folder ' + TrainingDataPath + os.path.sep + 'ImagesToFindC does not exist')                                                                                                      
                                                                                                                      
    TrainingInfo = TrainingInfo()
    if ImageType == 'D':
        TrainingInfo.initialize('DWM','DWM',2,10,'Gaussian',0,0.25,15)
    elif ImageType == 'C':
        TrainingInfo.initialize('CWM','CWM',2,10,'Gaussian',0,0.25,15)
        
    # Perform training 
    performTraining(MeasurementPercentageVector,TrainingDataPath,ImageType,ImageExtension,SizeImage,TrainingInfo,Resolution,WindowSize,c_vec,PercOfRD)
    
    UpdateERDParams = UpdateERDParams()
    UpdateERDParams.initialize(Update_ERD,MinWindSize,MaxWindSize,1.5)
    
    InitialMask=InitialMask()
    InitialMask.initialize(SizeImage[0],SizeImage[1],MaskType,1,PercentageInitialMask)
      
    # Find the best value of c
    Best_c,NumImagesForSLADS = performSLADStoFindC(CodePath,TrainingDataPath,TrainingImageSet,ImageType,ImageExtension,TrainingInfo,SizeImage,StopPercentageSLADS,Resolution,c_vec,UpdateERDParams,InitialMask,MaskType,reconPercVector,Classify)
    print('The best c was found to be: c = ' + str(Best_c))
    
    TrainingDBName = 'TrainingDB_' + TrainingImageSet
    ThetaSavePath = 'ResultsAndData' + os.path.sep + 'TrainingSavedFeatures' + os.path.sep + TrainingDBName + os.path.sep + 'c_' + str(Best_c) + os.path.sep
    
    print('The selected regression coefficients are saved in ' + ThetaSavePath + ' as Theta.npy' )
    
    if not os.path.exists(ThetaSavePath):
        os.makedirs(ThetaSavePath)
        
    ThetaLoadPath = 'ResultsAndData' + os.path.sep + 'TrainingData' + os.path.sep + TrainingDBName + os.path.sep + 'c_' + str(Best_c) + os.path.sep
    Theta = np.load(ThetaLoadPath + 'Theta.npy')    
    np.save(ThetaSavePath + 'Theta', Theta)
    
    
    # Find the Threshold on stopping condition that corresponds to the desired total distortion (TD) value set above
    if FindStopThresh=='Y':   
        Threshold=findStoppingThreshold(TrainingDataPath,NumImagesForSLADS,Best_c,PercentageInitialMask,DesiredTD,reconPercVector,SizeImage)
        print('For a TD of '+ str(DesiredTD) + ' set stopping function threshold to: ' + str(Threshold))
        print('**** Make sure to enter this value in runSimulations.py and in runSLADS.py')
        print('The threshold value is saved in:  ' + ThetaSavePath + ' as Threshold.npy')
        np.save(ThetaSavePath + 'Threshold', Threshold) 
