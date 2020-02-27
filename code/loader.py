#! /usr/bin/env python3
from scipy.io import loadmat
import os
import sys

def loadTestImage(CodePath,ImageSet,ImNum,ImageExtension,SimulationRun):
    
    if SimulationRun==1:
        loadPathImage = CodePath + 'ResultsAndData' + os.path.sep + 'TestingImages' + os.path.sep + 'TestingImageSet_' + ImageSet + os.path.sep 
    else:
        loadPathImage = CodePath + 'ResultsAndData' + os.path.sep + 'TrainingData' + os.path.sep + 'TrainingDB_' + ImageSet + os.path.sep + 'ImagesToFindC' + os.path.sep     
    from scipy import misc
    import glob
    cnt = 1

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

def loadOrGenerateInitialMask(loadPathInitialMask,MaskType,InitialMask,SizeImage):
    import numpy as np
    from generateInitialMask import generateInitialMask
    if MaskType=='H':
        StartingMeasurementMask=InitialMask.MaskType + '_' + str(InitialMask.MaskNumber) + '_' + str(InitialMask.RowSz) + '_' + str(InitialMask.ColSz) + '_Percentage_' + str(InitialMask.Percentage);
        loadPathInitialMask = loadPathInitialMask + os.path.sep + StartingMeasurementMask                                                               
        if not os.path.exists(loadPathInitialMask):                                                                                                                          
            sys.exit('Error!!! Check foder .ResultsAndData/InitialSamplingMasks/ for folder ' + loadPathInitialMask)                                                            
        Mask = np.load(loadPathInitialMask + os.path.sep + 'SampleMatrix.npy')
    else:
        Mask = generateInitialMask(InitialMask,SizeImage)
    return Mask