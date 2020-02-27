#! /usr/bin/env python3

import numpy as np


def generateInitialMask(InitialMask,SizeImage):
    if InitialMask.MaskType =='R':
        Mask = np.zeros((SizeImage[0],SizeImage[1]))
        UnifMatrix = np.random.rand(SizeImage[0],SizeImage[1])
        Mask = UnifMatrix<(InitialMask.Percentage/100)
    elif InitialMask.MaskType =='U':
        Mask = np.zeros((SizeImage[0],SizeImage[1]))
        ModVal = int(100/InitialMask.Percentage)
        for r in range(0,SizeImage[0]):
            for s in range(0,SizeImage[1]): 
                LinIdx = r*SizeImage[1]+s
                if np.remainder(LinIdx,ModVal)==0:
                    Mask[r][s]=1
    return Mask
        
        