#! /usr/bin/env python3
import numpy as np
import sys
sys.path.append('code')
import warnings
warnings.filterwarnings("ignore")

###############################################################################
############## USER INPUTS: L-0 ###############################################
###############################################################################

# Type of Image: D - for discrete (classified) image; C - for continuous
ImageType = 'D'

# Image extention
ImageExtension = '.png'

# if TrainingDB_X used enter 'X'             
TrainingImageSet = '1'

# Image resolution in pixels    
SizeImage = [64,64]

# Sweep range for c (to select best c for RD approximation)
c_vec = np.array([2,4,8,16,32])

# Stopping percentage for SLADS (to select C)
# Suggested: (64x64):50, (128x128):30, (256x256):20, (512x512):10
StopPercentageSLADS = 50

# Find threshold for stopping function Y/N
FindStopThresh = 'N'
# If 'Y', set the DesiredTD in L-1


###############################################################################
############## USER INPUTS: L-1 ###############################################
###############################################################################

# Sampling mask measurement percentages for training (best left unchanged)
MeasurementPercentageVector = np.array([5,10,20,40,80])

# Window size for approximate RD summation (best left unchanged)
WindowSize = [15,15]

# Update ERD or compute full ERD in SLADS (to find best c)
# with Update ERD, ERD only updated for a window surrounding new measurement
Update_ERD = 'Y' 
# Smallest ERD update window size permitted
MinWindSize = 3  
# Largest ERD update window size permitted  
MaxWindSize = 10       

# Initial Mask for SLADS (to find best c):
# Percentage of samples in initial mask
PercentageInitialMask = 1
# Type of initial mask   
MaskType = 'H'                   
# Choices: 
    # 'U': Uniform mask; can choose any percentage
    # 'R': Randomly distributed mask; can choose any percentage
    # 'H': low-dsicrepacy mask; can only choose 1% mask

# Desired total distortion (TD) value (to find threshold on stopping function)
DesiredTD=0
# TD = D(X,\hat(X))/(Number of pixels in image)
# D(X,\hat(X)) is difference between actual image X and reconstructed image 
# \hat(X) (summed over all pixels)
# For ImageType 'D' in range [0-1] for ImageType 'C' in range [0-max value]

###############################################################################
############################ END USER INPUTS ##################################
###############################################################################
NumReconsSLADS=10
PercOfRD= 50
from runTrainingScript import runTrainingScript
runTrainingScript(ImageType,ImageExtension,TrainingImageSet,SizeImage,c_vec,StopPercentageSLADS,FindStopThresh,MeasurementPercentageVector,WindowSize,Update_ERD,MinWindSize,MaxWindSize,PercentageInitialMask,MaskType,DesiredTD,NumReconsSLADS,PercOfRD)                  
