#! /usr/bin/env python3
import sys
sys.path.append('code')
import warnings
warnings.filterwarnings("ignore")
###############################################################################
############## USER INPUTS: L-0 ###############################################
###############################################################################

# Name of folder to save result in    
FolderName = 'Example_1'

# Type of Image: D - for discrete (classified) image; C - for continuous
ImageType = 'D'

# If TrainingDB_Y was used for training enter 'Y'          
TrainingImageSet = '1'   

# Image resolution in pixels   
SizeImage = [64,64]

# Value of c found in training     
c=1

# Maximum  sampling percentage 
StoppingPercentage = 50
# If you want to use stopping function used, enter threshold (from Training), 
# else leave at 0        
StoppingThrehsold = 0    

# Clasification setting
Classify = 'N'              
# 'N' - no classification needed (already classified or continuous data)
# '2C' - perform 2 class classification using otsu
# 'MC' - perform multi-class classification
# if 'MC' open .code/runSLADSOnce.py and search "elif Classify=='MC':" and  
# include your classification method (in all four locations) as instructed
                            
# Initial Mask for SLADS:
# Percentage of samples in initial mask
PercentageInitialMask = 1
# Type of initial mask   
MaskType = 'H'                   
# Choices: 
    # 'U': Uniform mask; can choose any percentage
    # 'R': Randomly distributed mask; can choose any percentage
    # 'H': low-dsicrepacy mask; can only choose 1% mask

# Batch Sampling
BatchSample = 'N'           
# If 'Y' set number of samples in each step in L-1 (NumSamplesPerIter)

PlotResult='Y'

###############################################################################
############## USER INPUTS: L-1 ###############################################
###############################################################################     

# The number of samples in each step  
NumSamplesPerIter = 10

# Update ERD or compute full ERD in SLADS
# with Update ERD, ERD only updated for a window surrounding new measurement
Update_ERD = 'Y' 
# Smallest ERD update window size permitted
MinWindSize = 3  
# Largest ERD update window size permitted  
MaxWindSize = 10      

###############################################################################
############################ END USER INPUTS ##################################
############################################################################### 
from runSLADSScript import runSLADSScript
runSLADSScript(FolderName,ImageType,TrainingImageSet,SizeImage,c,StoppingPercentage,StoppingThrehsold,Classify,PercentageInitialMask,MaskType,BatchSample,PlotResult,NumSamplesPerIter,Update_ERD,MinWindSize,MaxWindSize)