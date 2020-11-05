#====================================================================
#SLADS CONFIGURATION
#====================================================================

#PARAMETERS: L0
#==================================================================
#Is training of a model to be performed
trainingModel = True

#Is testing of a model to be performed
testingModel = True

#Is this an implementation run
impModel = False

#PARAMETERS: L1
#==================================================================

#What percentage of unmeasured idxs should be scanned in training
percOfRD = 50

#Define precision of the percentage averaging (as percentage is inconsistent between acquistion steps)
precision = 0.001

#If consistentcy in the random generator is desired for comparisons
consistentSeed = True

#Should existing results folders not be allowed to be overwritten?
preventResultsOverwrite = False

#Which regression model should be used: LS, or SLADS-Net NN
regModel = 'LS'

#Which scanning method shoud be used: pointwise or linewise
scanMethod = 'linewise'

#If linewise, should partial segments of a line be scanned
partialLineFlag = True

#What method should be used for linewise point selection: (percLine, x% of line) (meanThreshold, mean of chosen line ERD) (none, full lines)
lineMethod = 'percLine'

#Window size for approximate RD summation; 15 for 64x64, (width,height)
windowSize = [15,15]

#Stopping percentage for number of acquired pixels
stopPerc = 40

#What name should be used for sample data obtained with impModel
impSampleName = 'SAMPLE_1'

#PARAMETERS: L2
#==================================================================

#Sampling percentages for training
measurementPercs = np.arange(1,41).tolist()

#Possible c values for RD approximation
cValues = np.array([1, 2, 4, 8, 16, 32, 64, 128])

#Should animations be generated during testing/implementation
animationGen = True

#Should lines be allowed to be revisited
lineRevistFlag = False

#Percent free RAM to allocate pool; leave enough free for results overhead
percRAM = 90

#Number of processor threads to leave free
numFreeThreads = 2

#How many masks should be used for each percentage during training
numMasks = 1

#Running in a console/True, jupyter-notebook/False
consoleRunning = True

#PARAMETERS: L3
#DO NOT CHANGE - ALTERNATE OPTIONS NOT CURRENTLY FUNCTIONAL
#==================================================================

#How should the mz visualizations be weighted: 'equal'
mzWeighting='equal'

#Is LOOCV to be performed
LOOCV = False

#DEBUG/DEPRECATED PARAMETERS: L4
#DO NOT CHANGE - NO LONGER NEEDED/USED
#Will most likely be removed in a future update
#==================================================================

#Type of Images: D - for discrete (binary) image; C - for continuous
#'discrete' option not implemented at this time
imageType = 'C'

#Should a stopping threshold be found that corresponds to the best determined c value
#Not implemented at this time
findStopThresh = False
