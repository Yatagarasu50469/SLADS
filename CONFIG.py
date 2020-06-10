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
#Which algorithm should be used for nearest neighbor: kd_tree or ball_tree
#ball-tree is more accurate but cannot be run with 50k per line
algorithmNN = 'ball_tree'

#Which scanning method shoud be used: pointwise or linewise
scanMethod = 'linewise'

#If linewise, should partial segments of a line be scanned
partialLineFlag = True

#Should the image be resized to its physical aspect ratio
physResize = False

#What method should be used for linewise point selection: (percLine, 50% of line) (meanThreshold, mean of chosen line ERD) (none, full lines)
lineMethod = 'percLine'

#Window size for approximate RD summation; 15 for 64x64, (width,height)
windowSize = [15,15]

#Stopping percentage for number of acquired pixels
stopPerc = 50

#Name used for sample data obtained with impModel
impSampleName = 'SAMPLE_1'

#Percent free RAM to allocate pool; leave enough free for results overhead
percRAM = 90

#Number of processor threads to leave free
numFreeThreads = 2

#PARAMETERS: L2
#==================================================================

#Sampling percentages for training
measurementPercs = [5, 10, 20, 30, 40]

#Possible c values for RD approximation
cValues = np.array([1, 2, 4, 8, 16, 32])

#Should lines be allowed to be revisited
lineRevistFlag = False

#Should animations be generated during testing/implementation
animationGen = True

#How many masks should be used for each percentage during training
numMasks = 1

#Running in a console/True, jupyter-notebook/False
consoleRunning = True

#PARAMETERS: L3
#DO NOT CHANGE - ALTERNATE OPTIONS NOT CURRENTLY FUNCTIONAL
#==================================================================

#Should leave-one-out Cross Validation be performed instead of testing
#Not implemented at this time
LOOCV = False

#How should the mz visualizations be weighted: 'equal'
mzWeighting='equal'

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


