#====================================================================
#SLADS CONFIGURATION
#====================================================================

#PARAMETERS: L0
#==================================================================
#Is training of a model to be performed
trainingModel = True

#Is testing of a model to be performed
testingModel = True

#Should leave-one-out Cross Validation be performed instead of testing
#Not implemented at this time
LOOCV = False

#Is this an implementation run
impModel = False

#PARAMETERS: L1
#==================================================================

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
cValues = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

#Should animations be generated during testing/implementation
animationGen = True

#Type of Images: D - for discrete (binary) image; C - for continuous
#Not implemented at this time
imageType = 'C'

#Should a stopping threshold be found that corresponds to the best determined c value
#Not implemented at this time
findStopThresh = False

#How many masks should be used for each percentage during training
numMasks = 1

#Percent of reduction in distrotion limit for numRandomChoice determination
percOfRD = 50

#Running in a console/True, jupyter-notebook/False
consoleRunning = True
