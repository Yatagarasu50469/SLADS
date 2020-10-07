#====================================================================
#CONFIGURATION
#====================================================================
#==================================================================

##################################################################
#PARAMETERS: L0
#TASKS TO BE PERFORMED
##################################################################

#Is training of a model to be performed
trainingModel = True

#If trainingModel, should an existing database/cValue in RESULTS be loaded instad of creating a new one
loadTrainingDataset = True

#Is testing of a model to be performed
testingModel = True

#Is this an implementation run
impModel = False

#If an implementation run, what name should be used for data obtained
impSampleName = 'SAMPLE_1'

##################################################################


##################################################################
#PARAMETERS: L1
#TASK METHODS
##################################################################

#Which scanning method shoud be used: pointwise or linewise
scanMethod = 'pointwise'

#Stopping percentage for number of acquired pixels
stopPerc = 40

#==================================================================
#PARAMETERS: L1-0
#POINTWISE OPTIONS
#==================================================================

#What percentage of points should be initially acquired (random)
initialPercToScan = 1

#What percentage of points should be scanned each iteration
percToScan = 1

#==================================================================

#==================================================================
#PARAMETERS: L1-1
#LINEWISE OPTIONS
#==================================================================

#If linewise, should partial segments of a line be scanned (True), or between start/end points (False)
partialLineFlag = True

#Should lines be allowed to be revisited
lineRevistFlag = False

#What method should be used for point selection in a line: (percLine, 50% of line) (meanThreshold, mean of chosen line ERD) (none, full lines)
lineMethod = 'percLine'

#==================================================================

#==================================================================
#PARAMETERS: L1-2
#TRAINING DATA GENERATION
#==================================================================

#Sampling percentages for training, always includes initial measurement percentage
measurementPercs = np.arange(1,41).tolist()

#Possible c values for RD approximation
cValues = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])

#How many masks should be used for each percentage during training
numMasks = 1

#==================================================================

##################################################################


##################################################################
#PARAMETERS: L2
#MODEL PARAMETERS
##################################################################

#Which model should be used for training: cnn, unet, or rbdn
modelDef = 'cnn'

#How many filters should be used (doubles by layer in unet/rbdn, constant in cnn)
numStartFilters = 16

#What should the learning rate of the model's optimizer be
learningRate = 1e-3

#What should the batch size for pushing data through the network be
batchSize = 1

#How many epochs should a model train for at maximum
numEpochs = 1000

#Should the model training be cutoff early if no improvement is seen, using patience criteria
earlyCutoff = True

#How many epochs should the model training wait to see an improvement before terminating
maxPatience = 50

#How many epochs at minimum should be performed before starting to save the current best model and consider termination
minimumEpochs = 50

#What percentage of the training data should be used for training
#Early stopping is only functional with at least 1 validation sample
#If 1.0, will terminate at minimumEpochs+maxPatience
#If 1.0 set maxPatience to zero for model to save when intended
trainingSplit = 0.8

#Should visualizations of the training progression be generated
trainingProgressionVisuals = True

#If visualizations of the training progression are to be generated, how often should this occur
trainingVizSteps = 10

##################################################################


##################################################################
#PARAMETERS: L4
#GENERALLY NOT CHANGED
##################################################################

#Should the training data be visualized and saved
trainingDataPlot = True

#Should existing results folders not be allowed to be overwritten?
preventResultsOverwrite = False

#If consistency in the random generator is desired for inter-code comparisons
consistentSeed = True

#Should animations be generated during testing/implementation
animationGen = True

#Running in a console/True, jupyter-notebook/False
consoleRunning = True

#Define precision of the percentage averaging (as percentage is inconsistent between acquistion steps)
precision = 0.001

##################################################################


##################################################################
#PARAMETERS: L4
#DO NOT CHANGE - ALTERNATE OPTIONS NOT CURRENTLY FUNCTIONAL
##################################################################

#Is LOOCV to be performed
LOOCV = False

##################################################################


##################################################################
#PARAMETERS: L5
#DEBUG/DEPRECATED - WILL MOST LIKELY BE REMOVED IN FUTURE
##################################################################

#Explicitly state greater than or equal to the number of convolutional layers/branch in the network architecture
numConvolutionLayers = 1

#How training masks should be generated, randomly and/or using RD after initial at each measurement percentage
randomTrainingMasks = True
incrementalTrainingMasks = False

##################################################################
