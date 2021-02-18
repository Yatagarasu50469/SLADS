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
loadTrainingDataset = False

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

#Which model should be used for ERD generation (SLADS-LS, SLADS-Net, DLADS)
#Recommend DLADS if GPU(s) available for training and SLADS-LS if not
#Trained models are not intercompatible between models!
erdModel = 'DLADS'

#Which scanning method shoud be used: pointwise or linewise
scanMethod = 'linewise'

#Stopping percentage for number of acquired pixels
stopPerc = 40

#Should distances be penalized according to aspect ratio
#True if data artifically stretched through interpolation (nano-DESI MSI: True)
asymPenalty = True

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

#What method should be used for linewise point selection: (startEndPoints, partial line segment) (percLine, Top stopPerc% ERD locations) (none, full line)
lineMethod = 'percLine'

#Should lines be allowed to be revisited
lineRevistMethod = False

#Should all lines be scanned at least once
lineVisitAll = True

#==================================================================

#==================================================================
#PARAMETERS: L1-2
#TRAINING DATA GENERATION
#==================================================================

#Sampling percentages for training, always includes initial measurement percentage
measurementPercs = np.arange(1,41).tolist()

#Possible c values for RD approximation
cValues = np.array([1, 2, 4, 8, 16, 32, 64, 128])

#How many masks should be used for each percentage during training
numMasks = 1

#==================================================================

##################################################################

##################################################################
#PARAMETERS: L2
#MODEL PARAMETERS
##################################################################

#How many neighbors should be used in the reconstruction estimation
numNeighbors = 10

#==================================================================
#PARAMETERS: L2-1
#SLADS(-Net) MODEL PARAMETERS (disregard if using DLADS)
#==================================================================
featDistCutoff = 0.25

#==================================================================

#==================================================================
#PARAMETERS: L2-2
#DLADS MODEL PARAMETERS (disregard if using SLADS-LS or SLADS-Net)
#==================================================================

#Which model should be used for training: cnn, or unet
modelDef = 'cnn'

#How many filters should be used (doubles by layer in unet, constant in cnn)
numStartFilters = 64

#What should the learning rate of the model's optimizer be
learningRate = 1e-4

#What should the batch size for pushing data through the network be
batchSize = 1

#How many epochs should a model train for at maximum
numEpochs = 1000

#Should the model training be cutoff early if no improvement is seen, using patience criteria
earlyCutoff = True

#How many epochs should the model training wait to see an improvement before terminating
maxPatience = 100

#How many epochs at minimum should be performed before starting to save the current best model and consider termination
minimumEpochs = 50

#What percentage of the training data should be used for training
#Early stopping is only functional with at least 1 validation sample
#If 1.0, will terminate at minimumEpochs+maxPatience; not recomended!
#If 1.0 set maxPatience to zero for model to save when intended
trainingSplit = 0.8

#Should visualizations of the training progression be generated
trainingProgressionVisuals = True

#If visualizations of the training progression are to be generated, how often should this occur
trainingVizSteps = 10

#==================================================================

##################################################################


##################################################################
#PARAMETERS: L3
#GENERALLY NOT CHANGED
##################################################################

#Which system GPU(s) are available; ('None', whichever is available; '-1', CPU only)
availableGPUs = 'None'

#Should the training data be visualized and saved; turning off will save notable time in training
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
#Nothing here in this release
##################################################################
