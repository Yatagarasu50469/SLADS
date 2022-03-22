#====================================================================
#CONFIGURATION
#====================================================================
#==================================================================

##################################################################
#PARAMETERS: L0
#TASKS TO BE PERFORMED
##################################################################

#Is training of a model to be performed
trainingModel = False

#If trainingModel, should existing database/cValue in RESULTS be loaded instad of creating a new one
loadTrainValDatasets = False

#If validation dataset should be simulated
validationModel = False

#Is testing of a model to be performed
testingModel = False

#Is this an implementation run
impModel = False

#Is post-processing to be performed
postModel = False

##################################################################


##################################################################
#PARAMETERS: L1
#TASK METHODS
##################################################################

#Which model should be used for ERD generation (SLADS-LS, SLADS-Net, DLADS)
erdModel = 'DLADS'

#Which scanning method shoud be used: pointwise or linewise
scanMethod = 'linewise'

#Should only a single mz be used as the network input (allows evaluation over multiple, uses first mz in mz.csv local/global file)
mzSingle = False

#Should static window be used in RD generation
staticWindow = False

#If a static window is to be used, what size (symmetric) should it be ([15,15] for original SLADS)
staticWindowSize = 15

#If a dynamic window is to be used, what multiple of the sigma value should be used
dynWindowSigMult = 3

#==================================================================
#PARAMETERS: L1-0
#IMPLEMENTATION OPTIONS
#==================================================================

#If an implementation run, what name should be used for data obtained
impSampleName = 'SAMPLE_1'

#If an implementation run, where will the MSI files be located (location will be emptied on run); None equivalent to './RESULTS/IMP/'
impInputDir = None

#If the measurement times listed in the acquried MSI files do not start with 0 being at the left-side of the FOV
impOffset = True

#==================================================================

#==================================================================
#PARAMETERS: L1-1
#POINTWISE OPTIONS
#==================================================================

#What percentage of points should be initially acquired (random) during testing/implementation
initialPercToScan = 1

#Stopping percentage for number of acquired pixels during testing/implementation
stopPerc = 30

#If group-wise, what percentage of points should be acquired; otherwise set to None
percToScan = None

#What percentage of points should be acquired between visualization steps; if all steps should be, then set to None
percToViz = None

#==================================================================

#==================================================================
#PARAMETERS: L1-2
#LINEWISE OPTIONS
#==================================================================

#How should points be returned from a chosen line: (segLine; partial line segment) (percLine; top stopPerc% ERD locations) (none, full line)
lineMethod = 'segLine'

#How should individual points on a chosen line be selected: (single; one-by-one, updates ERD from reconstruction) (group; all chosen in one step)
linePointSelection = 'group'

#If using a segLine, how should the start and end points be determined (minPerc, left/right most of the top stopPerc ERD values) (otsu, left/right most of the foreground ERD found with Otsu)
segLineMethod = 'otsu'

#Should lines be allowed to be revisited
lineRevist = False

#Should all lines be scanned at least once
lineVisitAll = True

#Specify what line positions (percent height) should be used for initial acquistion
startLinePositions = [0.25, 0.50, 0.75]

#==================================================================

#==================================================================
#PARAMETERS: L1-3
#TRAINING DATA GENERATION
#==================================================================

#What percentage of points should be initially acquired (random) during training
initialPercToScanTrain = 1

#Stopping percentage for number of acquired pixels for training
stopPercTrain = 30

#Possible c values for RD approximation
#cValues = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
cValues = np.array([8])

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

#Which model should be used for training (unet)
modelDef = 'unet'

#How many filters should be used
numStartFilters = 64

#Which optimizer should be used('Nadam', 'Adam', or 'RMSProp')
optimizer = 'Nadam'

#Which loss function should be used for the optimizer ('MAE' or 'MSE')
lossFunc = 'MAE'

#What should the learning rate of the model's optimizer be
learningRate = 1e-5

#What should the batch size for pushing data through the network be
batchSize = 1

#How many epochs should a model train for at maximum
numEpochs = 1000

#How many tries should the model have to produce a non-nan result before quitting
maxTrainingAttempts = 5

#Should the model training be cutoff early if no improvement is seen, using patience criteria
earlyCutoff = True

#How many epochs should the model training wait to see an improvement before terminating
maxPatience = 50

#How many epochs at minimum should be performed before starting to save the current best model and consider termination
minimumEpochs = 10

#What percentage of the training data should be used for training
#Early stopping is only functional with at least 1 validation sample
#If 1.0, will terminate at minimumEpochs+maxPatience; not functional for DLADS!
#If 1.0 set maxPatience to zero for model to save when intended
trainingSplit = 0.8

#Should visualizations of the training progression be generated
trainingProgressionVisuals = True

#If visualizations of the training progression are to be generated, how often (epochs) should this occur
trainingVizSteps = 10

#==================================================================

##################################################################


##################################################################
#PARAMETERS: L3
#GENERALLY NOT CHANGED
##################################################################

#Should output visualizations be generated during acquisition? (Not recommended for simulation)
liveOutputFlag = False

#Should parallelization calls be used (True); if memory overflow issues develop, set to False
parallelization = True

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

#Define precision of the percentage averaging (as percentage is inconsistent between acquistion steps)
precision = 0.001

#Should mz images be shown with log normalized colorbars 
sysLogNorm = False

##################################################################


##################################################################
#PARAMETERS: L4
#DO NOT CHANGE - OPTIONS NOT CURRENTLY FUNCTIONAL
##################################################################

##################################################################


##################################################################
#PARAMETERS: L5
#DEBUG/DEPRECATED - WILL MOST LIKELY BE REMOVED IN FUTURE
##################################################################

##################################################################
