#====================================================================
#CONFIGURATION
#====================================================================

##################################################################
#PARAMETERS: L0
#TASKS TO BE PERFORMED
##################################################################

#Is training data generation to be performed
trainingDataGen = True

#Is training of a model to be performed
trainingModel = True

#If trainingModel, should existing database/cValue in RESULTS be loaded instad of creating a new one
loadTrainValDatasets = False

#If validation dataset should be simulated
validationModel = False

#Is testing of a model to be performed
testingModel = True

#Is this an implementation run
impModel = False

#Is post-processing to be performed
postModel = False

##################################################################


##################################################################
#PARAMETERS: L1
#COMPUTE HARDWARE & TASK METHODS
##################################################################

#Should parallelization calls be used; if memory overflow issues develop, try setting to False (massively increases computation time)
parallelization = True

#If parallelization is enabled, how many CPU threads should be used? (0 will use any/all available)
#Recommend starting at half of the available system threads if using hyperthreading,
#or 1-2 less than the number of system CPU cores if not using hyperthreading.
#Adjust to where the CPU just below 100% usage during parallel operations 
#May decrease value to help reduce RAM pressure. 
availableThreads = 0

#Which GPU(s) devices should be used for training; (Default: [-1], any/all available; CPU only: [])
gpus = [-1]

#Should training/validation data be entirely stored on GPU (default: True; improves training/validation performance, set to False if OOM occurs)
#DLADS/GLANDS specific
storeOnDevice = True

#RNG seed value to control run-to-run consistency, may slow performance (-1 to disable)
manualSeedValue = 0

#If the FOV should be masked during training, specify a dilation kernel size (odd) for the mask (Disabled mask: None, applied without dilation: 0; DESI default: None; MALDI default: 3)
trainMaskFOVDilation = None

#If the FOV should be masked during non-training, specify a dilation kernel size (odd) for the mask (Disabled mask: None, applied without dilation: 0; DESI default: None; MALDI default: 0)
otherMaskFOVDilation = None

#Should the percentage measured be relative to the FOV mask area (True) or the whole FOV area (False)
#Does not apply to training/validation generation; disabled automatically if dilation flags are set to None
#If enabled, will cause complications in evaluation if some samples have masks and others do not
percFOVMask = False

#==================================================================
#PARAMETERS: L1-0
#TASK METHODS
#==================================================================

#Which model should be used for ERD generation (SLADS-LS, SLADS-Net, DLADS, GLANDS)
erdModel = 'DLADS'

#Which scanning method shoud be used: pointwise or linewise
scanMethod = 'linewise'

#If pointwise stopping percentage for number of acquired locations
#If linewise and segLine method, percentage of locations to acquire per line
stopPerc = (1/3)*100

#Should the final reconstructed data of all channels be saved in .imzML format (MSI only; default: False)
imzMLExport = False

#Should an evaluation of reconstructions be performed across all channels (MSI only; default: False)
allChanEval = False

#==================================================================

#==================================================================
#PARAMETERS: L1-1
#IMPLEMENTATION/POST OPTIONS
#==================================================================

#If an implementation run, what name should be used for data obtained
impSampleName = 'SAMPLE'

#If an implementation run, where will the MSI files be located (location will be emptied on run); None equivalent to './RESULTS/IMP/'
impInputDir = None

#If the measurement times listed in the acquired MSI files do not start with 0 being at the left-side of the FOV
impOffset = True

#Should output visualizations be generated during acquisition? Highly not recommended (expensive) nor regularly validated (default: False)
liveOutputFlag = False

#==================================================================

#==================================================================
#PARAMETERS: L1-2
#POINTWISE OPTIONS
#==================================================================

#Percentage of points to initially acquire (random) during testing/implementation
initialPercToScan = 1

#Percentage of points (group-based) to acquire each iteration during testing/implementation (default: None will scan only one location per iteration)
#Temporarily sets reconstruction values as having been measured for updating the ERD. 
#If an oracle run, will set the RD values in unmeasured locations, that would be impacted by selected scan positions, to zero
percToScan = None

#Percentage of points to acquire between visualizations; if all steps should be, then set to None (pointwise default: 1; linewise default: None)
percToViz = None

#==================================================================

#==================================================================
#PARAMETERS: L1-3
#LINEWISE OPTIONS
#==================================================================

#How should points be returned from a chosen line: (segLine; partial line segment) (percLine; top stopPerc% ERD locations) (none, full line)
lineMethod = 'segLine'

#If using a segLine, how should the start and end points be determined (minPerc, segment of stopPerc length) (default: 'otsu', foreground ERD)
segLineMethod = 'otsu'

#Should lines be allowed to be revisited (default: False)
#Note: This function only works for fully-acquired simulations and should be disabled for all other operations; will almost certainly corrupt data if used incorrectly
lineRevist = False

#Should all lines be scanned/visited at least once
lineVisitAll = True

#Specify what line positions (percent height) should be used for initial acquistion
startLinePositions = [0.25, 0.50, 0.75]

#==================================================================

#==================================================================
#PARAMETERS: L1-4
#TRAINING DATA GENERATION (DLADS/SLADS SPECIFIC)
#==================================================================

#What percentage of points should be initially acquired (random) during training and c value optimization
initialPercToScanTrain = 1

#Stopping percentage for number of acquired pixels for training and c value optimization; always out of total FOV
stopPercTrain = 30

#How many masks should be used for each percentage during training
numMasks = 1

#Should visualizations of the training/validation samples be generated during database generation (default: False)
visualizeTrainingData = False

#If using IDW reconstruction, how many neighbors should be considered
numNeighbors = 10

#Possible c values for RD approximation
cValues = [1, 2, 4, 8, 16, 32, 64, 128, 256]

#When optimizing c, percentage of points (group-based) to acquire; otherwise set to None (default: 1)
#Temporarily sets the RD values in unmeasured locations, that would be impacted by selected scan positions, to zero
#Note that using percToVizC is a more accurate, but expensive method
percToScanC = 1

#When optimizing c, percentage of points to acquire between visualizations; if all steps should be, then set to None (default: None)
percToVizC = None

#==================================================================

##################################################################


##################################################################
#PARAMETERS: L2
#NEURAL NETWORK MODEL PARAMETERS (DLADS/GLANDS SPECIFIC)
##################################################################

#Specify what input data constitutes a model (DLADS or GLANDS) input (Options: 'opticalData', 'mask', 'reconData', 'measureData')
#reconData is only available for DLADS
#DLADS default: ['mask', 'reconData', 'measureData']
#GLANDS default: ['mask', 'measureData']
inputChannels = ['mask', 'reconData', 'measureData']

#How many filters should be used at the top of the network
numStartFilters = 64

#Which optimizer should be used ('AdamW', 'Adam', 'Nadam', 'SGD' or 'RMSProp')
optimizer = 'Nadam'

#What should the learning rate of the model optimizer(s) be
learningRate = 1e-5

#Beta 1  parameter if applicable to the specified optimizer (default: 0.5)
beta1 = 0.5

#Beta 2  parameter if applicable to the specified optimizer (default: 0.5)
beta2 = 0.999

#What percentage of the training data should be used for training (default: 0.8)
#1.0 or using only one input sample will use training loss for early stopping criteria
trainingSplit = 0.8

#How many epochs should a model train for at maximum (default: 10000)
numEpochs = 10000

#How many epochs should the model training wait to see an improvement before terminating (default: 100)
maxPatience = 100

#How many epochs at minimum should be performed before starting to save the current best model and consider termination (default: 10)
minimumEpochs = 10

#Should the training data be augmented at the end of each epoch (default: True)
augTrainData = True

#Should visualizations of the training progression be generated (default: True)
trainingProgressionVisuals = True

#If visualizations of the training progression are to be generated, how often (epochs) should this occur (default: 10)
trainingVizSteps = 10

#GLANDS ONLY: Training data batch size (default: 1)
batchsize_TRN = 1

#GLANDS ONLY: Validation data batch size; (default: -1, sets as total length of validation set, set manually if OOM occurs)
batchsize_VAL = -1

#==================================================================

##################################################################


##################################################################
#PARAMETERS: L3
#GENERALLY NOT CHANGED
##################################################################

#If an alternate .csv file should be used instead of './channels.csv', in the event that a sample does not include its own, specify it here (default: None)
overrideChannelsFile = None

#If a folder other than the default './INPUT/' should be used, specify it here (default: None)
overrideInputsFolder = None

#If a folder other than the default './RESULTS/' should be used, specify it here (default: None)
overrideResultsFolder = None

#If existing allChanImages.hdf5 and squareAllImages.hdf5 files should be overwritten or should attempt to be loaded (default: False)
overwriteAllChanFiles = False

#Should static window be used in RD generation
staticWindow = False

#If a static window is to be used, what size (symmetric) should it be (default: 15 for SLADS)
staticWindowSize = 15

#If a dynamic window is to be used, what multiple of the sigma value should be used
dynWindowSigMult = 3

#Should only a single channel be used as the network input (MSI uses first channel in channels.csv local/global file, IMAGE uses first channel read)
#SLADS/DLADS compatible only
chanSingle = False

#Distance to no longer consider surrounding features (disregard if using DLADS, 0.25 for SLADS(-Net))
featDistCutoff = 0.25

#Should existing results folders not be allowed to be overwritten?
preventResultsOverwrite = False

#Define precision of the percentage averaging (as percentage is inconsistent between acquistion steps)
precision = 0.001

#Method overwrite file; will execute specified file to overwrite otherwise defined methods/parameters
overWriteFile = None

#Should testing/simulation sample data object be saved; sometimes needed for post-processing and results writeup
storeTestingSampleData = False

#Should progress bars use ascii formatting
asciiFlag = False

#If performing a benchmark, should processing be skipped (default: False)
benchmarkNoProcessing = False

#Should the program save measured location information to disk after every iteration; intended for debugging (default: False)
saveIterationFlag = False

#Should otherwise hidden warnings and error logging be generated (default: False)
debugMode = False

##################################################################


##################################################################
#PARAMETERS: L4
#DO NOT CHANGE - ALTERNATIVE OPTIONS NOT CURRENTLY FUNCTIONAL
##################################################################

#Should m/z channels be automatically selected based on the training data (MSI Specific)
#Not currently implemented; initial import would readScanData with no channels, then determine m/z channels, then readScanData with channels...
mzAutoSelection = False

##################################################################


##################################################################
#PARAMETERS: L5
#DEBUG/DEPRECATED - OPTIONS LIKELY TO BE REMOVED IN FUTURE
##################################################################

#If all samples have FOV aligned optical image files, how should they be applied to the E/RD: 'directBias', 'secDerivBias' or None (default: None)
#This parameter must be consistently applied for optimizing c value, training a model, and eventual testing/implementation
applyOptical = None

##################################################################

