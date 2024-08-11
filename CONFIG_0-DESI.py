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

#Is post-processing of existing scanned data to be performed
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
availableThreads = 0

#Which GPU(s) devices should be used (last specified used for training); (default: [-1], any/all available; CPU only: [])
gpus = [-1]

#Should training/validation data be entirely stored on GPU (default: True; improves training/validation efficiency, set to False if OOM occurs)
#DLADS-PY, DLADS, and GLANDS specific
storeOnDevice = True

#RNG seed value to control run-to-run consistency, may slow performance (-1 to disable)
manualSeedValue = 0

#If the FOV should be masked during training, specify a dilation kernel size (odd) for the mask (Disabled mask: None, applied without dilation: 0; DESI default: None; MALDI default: 3)
trainMaskFOVDilation = None

#If the FOV should be masked during non-training, specify a dilation kernel size (odd) for the mask (Disabled mask: None, applied without dilation: 0; DESI default: None; MALDI default: 0)
otherMaskFOVDilation = None

#Should the percentage measured be relative to the FOV mask area (True) or the whole FOV area (False) (default: False)
#Does not apply to training/validation generation; disabled automatically if dilation flags are set to None
#If enabled, will cause complications in evaluation if some samples have masks and others do not
percFOVMask = False

#==================================================================
#PARAMETERS: L1-0
#TASK METHODS
#==================================================================

#Which model should be used for ERD generation (SLADS-LS, SLADS-Net, DLADS-TF, DLADS-PY, DLADS, GLANDS)
#SLADS-LS, SLADS-Net, DLADS-TF, and DLADS-PY are legacy methods only intended for simulated reference/benchmarking
#GLANDS has not yet been included for release
erdModel = 'DLADS'

#Which scanning method shoud be used: pointwise or linewise
scanMethod = 'pointwise'

#If pointwise stopping percentage for number of acquired locations
#If linewise and segLine method, percentage of locations to acquire per line
stopPerc = (1/3)*100

#Should the final reconstructed data of all channels be saved in .imzML format (MSI only; default: False)
imzMLExport = False

#Should an evaluation of reconstructions be performed across all channels (MSI only; default: False)
allChanEval = True

#Should input data be standardized for RDPP/ERD computations (default: True)
#If this and the value of padInputData changes, then training data generation will need to be rerun
standardizeInputData = True

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

#Should output visualizations be generated during acquisition; this is neither recommended (computationally expensive) nor regularly validated (default: False)
liveOutputFlag = False

#==================================================================

#==================================================================
#PARAMETERS: L1-2
#POINTWISE OPTIONS
#==================================================================

#Percentage of points to initially acquire (random) during testing/implementation (default: 1)
initialPercToScan = 1

#Percentage of points (group-based) to acquire each iteration during testing/implementation (default: None will scan only one location per iteration)
#Temporarily sets reconstruction values as having been measured for updating the ERD. 
#If an oracle run, will set the RD values in unmeasured locations, that would be impacted by selected scan positions, to zero
percToScan = None

#Percentage of points to acquire between visualizations; if all steps should be, then set to None (pointwise default: 1; linewise default: None)
percToViz = 1

#==================================================================

#==================================================================
#PARAMETERS: L1-3
#LINEWISE OPTIONS
#==================================================================

#How should points be returned from a chosen line: (segLine; partial line segment) (percLine; top stopPerc% ERD locations) (none, full line)
lineMethod = 'segLine'

#If using a segLine, how should the start and end points be determined (minPerc, segment of stopPerc length) (default: 'otsu')
segLineMethod = 'otsu'

#Should lines be allowed to be revisited (default: False)
#This function only works for fully-acquired simulations and should be disabled for all other operations; will almost certainly corrupt data if used incorrectly
lineRevist = False

#Should all lines be scanned/visited at least once (default: True)
lineVisitAll = True

#Specify what line positions (percent height) should be used for initial acquistion
startLinePositions = [0.25, 0.50, 0.75]

#==================================================================

#==================================================================
#PARAMETERS: L1-4
#TRAINING DATA GENERATION (DLADS/SLADS SPECIFIC)
#==================================================================

#What percentage of points should be initially acquired (random) during training and c value optimization (default: 1)
initialPercToScanTrain = 1

#Stopping percentage for number of acquired pixels for training and c value optimization; always out of total FOV (default: 30)
stopPercTrain = (1/3)*100

#How many masks should be used for each percentage during training (default: 1)
numMasks = 1

#Should visualizations of the training/validation samples be generated during database generation (default: False)
visualizeTrainingData = False

#If using IDW reconstruction, how many neighbors should be considered (default: 10)
numNeighbors = 10

#Possible c values for RD approximation
cValues = [1, 2, 4, 8, 16, 32]

#When optimizing c, percentage of points (group-based) to acquire between steps; set as None to disable (default: 1)
#Temporarily sets the RD values in unmeasured locations, that would be impacted by selected scan positions, to zero
percToScanC = 1

#When optimizing c, percentage of points to acquire between recorded/tracked metrics (default: None)
#Set as None to consider all measurement steps; if percToScanC = None, then to avoid OOM, recommend setting to 1
percToVizC = None

#==================================================================

##################################################################


##################################################################
#PARAMETERS: L2
#MODEL TRAINING PARAMETERS (DLADS/GLANDS SPECIFIC)
##################################################################

#==================================================================
#PARAMETERS: L2-1
#TRAINING DATA HANDLING
#==================================================================

#Specify what input data constitutes a model (DLADS or GLANDS) input (Options: 'opticalData', 'mask', 'reconData', 'measureData', 'combinedData')
#DLADS default: ['mask',  'combinedData']
#DLADS(-TF,-PY) default: ['mask', 'reconData', 'measureData']
#GLANDS default: ['mask', 'measureData']
inputChannels = ['mask', 'combinedData']

#What percentage of the training data should be used for training (default: 0.8)
#1.0 or using only one input sample will use training loss for early stopping criteria
trainingSplit = 0.8

#Should the training data be augmented at the end of each epoch (default: True)
augTrainData = True

#Should visualizations of the training progression be generated (default: True)
trainingProgressionVisuals = True

#How often (epochs) should visualizations of the training progression be generated (default: 10)
trainingVizSteps = 10

#GLANDS ONLY: Training data batch size (default: 1)
batchsize_TRN = 1

#GLANDS ONLY: Validation data batch size; (default: -1, sets as total length of validation set, set manually if OOM occurs)
batchsize_VAL = -1

#==================================================================

#==================================================================
#PARAMETERS: L2-2
#ARCHITECTURE PARAMETERS
#==================================================================

#Should input data be padded for even dimensions throughout up and down sampling (default: True)
#If this and the value of standardizeInputData changes, then training data generation will need to be rerun
padInputData = True

#Reference number of how many convolutional filters to use in building the model
numStartFilters = 64

#What initialization should be used: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal' (default: 'xavier_uniform')
initialization='xavier_uniform'

#Activations for network sections (input, down, embedding down, upsampling, embedding up, final): 'leaky_relu', 'relu', 'prelu', 'linear'
inAct, dnAct, upAct, fnAct = 'leaky_relu', 'leaky_relu', 'leaky_relu', 'relu'

#Negative slope for any 'leaky_relu' or 'prelu' activations (default: 0.2)
leakySlope = 0.2

#Should bias be enabled in convolutional layers (default: True)
useBias = True

#Binomial blur padding model mode: 'reflection', 'zero', 'partialConvolution'
blurPaddingMode = 'reflection'

#Sigma for binomial filter applied during upsampling; 0 to disable (default: 3)
sigmaUp = 3

#Sigma for binomial filter applied during downsampling; 0 to disable (default: 1)
sigmaDn = 1

#Should instance normalization be used throughout the network (default: False)
dataNormalize = False

#==================================================================

#==================================================================
#PARAMETERS: L2-2
#OPTIMIZATION PARAMETERS
#==================================================================

#==================================================================

#Which optimizer should be used: 'AdamW', 'Adam', 'NAdam', 'RMSprop', or 'SGD' (default: 'AdamW')
optimizer = 'AdamW'

#What should the initial learning rate for the model optimizer(s) be (default: 1e-4)
learningRate = 1e-4

#How many epochs should the model training wait to see an improvement using the early stopping criteria (default: 100)
maxPatience = 100

#How many epochs should a model be allowed to train for (default: 1000)
maxEpochs = 1000

#If cosine annealing with warm restarts should be used (default: True)
scheduler_CAWR = True

#If using cosine annealing, what should the period be between resets (default: 1)
schedPeriod = 1

#If using cosine annealing, how should the period length be multiplied at each reset (default: 1)
schedMult = 1

#Across how many epochs across should two moving averages be considered for early stopping criteria (default: 0)
#Historical loss = mean(losses[-sepEpochs*2:-sepEpochs]; Current loss = mean(losses[-sepEpochs:])
#0 will trigger early stopping criteria relative to occurance of best loss
sepEpochs = 0

#How many epochs should a model be allowed to remain wholly stagnant for before training termination (default: 10)
maxStagnation = 10

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

#Should static window be used in RD generation (default: False)
staticWindow = False

#If a static window is to be used, what size (symmetric) should it be (default: 15)
staticWindowSize = 15

#If a dynamic window is to be used, what multiple of the sigma value should be used (default: 3)
dynWindowSigMult = 3

#Should only a single channel be used as the network input (default: False)
#MSI uses first channel in channels.csv local/global file, IMAGE uses first channel read
chanSingle = False

#Distance for SLADS to no longer consider surrounding features (default: 0.25)
featDistCutoff = 0.25

#Should existing results folders not be allowed to be overwritten? (default: False)
preventResultsOverwrite = False

#Define precision of the percentage averaging (as percentage is inconsistent between acquistion steps) (default: 0.001)
precision = 0.001

#Method overwrite file; will execute specified file to overwrite otherwise defined methods/parameters (default: None)
overWriteFile = None

#Model overwrite file; will execute specified file to overwrite otherwise defined methods/parameters at model definition time (default: None)
overWriteModelFile = None

#Should testing/simulation sample data object be saved; sometimes needed for post-processing and results writeup (default: False)
storeTestingSampleData = False

#Should progress bars use ascii formatting (default: False)
asciiFlag = False

#If performing a benchmark, should processing be skipped (default: False)
benchmarkNoProcessing = False

#Should simulated sampling in testing be bypassed to work on OOM errors; relevant sections must have already been run successfully (default: False)
bypassSampling = False

#Should result objects generated in testing be kept (only needed if bypassSampling is to be used later) (default: False)
keepResultData = False

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

##################################################################