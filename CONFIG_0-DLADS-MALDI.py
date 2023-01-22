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
#TASK METHODS & HARDWARE
##################################################################

#Which scanning method shoud be used: pointwise or linewise
scanMethod = 'pointwise'

#If the FOV should be masked during training, specify a dilation kernel size (odd) for the mask (Disabled mask: None, applied without dilation: 0; DESI default: None; MALDI default: 3)
trainMaskFOVDilation = 3

#If the FOV should be masked during non-training, specify a dilation kernel size (odd) for the mask (Disabled mask: None, applied without dilation: 0; DESI default: None; MALDI default: 0)
otherMaskFOVDilation = 0

#Should the percentage measured be relative to the FOV mask area (True) or the whole FOV area (False)
#Does not apply to training/validation generation
#If enabled, will cause complications in evaluation if some samples have masks and others do not
percFOVMask = True

#Should inputs to model and RDPP calculations be adjusted ('rescale', 'standardize', or default: None)
#Only affects DLADS and GLANDS pipelines
dataAdjust = None

#Should the final reconstructed data be saved in .imzML format
imzMLExport = True

#If using IDW reconstruction, how many neighbors should be considered
numNeighbors = 10

#Should parallelization calls be used; if memory overflow issues develop, try setting to False (massively increases computation time)
parallelization = True

#Which GPU(s) should be used for training; ('None', any/all available; '-1', CPU only)
availableGPUs = 'None'

#If parallelization is enabled how many CPU threads should be reserved for other computer functions
#Recommend starting at half of the available threads; decrease just enough that CPU is not pinned at 100% in parallel operations
#May increase up to the total number of system CPU threads to decrease relative memory pressure
reserveThreadCount = 16

#==================================================================
#PARAMETERS: L1-0
#LEGACY TASK METHODS
#==================================================================

#Which model should be used for ERD generation (SLADS-LS, SLADS-Net, DLADS)
erdModel = 'DLADS'

#Should only a single channel be used as the network input (allows evaluation over multiple, MSI uses first channel in channels.csv local/global file, IMAGE uses first channel read)
chanSingle = False

#Should static window be used in RD generation
staticWindow = False

#If a static window is to be used, what size (symmetric) should it be ([15,15] for SLADS(-Net))
staticWindowSize = 15

#If a dynamic window is to be used, what multiple of the sigma value should be used
dynWindowSigMult = 3

#Distance to no longer consider surrounding features (disregard if using DLADS, 0.25 for SLADS(-Net))
featDistCutoff = 0.25

#==================================================================
#PARAMETERS: L1-1
#IMPLEMENTATION OPTIONS
#==================================================================

#If an implementation run, what name should be used for data obtained
impSampleName = 'SAMPLE'

#If an implementation run, where will the MSI files be located (location will be emptied on run); None equivalent to './RESULTS/IMP/'
impInputDir = None

#If the measurement times listed in the acquried MSI files do not start with 0 being at the left-side of the FOV
impOffset = True

#Should output visualizations be generated during acquisition? (Not recommended; substantially reduces performance)
liveOutputFlag = False

#==================================================================

#==================================================================
#PARAMETERS: L1-2
#POINTWISE OPTIONS
#==================================================================

#What percentage of points should be initially acquired (random) during testing/implementation
initialPercToScan = 1

#Stopping percentage for number of acquired pixels during testing/implementation
stopPerc = (1/3)*100

#If group-wise, what percentage of points should be acquired; otherwise set to None
percToScan = None

#What percentage of points should be acquired between visualization steps; if all steps should be, then set to None
percToViz = 1

#==================================================================

#==================================================================
#PARAMETERS: L1-3
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
#PARAMETERS: L1-4
#TRAINING DATA GENERATION
#==================================================================

#What percentage of points should be initially acquired (random) during training
initialPercToScanTrain = 1

#Stopping percentage for number of acquired pixels for training; always out of total FOV
stopPercTrain = 30

#Possible c values for RD approximation
cValues = np.array([1, 2, 4, 8, 16, 32, 64])

#Should the PSNR of all channels be used for c value optimization (True, computationally expensive), or just targeted channels (default: False)
cAllChanOpt = False

#How many masks should be used for each percentage during training
numMasks = 1

#==================================================================

##################################################################


##################################################################
#PARAMETERS: L2
#NEURAL NETWORK MODEL PARAMETERS
##################################################################

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

#Should the training data be augmented at the end of each epoch
augTrainData = True

#How many epochs should a model train for at maximum
numEpochs = 1000

#Should the model training be cutoff early if no improvement is seen, using patience criteria
earlyCutoff = True

#How many epochs should the model training wait to see an improvement before terminating
maxPatience = 25

#How many epochs at minimum should be performed before starting to save the current best model and consider termination
minimumEpochs = 10

#What percentage of the training data should be used for training (setting as 1.0 or using one input sample will use training loss for early stopping criteria)
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

#Should existing results folders not be allowed to be overwritten?
preventResultsOverwrite = False

#If consistency in the random generator is desired for inter-code comparisons (does not affect DLADS training)
consistentSeed = True

#Define precision of the percentage averaging (as percentage is inconsistent between acquistion steps)
precision = 0.001

#Method overwrite file; will execute specified file to overwrite otherwise defined methods
overWriteFile = None

#Should testing/simulation sample data object be saved; sometimes needed for post-processing and results writeup
storeTestingSampleData = False

#Shoud progress bars use ascii formatting
asciiFlag = False

##################################################################


##################################################################
#PARAMETERS: L4
#DO NOT CHANGE - ALTERNATIVE OPTIONS NOT CURRENTLY FUNCTIONAL
##################################################################

##################################################################


##################################################################
#PARAMETERS: L5
#DEBUG/DEPRECATED - OPTIONS LIKELY TO BE REMOVED IN FUTURE
##################################################################

##################################################################
