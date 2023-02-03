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

#If parallelization is enabled how many CPU threads should be reserved for other computer functions
#Recommend starting at half of the available threads if using hyperthreading; decrease just enough that CPU is not pinned at 100% in parallel training operations
#May increase up to the total number of system CPU threads to decrease relative memory pressure
reserveThreadCount = 0

#Which GPU(s) should be used for training; ('None', any/all available; '-1', CPU only)
availableGPUs = 'None'

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

#Which model should be used for ERD generation (SLADS-LS, SLADS-Net, DLADS)
erdModel = 'DLADS'

#Which scanning method shoud be used: pointwise or linewise
scanMethod = 'linewise'

#If pointwise stopping percentage for number of acquired locations
#If linewise and percLine method, percentage of locations to acquire per line
stopPerc = (1/3)*100

#Should the final reconstructed data be saved in .imzML format
imzMLExport = True

#Should inputs to model and RDPP calculations be adjusted: 'rescale', 'standardize', or None (default: None)
#Only affects DLADS and GLANDS pipelines; setting must match between model training and use
dataAdjust = None

#If using IDW reconstruction, how many neighbors should be considered
numNeighbors = 10

#Should static window be used in RD generation
staticWindow = False

#If a static window is to be used, what size (symmetric) should it be ([15,15] for SLADS(-Net))
staticWindowSize = 15

#If a dynamic window is to be used, what multiple of the sigma value should be used
dynWindowSigMult = 3

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

#Percentage of points to initially acquire (random) during testing/implementation
initialPercToScan = 1

#Percentage of points (group-based) to acquire during testing/implementation; otherwise set to None (default: None)
#Temporarily sets reconstruction values as having been measured for updating the ERD. 
#If an oracle run, will set the RD values in unmeasured locations, that would be impacted by selected scan positions, to zero
percToScan = None

#Percentage of points to acquire between visualizations; if all steps should be, then set to None (default: 1)
percToViz = None

#==================================================================

#==================================================================
#PARAMETERS: L1-3
#LINEWISE OPTIONS
#==================================================================

#How should points be returned from a chosen line: (segLine; partial line segment) (percLine; top stopPerc% ERD locations) (none, full line)
lineMethod = 'segLine'

#How should individual points on a chosen line be selected: 'single' or 'group' (default: 'group')
linePointSelection = 'group'

#If using a segLine, how should the start and end points be determined (minPerc, left/right most of the top stopPerc ERD values) (otsu, left/right most of the foreground ERD found with Otsu)
segLineMethod = 'otsu'

#Should all lines be scanned at least once
lineVisitAll = True

#Specify what line positions (percent height) should be used for initial acquistion
startLinePositions = [0.25, 0.50, 0.75]

#==================================================================

#==================================================================
#PARAMETERS: L1-4
#TRAINING DATA GENERATION
#==================================================================

#What percentage of points should be initially acquired (random) during training and c value optimization
initialPercToScanTrain = 1

#Stopping percentage for number of acquired pixels for training and c value optimization; always out of total FOV
stopPercTrain = 30

#How many masks should be used for each percentage during training
numMasks = 1

#==================================================================

#==================================================================
#PARAMETERS: L1-5
#c VALUE OPTIMIZATION
#==================================================================

#Possible c values for RD approximation
cValues = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])

#When optimizing c, percentage of points (group-based) to acquire; otherwise set to None (default: 1)
#Temporarily sets the RD values in unmeasured locations, that would be impacted by selected scan positions, to zero
#Note that using percToVizC is a more accurate, but expensive method
percToScanC = 1

#When optimizing c, percentage of points to acquire between visualizations; if all steps should be, then set to None (default: None)
percToVizC = None

#Should the PSNR of all channels be used for c value optimization (True, computationally expensive), or just targeted channels (default: False)
cAllChanOpt = False

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

#Which loss function should be used for the optimizer ('MAE' (MSI default) or 'MSE' (SEM image default))
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
maxPatience = 50

#How many epochs at minimum should be performed before starting to save the current best model and consider termination
minimumEpochs = 10

#What percentage of the training data should be used for training (setting as 1.0 or using one input sample will use training loss for early stopping criteria)
trainingSplit = 0.8

#Should visualizations of the training/validation samples be generated during database generation
visualizeTrainingData = True

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

#Should only a single channel be used as the network input (MSI uses first channel in channels.csv local/global file, IMAGE uses first channel read)
chanSingle = False

#Distance to no longer consider surrounding features (disregard if using DLADS, 0.25 for SLADS(-Net))
featDistCutoff = 0.25

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

#Should lines be allowed to be revisited
lineRevist = False

##################################################################

