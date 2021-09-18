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

#If trainingModel, should existing database/cValue in RESULTS be loaded instad of creating a new one
loadTrainValDatasets = False

#If validation dataset should be simulated
validationModel = True

#Is testing of a model to be performed
testingModel = True

#Is this an implementation run
impModel = False

#If an implementation run, what name should be used for data obtained
impSampleName = 'SAMPLE_1'

#If an implementation run, where will the MSI files be located (location will be emptied on run); None equivalent to './RESULTS/IMP/'
impInputDir = None

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
scanMethod = 'pointwise'

#var, max, avg, sum, original (original collapses before difference between recon and ground-truth mz)
RDMethod = 'sum'

#Should a global mz specification be used (True), or should the mz be taken for each sample independently (False)
mzGlobalSpec = True

#==================================================================
#PARAMETERS: L1-0
#POINTWISE OPTIONS
#==================================================================

#What percentage of points should be initially acquired (random) during testing/implementation
initialPercToScan = 1

#Stopping percentage for number of acquired pixels during testing/implementation
stopPerc = 40

#If group-wise, what percentage of points should be acquired; otherwise set to None
percToScan = None

#What percentage of points should be acquired between visualization steps; if all steps should be, then set to None
percToViz = 1

#==================================================================

#==================================================================
#PARAMETERS: L1-1
#LINEWISE OPTIONS
#==================================================================

#What method should be used for linewise point selection: (segLine, partial line segment) (percLine, Top stopPerc% ERD locations) (none, full line)
lineMethod = 'segLine'

#Should lines be allowed to be revisited
lineRevist = False

#Should all lines be scanned at least once
lineVisitAll = False

#==================================================================

#==================================================================
#PARAMETERS: L1-2
#TRAINING DATA GENERATION
#==================================================================

#What percentage of points should be initially acquired (random) during training
initialPercToScanTrain = 1

#Stopping percentage for number of acquired pixels for training
stopPercTrain = 40

#Possible c values for RD approximation
cValues = np.array([1, 2, 4, 8, 16])

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

#How many augmented versions of the training data should be added to the base set
numAugTimes = 0

#What inputs should be used to the network (ReconAndMeasured, ReconValues, MeasuredValues, AverageReconValues, AverageMeasuredValues, AverageReconAndMeasured)
inputMethod = 'ReconAndMeasured'

#Which model should be used for training: cnn, unet, or flatunet
modelDef = 'unet'

#How many filters should be used
numStartFilters = 32

#Which optimizer should be used('Nadam', 'Adam', or 'RMSProp')
optimizer = 'Nadam'

#Which loss function should be used for the optimizer ('MAE', or 'MSE')
lossFunc = 'MAE'

#What should the learning rate of the model's optimizer be
learningRate = 1e-4

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

#Are the filename line numbers going to be labeled sequentially rather than by physical row number during implementation
unorderedNames = False

#Should parallelization calls be used (True); if memory overflow issues develop, set to False
parallelization = True

#Which system GPU(s) are available; ('None', whichever is available; '-1', CPU only)
availableGPUs = '1'

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

#Should mz images be shown with log normalized colorbars 
sysLogNorm = False

#Should only a single mz be used as the network input (allows evaluation over multiple, uses first mz in mz.csv local/global file)
#WARNING: ONLY should enable when using SLADS variants; Purpose is proof that considering multiple mz channels is better than a single
#NOTE: DELETE ASAP!!! Very likely to break expected usage - After results published
mzSingle = False

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

#Should total distortion be used for c value determination and static window in RD generation [15,15] be used (For comparison with originally published SLADS methods)
legacyFlag = False

##################################################################
