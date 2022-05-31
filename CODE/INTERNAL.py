#==================================================================
#INTERNAL SETUP
#==================================================================

#AESTHETIC SETUP
#==================================================================
#Determine console size if applicable
if systemOS != 'Windows':
    consoleRows, consoleColumns = os.popen('stty size', 'r').read().split()
elif systemOS == 'Windows':
    h = windll.kernel32.GetStdHandle(-12)
    csbi = create_string_buffer(22)
    res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
    (bufx, bufy, curx, cury, wattr, left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
    consoleRows = bottom-top
    consoleColumns = right-left
    
#INTERNAL OBJECT SETUP
#==================================================================

#Limit GPU(s) if indicated
if availableGPUs != 'None': os.environ["CUDA_VISIBLE_DEVICES"] = availableGPUs
numGPUs = len(tf.config.experimental.list_physical_devices('GPU'))

#Initialize ray instance; leave 1 processor thread free if possible; make sure a ray instance isn't already running
ray.shutdown()
if parallelization: 
    numberCPUS = multiprocessing.cpu_count()-1
    if numberCPUS == 1: parallelization = False
else: numberCPUS = 1
ray.init(num_cpus=numberCPUS, configure_logging=True, logging_level=logging.ERROR, include_dashboard=False)

#Allow partial GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

#If the number of gpus to be used is greater thconfigure_logging an 1, then increase the batch size accordingly for distribution
if len(gpus)>1: batchSize*=len(gpus)

#RAY REMOTE METHOD DEFINITIONS
#==================================================================

#Define deployment for trained models
@serve.deployment(route_prefix="/ModelServer", ray_actor_options={"num_gpus": numGPUs})
class ModelServer:

    def __init__(self, erdModel, modelPath):
        self.erdModel = erdModel
        if self.erdModel == 'SLADS-LS' or self.erdModel == 'SLADS-Net': self.model = np.load(modelPath+'.npy', allow_pickle=True).item()
        elif self.erdModel == 'DLADS' or self.erdModel == 'GLANDS': self.model = tf.function(tf.keras.models.load_model(modelPath, compile=False), experimental_relax_shapes=True)

    def __call__(self, data):
        if self.erdModel == 'SLADS-LS' or self.erdModel == 'SLADS-Net': return self.model.predict(data)
        elif self.erdModel == 'DLADS' or self.erdModel == 'GLANDS': return self.model(data, training=False)[:,:,:,0].numpy()

#Load m/z and TIC data from a specified MSI file
@ray.remote
def scanData_parhelper(sampleData, scanFileName):

    #Establish file pointer and line number (1 indexed) for the specific scan; flag indicates 'good'/'bad' data file (primarily checking for files without data)
    readErrorFlag = False
    try: data = mzFile(scanFileName)
    except: readErrorFlag = True
    
    #If the data file is 'good' then continue processing
    if not readErrorFlag:
        
        #Extract line number from the filename, removing leading zeros, subtract 1 for zero indexing
        fileNum = int(scanFileName.split('line-')[1].split('.')[0].lstrip('0'))-1
        
        #If the file numbers are not the physical row numbers, then obtain correct number from stored LUT
        if sampleData.unorderedNames: 
            try: lineNum = sampleData.physicalLineNums[fileNum+1]
            except: 
                print('Warning - Attempt to find the physical line number for the file: ' + scanFileName + ' has failed; the file will therefore be ignored this iteration.')
                readErrorFlag = True
        else: lineNum = fileNum
        
    #If the data file is still 'good' then continue processing
    if not readErrorFlag:
        
        #If ignoring missing lines, then determine the offset for correct indexing
        if sampleData.ignoreMissingLines and len(sampleData.missingLines) > 0: lineNum -= int(np.sum(lineNum > sampleData.missingLines))

        #Obtain the total ion chromatogram and extract original times
        ticData = np.asarray(data.xic(data.time_range()[0], data.time_range()[1]))
        origTimes, TICData = ticData[:,0], ticData[:,1]
        
        #If the data is being sparesly acquired, then the listed times in the file need to be shifted; convert np.float to float for method compatability
        if (impModel or postModel) and impOffset and scanMethod == 'linewise' and (lineMethod == 'segLine' or lineMethod == 'fullLine'): origTimes += (np.argwhere(sampleData.mask[lineNum]==1).min()/sampleData.finalDim[1])*(((sampleData.sampleWidth*1e3)/sampleData.scanRate)/60)
        elif (impModel or postModel) and impOffset: sys.exit('Error - Using implementation mode with an offset but not segmented-linewise operation is not a supported configuration.')
        mzData = [np.interp(sampleData.newTimes, origTimes, np.nan_to_num(np.asarray(data.xic(data.time_range()[0], data.time_range()[1], float(sampleData.mzRanges[mzRangeNum][0]), float(sampleData.mzRanges[mzRangeNum][1])))[:,1], nan=0, posinf=0, neginf=0), left=0, right=0) for mzRangeNum in range(0, len(sampleData.mzRanges))]
        
        #Interpolate TIC to final new times
        TICData = np.interp(sampleData.newTimes, origTimes, np.nan_to_num(TICData, nan=0, posinf=0, neginf=0), left=0, right=0)
        
        return lineNum, mzData, TICData, scanFileName

#Visualize multiple sample progression steps at once; reimport matplotlib to set backend for non-interactive visualization
@ray.remote
def visualize_parhelper(samples, sampleData, dir_progression, dir_mzProgressions, indexes):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    for index in indexes: visualize_serial(samples[index], sampleData, dir_progression, dir_mzProgressions)

#Perform gaussianGenerator for a set of sigma values
@ray.remote
def gaussian_parhelper(RDPP, idxs, sigmaValues, indexes):
    return [computeRDValue(RDPP, idxs[index], sigmaValues[index]) for index in indexes]

#Run multiple sampling instances in parallel
@ray.remote
def runSampling_parhelper(sampleData, cValue, model, percToScan, percToViz, bestCFlag, oracleFlag, lineVisitAll, liveOutputFlag, dir_Results, datagenFlag, impModel, tqdmHide):
    return runSampling(sampleData, cValue, model, percToScan, percToViz, bestCFlag, oracleFlag, lineVisitAll, liveOutputFlag, dir_Results, datagenFlag, impModel, tqdmHide)

#Visualize multiple sample progression steps at once; reimport matplotlib to set backend for non-interactive visualization
@ray.remote
def visualizeTraining_parhelper(result, maskNum, trainDataFlag, valDataFlag, indexes):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    for index in indexes: visualizeTraining_serial(result.samples[index], result, maskNum, trainDataFlag, valDataFlag)

#NUMBA SETUP - METHOD PRE-COMPILATION
#==================================================================
_ = secondComputeRDValue(np.empty((0,0)), [0,0], 0, 0, [0])

#PATH/DIRECTORY SETUP
#==================================================================

#Set a base model name for the specified configuration; must specify/append c value during run
modelName = 'model_'
if erdModel == 'SLADS-LS': modelName += 'SLADS-LS_'
elif erdModel == 'SLADS-Net': modelName += 'SLADS-Net_'
elif erdModel == 'DLADS': modelName += 'DLADS_'
if mzSingle: modelName += 'mzSingle_'
else: modelName += 'mzMultiple_'
if staticWindow: modelName += 'statWin_' + str(staticWindowSize) + '_'
if not staticWindow: modelName += 'dynWin_' + str(dynWindowSigMult) + '_'

#Data input directories
dir_InputData = '.' + os.path.sep + 'INPUT' + os.path.sep
dir_TrainingData = dir_InputData + 'TRAIN' + os.path.sep
dir_TestingData = dir_InputData + 'TEST' + os.path.sep
dir_PostData = dir_InputData + 'POST' + os.path.sep
if impInputDir == None:  dir_ImpData = dir_InputData + 'IMP' + os.path.sep
else: dir_ImpData = impInputDir

#Results directories
dir_Results = '.' + os.path.sep + 'RESULTS' + os.path.sep
dir_TrainingResults = dir_Results + 'TRAIN' + os.path.sep
dir_TrainingModelResults = dir_TrainingResults + 'Model Training Images' + os.path.sep
dir_TrainingResultsImages = dir_TrainingResults + 'Training Data Images' + os.path.sep
dir_ValidationTrainingResultsImages = dir_TrainingResults + 'Validation Data Images' + os.path.sep
dir_ValidationResults = dir_Results + 'VALIDATION' + os.path.sep
dir_TestingResults = dir_Results + 'TEST' + os.path.sep
dir_ImpResults = dir_Results + 'IMP'+ os.path.sep
dir_PostResults = dir_Results + 'POST'+ os.path.sep

#Check that the result directory exists for cases where existing training data/model are to be used
if (not os.path.exists(dir_Results)) and (not trainingModel): 
    sys.exit('Error - dir_Results: ./RESULTS/ does not exist')
elif not os.path.exists(dir_Results):
    os.makedirs(dir_Results)

#Input data directories
if not os.path.exists(dir_InputData): sys.exit('Error - dir_InputData: ./INPUT/ does not exist')
if not os.path.exists(dir_TrainingData) and trainingModel: sys.exit('Error - dir_TrainingData: ./INPUT/TRAIN/ does not exist')
if not os.path.exists(dir_TestingData) and testingModel: sys.exit('Error - dir_InputData: ./INPUT/TEST/ does not exist')
if not os.path.exists(dir_ImpData) and impModel: sys.exit('Error - dir_ImpData: ./INPUT/IMP/ does not exist')
if not os.path.exists(dir_PostData) and postModel: sys.exit('Error - dir_PostData: ./INPUT/POST/ does not exist')

#As needed, reset the training directories
if trainingModel and not loadTrainValDatasets:
    if os.path.exists(dir_TrainingResults): shutil.rmtree(dir_TrainingResults)
    os.makedirs(dir_TrainingResults)
    os.makedirs(dir_TrainingModelResults)
    os.makedirs(dir_TrainingResultsImages)
    os.makedirs(dir_ValidationTrainingResultsImages)
if trainingModel and loadTrainValDatasets:
    if os.path.exists(dir_TrainingModelResults): shutil.rmtree(dir_TrainingModelResults)
    os.makedirs(dir_TrainingModelResults)
    
#Clear validation, testing, and implementation directories 
if os.path.exists(dir_ValidationResults): shutil.rmtree(dir_ValidationResults)
os.makedirs(dir_ValidationResults)
if os.path.exists(dir_TestingResults): shutil.rmtree(dir_TestingResults)
os.makedirs(dir_TestingResults)
dir_ImpDataFinal = dir_ImpData + impSampleName + os.path.sep
if os.path.exists(dir_ImpDataFinal): shutil.rmtree(dir_ImpDataFinal)
os.makedirs(dir_ImpDataFinal)
if os.path.exists(dir_PostResults): shutil.rmtree(dir_PostResults)
os.makedirs(dir_PostResults)

#Clear the screen
os.system('cls' if os.name=='nt' else 'clear')
