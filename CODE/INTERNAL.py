#==================================================================
#INTERNAL SLADS SETUP
#==================================================================

#AESTHETIC SETUP
#==================================================================
#Determine console size if applicable
if consoleRunning and systemOS != 'Windows':
    consoleRows, consoleColumns = os.popen('stty size', 'r').read().split()
elif consoleRunning and systemOS == 'Windows':
    h = windll.kernel32.GetStdHandle(-12)
    csbi = create_string_buffer(22)
    res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
    (bufx, bufy, curx, cury, wattr, left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
    consoleRows = bottom-top
    consoleColumns = right-left
else:
    consoleRows, consoleColumns = 40, 40

#INTERNAL OBJECT SETUP
#==================================================================

#Limit GPU(s) if indicated
if availableGPUs != 'None': os.environ["CUDA_VISIBLE_DEVICES"] = availableGPUs
numGPUs = len(tf.config.experimental.list_physical_devices('GPU'))

#Initialize multiprocessing pool; make sure a pool isn't still running from a ctl+c exit
if parallelization: 
    ray.shutdown()
    numberCPUS = multiprocessing.cpu_count()
    if numberCPUS>2: numberCPUS = numberCPUS-2
    ray.init(num_cpus=numberCPUS, logging_level=logging.ERROR)

#Allow partial GPU memory allocation
for gpu in tf.config.experimental.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(gpu, True)

#Check chosen regression model is available
if not erdModel in ['SLADS-LS', 'SLADS-Net', 'DLADS']: sys.exit('Error - Specified erdModel is not available')

#Define deployment for trained models
@serve.deployment(route_prefix="/ModelServer", ray_actor_options={"num_gpus": numGPUs})
class ModelServer:

    def __init__(self, erdModel, model_path):
        self.erdModel = erdModel
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': self.model = np.load(model_path, allow_pickle=True).item()
        elif erdModel == 'DLADS': self.model = tf.function(tf.keras.models.load_model(model_path), experimental_relax_shapes=True)

    def __call__(self, data):
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': return self.model.predict(data)
        elif erdModel == 'DLADS': return self.model(data, training=False)[0,:,:,0].numpy()

#PATH/DIRECTORY SETUP
#==================================================================

#Data input directories
dir_InputData = '.' + os.path.sep + 'INPUT' + os.path.sep
dir_TrainingData = dir_InputData + 'TRAIN' + os.path.sep
dir_TestingData = dir_InputData + 'TEST' + os.path.sep
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

#Clear the screen
os.system('cls' if os.name=='nt' else 'clear')
