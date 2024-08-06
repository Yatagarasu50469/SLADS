#==================================================================
#MAIN
#==================================================================

#Import sys here rather than in EXTERNAL.py, as needed to parse input arguments
import sys

#Obtain the configuration file and version number from input variables
configFileName = sys.argv[1]
dir_tmp = sys.argv[2]
try: versionNum = sys.argv[3]
except: versionNum = 'N/A'

#Load in chosen configuration options; must be done first to set global variables for certain functions/methods correctly
exec(open(configFileName, encoding='utf-8').read())

#Import and setup external libraries
exec(open("./CODE/EXTERNAL.py", encoding='utf-8').read())

#Setup any configuration-derived global variables
exec(open("./CODE/DERIVED.py", encoding='utf-8').read())

#Setup aesthetics
exec(open("./CODE/AESTHETICS.py", encoding='utf-8').read())

#Setup internal directories and naming conventions
exec(open("./CODE/INTERNAL.py", encoding='utf-8').read())

#Configure computation resources
exec(open("./CODE/COMPUTE.py", encoding='utf-8').read())

#Setup model definitions
exec(open("./CODE/MODEL_SLADS.py", encoding='utf-8').read()) #computePolyFeatures and associated data really should be segmented out of DEFINITIONS; run at training or inference time
if erdModel=='DLADS': exec(open("./CODE/MODEL_DLADS.py", encoding='utf-8').read())
elif erdModel=='DLADS-TF': exec(open("./CODE/MODEL_DLADS_TF.py", encoding='utf-8').read())
elif erdModel=='DLADS-PY': exec(open("./CODE/MODEL_DLADS_PY.py", encoding='utf-8').read())
elif erdModel =='GLANDS': exec(open("./CODE/MODEL_GLANDS.py", encoding='utf-8').read())
if overWriteModelFile != None: exec(open(overWriteModelFile, encoding='utf-8').read())

#Import local and remote method/class definitions as needed
exec(open("./CODE/DEFINITIONS.py", encoding='utf-8').read())
exec(open("./CODE/REMOTE.py", encoding='utf-8').read())
if trainingDataGen or trainingModel or validationModel: exec(open("./CODE/TRAINING.py", encoding='utf-8').read())
if validationModel or testingModel: exec(open("./CODE/SIMULATION.py", encoding='utf-8').read())
if impModel: exec(open("./CODE/EXPERIMENTAL.py", encoding='utf-8').read())
if postModel: exec(open("./CODE/POSTPROCESS.py", encoding='utf-8').read())
if overWriteFile != None: exec(open(overWriteFile, encoding='utf-8').read())

#Convert any training and testing images into compatible samples
processImages(dir_TrainingData, natsort.natsorted(glob.glob(dir_ImagesTrainData+'*')), 'Training')
processImages(dir_TestingData, natsort.natsorted(glob.glob(dir_ImagesTestData+'*')), 'Testing')

#GLANDS model does not require an optimized c value, nor pre-generated databases
if erdModel == 'GLANDS': 
    optimalC = None
    trainingDatabase = None
    validationDatabase = None

#Generate or load training data and optimal c value
if trainingDataGen or trainingModel or validationModel:
    
    #If configured, try loading existing training/validation datasets
    if loadTrainValDatasets:
        try:
            trainingValidationSampleData = pickle.load(open(dir_TrainingResults + 'trainingValidationSampleData.p', "rb" ))
            validationSampleData = trainingValidationSampleData[int(trainingSplit*len(trainingValidationSampleData)):]
            trainingSampleData = trainingValidationSampleData[:int(trainingSplit*len(trainingValidationSampleData))]
            if erdModel != 'GLANDS': 
                optimalC = np.load(dir_TrainingResults + 'optimalC.npy', allow_pickle=True).item()
                modelName += 'c_' + str(optimalC)
                trainingDatabase = pickle.load(open(dir_TrainingResults + 'trainingDatabase.p', "rb" ))
                validationDatabase = pickle.load(open(dir_TrainingResults + 'validationDatabase.p', "rb" ))
        except:
            loadTrainValDatasets = False
            print('\nWarning - Unable to load existing training/validation data files; will now construct from scratch.')

    #If not configured or unable to load existing datasets, generate such and optimize the c value
    if not loadTrainValDatasets:
    
        #Import training/validation data
        sectionTitle('IMPORTING TRAINING/VALIDATION SAMPLES')
        
        #Perform import and setup for training and validation datasets
        trainingValidationSampleData = importInitialData(natsort.natsorted(glob.glob(dir_TrainingData + '/*'), reverse=False))
        _ = gc.collect()
        validationSampleData = trainingValidationSampleData[int(trainingSplit*len(trainingValidationSampleData)):]
        trainingSampleData = trainingValidationSampleData[:int(trainingSplit*len(trainingValidationSampleData))]
        
        #GLANDS does not require an optimized c, nor utilize a pre-generated databases
        if erdModel != 'GLANDS': 
            
            #Optimize the c value
            sectionTitle('OPTIMIZING C VALUE')
            optimalC = optimizeC(trainingValidationSampleData)
            _ = gc.collect()
            modelName += 'c_' + str(optimalC)
            
            #Generate a training database for the optimal c value and training samples
            sectionTitle('GENERATING TRAINING/VALIDATION DATASETS')
            trainingDatabase, validationDatabase = genTrainValDatabases(trainingValidationSampleData, optimalC)
            _ = gc.collect()
else:
    optimalC = np.load(dir_TrainingResults + 'optimalC.npy', allow_pickle=True).item()
    modelName += 'c_' + str(optimalC)

#Perform model training as configured
if trainingModel:
    sectionTitle('PERFORMING TRAINING')
    trainModel(trainingDatabase, validationDatabase, trainingSampleData, validationSampleData, modelName)

#Clear large variables from memory that are no longer needed
try: del trainingValidationSampleData
except: pass
try: del trainingDatabase
except: pass
try: del validationDatabase
except: pass
try: del trainingSampleData
except: pass
_ = gc.collect()

#If a model needs to be simulated with validation data
if validationModel:
    sectionTitle('PERFORMING SIMULATION ON VALIDATION SET')
    simulateSampling([sampleData.sampleFolder for sampleData in validationSampleData], dir_ValidationResults, optimalC, modelName)

#Clear large variables from memory that are no longer needed
try: del validationSampleData
except: pass
_ = gc.collect()

#If a model needs to be simulated with testing data
if testingModel:
    sectionTitle('PERFORMING SIMULATION ON TESTING SET')
    simulateSampling(natsort.natsorted(glob.glob(dir_TestingData + '/*'), reverse=False), dir_TestingResults, optimalC, modelName)
    _ = gc.collect()
    
#If a model is to be used in an implementation
if impModel:
    sectionTitle('PERFORMING PHYSICAL EXPERIMENT')
    performImplementation(optimalC, modelName)
    _ = gc.collect()

#If post-processing is to be performed
if postModel:
    sectionTitle('POST-PROCESSING SAMPLES')
    postprocess(natsort.natsorted(glob.glob(dir_PostData + '/*'), reverse=False), optimalC, modelName)
    _ = gc.collect()

#Shutdown ray
if parallelization: 
    _ = ray.shutdown()
    rayUp=False

#Copy the results folder, the config file and ray log directory into it if applicable
_ = shutil.copytree(dir_Results, destResultsFolder)
_ = shutil.copy(configFileName, destResultsFolder+'/'+os.path.basename(configFileName))
if debugMode: _ = shutil.copytree(dir_tmp, destResultsFolder+'/TMP')

#Notate the completion of intended operations
sectionTitle('CONFIGURATION COMPLETE')

#Make sure this process is closed
exit()
