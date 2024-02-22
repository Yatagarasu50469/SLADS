#==================================================================
#MAIN
#==================================================================

#Import sys here rather than in EXTERNAL.py, as needed to parse input arguments
import sys

#Obtain the configuration file and version number from input variables
configFileName = sys.argv[1]
try: versionNum = sys.argv[2]
except: versionNum = 'N/A'

#Load in chosen configuration options; must be done first to set global variables for certain functions/methods correctly
exec(open(configFileName, encoding='utf-8').read())

#Import and setup external libraries
exec(open("./CODE/EXTERNAL.py", encoding='utf-8').read())

#Setup aesthetics
exec(open("./CODE/AESTHETICS.py", encoding='utf-8').read())

#Setup internal directory and naming conventions
exec(open("./CODE/INTERNAL.py", encoding='utf-8').read())

#Configure computation resources
exec(open("./CODE/COMPUTE.py", encoding='utf-8').read())

#Setup model definitions
exec(open("./CODE/MODEL_DLADS.py", encoding='utf-8').read())
#exec(open("./CODE/MODEL_GLANDS.py", encoding='utf-8').read())

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

#Generate training data if not already done so, optimizing c value if applicable
if trainingDataGen or trainingModel or validationModel:
    
    #If configured, try loading existing training/validation datasets, else generate such and optimize the c value from scratch
    if loadTrainValDatasets:
        try:
            trainingValidationSampleData = pickle.load(open(dir_TrainingResults + 'trainingValidationSampleData.p', "rb" ))
            validationSampleData = trainingValidationSampleData[int(trainingSplit*len(trainingValidationSampleData)):]
            trainingSampleData = trainingValidationSampleData[:int(trainingSplit*len(trainingValidationSampleData))]
            if erdModel != 'GLANDS': 
                trainingDatabase = pickle.load(open(dir_TrainingResults + 'trainingDatabase.p', "rb" ))
                validationDatabase = pickle.load(open(dir_TrainingResults + 'validationDatabase.p', "rb" ))
                optimalC = np.load(dir_TrainingResults + 'optimalC.npy', allow_pickle=True).item()
        except:
            loadTrainValDatasets = False
            print('\nWarning - Unable to load existing training/validation data files; will now construct from scratch.')

    if not loadTrainValDatasets:
    
        #Import training/validation data
        sectionTitle('IMPORTING TRAINING/VALIDATION SAMPLES')
        
        #Perform import and setup for training and validation datasets
        trainingValidationSampleData = importInitialData(natsort.natsorted(glob.glob(dir_TrainingData + '/*'), reverse=False))
        validationSampleData = trainingValidationSampleData[int(trainingSplit*len(trainingValidationSampleData)):]
        trainingSampleData = trainingValidationSampleData[:int(trainingSplit*len(trainingValidationSampleData))]
        
        #GLANDS does not require an optimized c, nor utilize a pre-generated databases
        if erdModel != 'GLANDS': 
        
            #Optimize the c value
            sectionTitle('OPTIMIZING C VALUE')
            optimalC = optimizeC(trainingValidationSampleData)
        
            #Generate a training database for the optimal c value and training samples
            sectionTitle('GENERATING TRAINING/VALIDATION DATASETS')
            trainingDatabase, validationDatabase = genTrainValDatabases(trainingValidationSampleData, optimalC)

#Perform model training or load one prepared earlier
if trainingModel:
    sectionTitle('PERFORMING TRAINING')
    if erdModel != 'GLANDS': modelName += 'c_' + str(optimalC)
    trainModel(trainingDatabase, validationDatabase, trainingSampleData, validationSampleData, modelName)
elif erdModel != 'GLANDS':
        optimalC = np.load(dir_TrainingResults + 'optimalC.npy', allow_pickle=True).item()
        modelName += 'c_' + str(optimalC)
    
#If a model needs to be simulated with validation data
if validationModel:
    sectionTitle('PERFORMING SIMULATION ON VALIDATION SET')
    simulateSampling([sampleData.sampleFolder for sampleData in validationSampleData], dir_ValidationResults, optimalC, modelName)

#If a model needs to be simulated with testing data
if testingModel:
    sectionTitle('PERFORMING SIMULATION ON TESTING SET')
    simulateSampling(natsort.natsorted(glob.glob(dir_TestingData + '/*'), reverse=False), dir_TestingResults, optimalC, modelName)

#If a model is to be used in an implementation
if impModel:
    sectionTitle('PERFORMING PHYSICAL EXPERIMENT')
    performImplementation(optimalC, modelName)

#If post-processing is to be performed
if postModel:
    sectionTitle('POST-PROCESSING SAMPLES')
    postprocess(natsort.natsorted(glob.glob(dir_PostData + '/*'), reverse=False), optimalC, modelName)

#Copy the results folder and the config file into it
resultCopy = shutil.copytree(dir_Results, destResultsFolder)
configCopy = shutil.copy(configFileName, destResultsFolder+'/'+os.path.basename(configFileName))

#Shutdown ray
if parallelization: _ = ray.shutdown()

#Notate the completion of intended operations
sectionTitle('CONFIGURATION COMPLETE')

#Make sure this process is closed
exit()
