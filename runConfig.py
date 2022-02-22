#==================================================================
#MAIN PROGRAM
#==================================================================

#Import all involved external libraries
exec(open("./CODE/EXTERNAL.py", encoding='utf-8').read())

#Obtain the configuration file and version number from input variables
configFileName = sys.argv[1]
try: versionNum = sys.argv[2]
except: versionNum = 'N/A'

#Load in chosen configuration options
exec(open(configFileName, encoding='utf-8').read())

#Import general method and class definitions (must be done after every configuration, otherwise global variables are not correct)
exec(open("./CODE/DEFS.py", encoding='utf-8').read())

#Setup directories and internal variables
exec(open("./CODE/INTERNAL.py", encoding='utf-8').read())

#Print out the program header
programTitle(versionNum, configFileName)

#Indicate the destination for the results of the configuration
destResultsFolder = './RESULTS_'+os.path.splitext(os.path.basename(configFileName).split('_')[1])[0]

#If the destination exists, output an error, or delete the folder
if preventResultsOverwrite: sys.exit('Error! - The destination results folder already exists')
elif os.path.exists(destResultsFolder): shutil.rmtree(destResultsFolder)

#Obtain the file paths for the intended training data if needed
if trainingModel or validationModel: trainValidationSamplePaths = natsort.natsorted(glob.glob(dir_TrainingData + '/*'), reverse=False)

#Train model if not already done so; otherwise load optimal c value
if trainingModel:
    
    #Import any specfic training function and class definitions
    exec(open("./CODE/TRAINING.py", encoding='utf-8').read())
    
    #If the dataset has not been generated then generate and find the best c for one, otherwise load the best c previously determined
    if not loadTrainValDatasets:
        
        #Import training/validation data
        sectionTitle('IMPORTING TRAINING/VALIDATION SAMPLES')
        
        #Perform import and setup for training and validation datasets
        trainingValidationSampleData = importInitialData(trainValidationSamplePaths)
        validationSampleData = trainingValidationSampleData[int(trainingSplit*len(trainingValidationSampleData)):]
        trainingSampleData = trainingValidationSampleData[:int(trainingSplit*len(trainingValidationSampleData))]
    
        #Optimize the c value
        sectionTitle('OPTIMIZING C VALUE')
        optimalC = optimizeC(trainingValidationSampleData)
        
        #Generate a training database for the optimal c value and training samples
        sectionTitle('GENERATING TRAINING/VALIDATION DATASETS')
        trainingDatabase, validationDatabase = generateDatabases(trainingValidationSampleData, optimalC)
        
    else:
        trainingValidationSampleData = pickle.load(open(dir_TrainingResults + 'trainingValidationSampleData.p', "rb" ))
        validationSampleData = trainingValidationSampleData[int(trainingSplit*len(trainingValidationSampleData)):]
        trainingSampleData = trainingValidationSampleData[:int(trainingSplit*len(trainingValidationSampleData))]
        optimalC = np.load(dir_TrainingResults + 'optimalC.npy', allow_pickle=True).item()
        trainingDatabase = pickle.load(open(dir_TrainingResults + 'trainingDatabase.p', "rb" ))
        validationDatabase = pickle.load(open(dir_TrainingResults + 'validationDatabase.p', "rb" ))

    #Train model(s) for the given database and c value
    sectionTitle('PERFORMING TRAINING')
    trainModel(trainingDatabase, validationDatabase, trainingSampleData, validationSampleData, optimalC)
else: 
    optimalC = np.load(dir_TrainingResults + 'optimalC.npy', allow_pickle=True).item()

#If needed import any specific testing function and class definitions
if validationModel or testingModel: exec(open("./CODE/SIMULATION.py", encoding='utf-8').read())

#If a model needs to be tested with validation data
if validationModel:
    sectionTitle('PERFORMING SIMULATION ON VALIDATION SET')
    
    if not trainingModel: 
        trainingValidationSampleData = pickle.load(open(dir_TrainingResults + 'trainingValidationSampleData.p', "rb" ))
        validationSampleData = trainingValidationSampleData[int(trainingSplit*len(trainingValidationSampleData)):]
        trainingSampleData = trainingValidationSampleData[:int(trainingSplit*len(trainingValidationSampleData))]
    
    #Obtain the file paths for the intended simulation data
    validationSamplePaths = [sampleData.sampleFolder for sampleData in validationSampleData]
    
    #Perform simulations
    simulateSLADS(validationSamplePaths, dir_ValidationResults, optimalC)

#If a model needs to be tested
if testingModel:
    sectionTitle('PERFORMING SIMULATION ON TESTING SET')
    
    #Obtain the file paths for the intended simulation data
    testSamplePaths = natsort.natsorted(glob.glob(dir_TestingData + '/*'), reverse=False)
    
    #Perform simulations
    simulateSLADS(testSamplePaths, dir_TestingResults, optimalC)

#If a model is to be used in an implementation
if impModel:
    sectionTitle('IMPLEMENTING MODEL')

    #Import any specific implementation function and class definitions
    exec(open("./CODE/EXPERIMENTAL.py", encoding='utf-8').read())

    #Begin performing an implementation
    performImplementation(optimalC)

#Copy the results folder and the config file into it
resultCopy = shutil.copytree('./RESULTS', destResultsFolder)
configCopy = shutil.copy(configFileName, destResultsFolder+'/'+os.path.basename(configFileName))

#Shutdown the ray and model server(s)
if parallelization: ray.shutdown()

#Notate the completion of intended operations
sectionTitle('PROGRAM COMPLETE')

#Make sure this process is closed
exit()
