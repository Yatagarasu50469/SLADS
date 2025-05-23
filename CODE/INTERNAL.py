#==================================================================
#INTERNAL - DIRECTORY SETUP
#==================================================================

#Indicate and setup the destination folder for results of this configuration
destResultsFolder = './RESULTS_'+os.path.splitext(os.path.basename(configFileName).split('_')[1])[0]

#If the folder already exists, either remove it, or append a novel value to it
if os.path.exists(destResultsFolder):
    if not preventResultsOverwrite: 
        shutil.rmtree(destResultsFolder)
    else: 
        destinationNameValue = 0
        destResultsFolder_Base = copy.deepcopy(destResultsFolder)
        while True:
            destResultsFolder = destResultsFolder_Base + '_' + str(destinationNameValue)
            if not os.path.exists(destResultsFolder): break
            destinationNameValue += 1

#Set a base model name for the specified configuration; must specify/append c value during run
modelName = 'model_'
modelName += erdModel.replace('-', '_') + '_'
if chanSingle: modelName += 'chanSingle_'
else: modelName += 'chanMultiple_'
if staticWindow: modelName += 'statWin_' + str(staticWindowSize) + '_'
if not staticWindow: modelName += 'dynWin_' + str(dynWindowSigMult) + '_'

#Data input directories
dir_InputData = '.' + os.path.sep + 'INPUT' + os.path.sep
if overrideInputsFolder != None: dir_InputData = overrideInputsFolder
dir_TrainingData = dir_InputData + 'TRAIN' + os.path.sep
dir_TestingData = dir_InputData + 'TEST' + os.path.sep
dir_PostData = dir_InputData + 'POST' + os.path.sep
dir_ImagesData = dir_InputData+'IMAGES' + os.path.sep
dir_ImagesTrainData = dir_ImagesData+'TRAIN' + os.path.sep
dir_ImagesTestData = dir_ImagesData+'TEST' + os.path.sep
if impInputDir == None:  dir_ImpData = dir_InputData + 'IMP' + os.path.sep
else: dir_ImpData = impInputDir

#Results directories
dir_Results = '.' + os.path.sep + 'RESULTS' + os.path.sep
if overrideResultsFolder != None: dir_Results = overrideResultsFolder
dir_TrainingResults = dir_Results + 'TRAIN' + os.path.sep
dir_TrainingModelResults = dir_TrainingResults + 'Model Training Images' + os.path.sep
dir_TrainingResultsImages = dir_TrainingResults + 'Training Data Images' + os.path.sep
dir_ValidationTrainingResultsImages = dir_TrainingResults + 'Validation Data Images' + os.path.sep
dir_ValidationResults = dir_Results + 'VALIDATION' + os.path.sep
dir_TestingResults = dir_Results + 'TEST' + os.path.sep
dir_ImpResults = dir_Results + 'IMP'+ os.path.sep
dir_PostResults = dir_Results + 'POST'+ os.path.sep

#Check that the result directory exists for cases where existing training data/model are to be used
if (not os.path.exists(dir_Results)) and (not trainingModel): sys.exit('\nError - dir_Results: ' + dir_Results + ' does not exist')
elif not os.path.exists(dir_Results): os.makedirs(dir_Results)

#Input data directories
if not os.path.exists(dir_InputData): sys.exit('\nError - dir_InputData does not exist')
if not os.path.exists(dir_TrainingData) and (trainingDataGen or trainingModel): sys.exit('\nError - dir_TrainingData: ' + dir_TrainingData + ' does not exist')
if not os.path.exists(dir_TestingData) and testingModel: sys.exit('\nError - dir_TestingData: ' + dir_TestingData + ' does not exist')
if not os.path.exists(dir_ImagesData) and (trainingDataGen or trainingModel):  sys.exit('\nError - dir_ImagesData: ' + dir_ImagesData + ' does not exist')
if not os.path.exists(dir_ImagesTrainData) and (trainingDataGen or trainingModel):  sys.exit('\nError - dir_ImagesTrainData: ' + dir_ImagesTrainData + ' does not exist')
if not os.path.exists(dir_ImagesTestData) and (trainingDataGen or trainingModel):  sys.exit('\nError - dir_ImagesTestData: ' + dir_ImagesTestData + ' does not exist')
if not os.path.exists(dir_ImpData) and impModel: sys.exit('\nError - dir_ImpData: ' + dir_ImpData + ' does not exist')
if not os.path.exists(dir_PostData) and postModel: sys.exit('\nError - dir_PostData: ' + dir_PostData + ' does not exist')

#As needed, reset the training directories
if (trainingDataGen or trainingModel) and not loadTrainValDatasets:
    if os.path.exists(dir_TrainingResults): shutil.rmtree(dir_TrainingResults)
    os.makedirs(dir_TrainingResults)
    if trainingDataGen:
        os.makedirs(dir_TrainingResultsImages)
        os.makedirs(dir_ValidationTrainingResultsImages)
    if trainingModel:
        os.makedirs(dir_TrainingModelResults)
if trainingModel and loadTrainValDatasets:
    if os.path.exists(dir_TrainingModelResults): shutil.rmtree(dir_TrainingModelResults)
    os.makedirs(dir_TrainingModelResults)
    
#Clear validation, testing, and implementation directories 
if os.path.exists(dir_ValidationResults): shutil.rmtree(dir_ValidationResults)
os.makedirs(dir_ValidationResults)

#If not using bypassSampling, then clear results directory, otherwise clear the encapsulated files, without deleting the actual directories
if not bypassSampling:
    if os.path.exists(dir_TestingResults): shutil.rmtree(dir_TestingResults)
    os.makedirs(dir_TestingResults)
else: 
    fileList = [file for file in glob.glob(dir_TestingResults+'**/*', recursive=True) if (not file.endswith(".p") and os.path.isfile(file))]
    for file in fileList: os.remove(file)

dir_ImpDataFinal = dir_ImpData + impSampleName + os.path.sep
if os.path.exists(dir_ImpDataFinal): shutil.rmtree(dir_ImpDataFinal)
os.makedirs(dir_ImpDataFinal)
if os.path.exists(dir_PostResults): shutil.rmtree(dir_PostResults)
os.makedirs(dir_PostResults)
