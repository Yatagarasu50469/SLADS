#==================================================================
#INTERNAL DIRECTORY SETUP
#==================================================================

#Indicate and setup the destination folder for results of this configuration
destResultsFolder = './RESULTS_'+os.path.splitext(os.path.basename(configFileName).split('_')[1])[0]
if preventResultsOverwrite: sys.exit('Error! - The destination results folder already exists')
elif os.path.exists(destResultsFolder): shutil.rmtree(destResultsFolder)

#Set a base model name for the specified configuration; must specify/append c value during run
modelName = 'model_'
if erdModel == 'SLADS-LS': modelName += 'SLADS-LS_'
elif erdModel == 'SLADS-Net': modelName += 'SLADS-Net_'
elif erdModel == 'DLADS': modelName += 'DLADS_'
if chanSingle: modelName += 'chanSingle_'
else: modelName += 'chanMultiple_'
if staticWindow: modelName += 'statWin_' + str(staticWindowSize) + '_'
if not staticWindow: modelName += 'dynWin_' + str(dynWindowSigMult) + '_'
if applyOptical != None: modelName += 'opt_' + applyOptical + '_'

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
dir_TrainingResults = dir_Results + 'TRAIN' + os.path.sep
dir_TrainingModelResults = dir_TrainingResults + 'Model Training Images' + os.path.sep
dir_TrainingResultsImages = dir_TrainingResults + 'Training Data Images' + os.path.sep
dir_ValidationTrainingResultsImages = dir_TrainingResults + 'Validation Data Images' + os.path.sep
dir_ValidationResults = dir_Results + 'VALIDATION' + os.path.sep
dir_TestingResults = dir_Results + 'TEST' + os.path.sep
dir_ImpResults = dir_Results + 'IMP'+ os.path.sep
dir_PostResults = dir_Results + 'POST'+ os.path.sep

#Check that the result directory exists for cases where existing training data/model are to be used
if (not os.path.exists(dir_Results)) and (not trainingModel): sys.exit('Error - dir_Results: ./RESULTS/ does not exist')
elif not os.path.exists(dir_Results): os.makedirs(dir_Results)

#Input data directories
if not os.path.exists(dir_InputData): sys.exit('Error - dir_InputData: ./INPUT/ does not exist')
if not os.path.exists(dir_TrainingData) and (trainingDataGen or trainingModel): sys.exit('Error - dir_TrainingData: ./INPUT/TRAIN/ does not exist')
if not os.path.exists(dir_TestingData) and testingModel: sys.exit('Error - dir_InputData: ./INPUT/TEST/ does not exist')
if not os.path.exists(dir_ImagesData) and (trainingDataGen or trainingModel):  sys.exit('Error - dir_ImagesData: ./INPUT/IMAGES/ does not exist')
if not os.path.exists(dir_ImagesTrainData) and (trainingDataGen or trainingModel):  sys.exit('Error - dir_ImagesTrainData: ./INPUT/IMAGES/TRAIN/ does not exist')
if not os.path.exists(dir_ImagesTestData) and (trainingDataGen or trainingModel):  sys.exit('Error - dir_ImagesTestData: ./INPUT/IMAGES/TEST/ does not exist')
if not os.path.exists(dir_ImpData) and impModel: sys.exit('Error - dir_ImpData: ./INPUT/IMP/ does not exist')
if not os.path.exists(dir_PostData) and postModel: sys.exit('Error - dir_PostData: ./INPUT/POST/ does not exist')

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
if os.path.exists(dir_TestingResults): shutil.rmtree(dir_TestingResults)
os.makedirs(dir_TestingResults)
dir_ImpDataFinal = dir_ImpData + impSampleName + os.path.sep
if os.path.exists(dir_ImpDataFinal): shutil.rmtree(dir_ImpDataFinal)
os.makedirs(dir_ImpDataFinal)
if os.path.exists(dir_PostResults): shutil.rmtree(dir_PostResults)
os.makedirs(dir_PostResults)
