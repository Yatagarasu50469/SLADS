#==================================================================
#INTERNAL SLADS SETUP
#==================================================================

#INTERNAL OBJECT SETUP
#==================================================================
#Initialize the information object for training and testing
#(reconMethod, featReconMethod, neighborWeightsPower, numNeighbors, filterType, featDistCutoff, resolution)
info = Info('CWM', 'CWM', 2, 10, 'Gaussian', 0.25, 1, imageType)

#Setup parallization pool
ray.init(num_cpus=num_threads)

#PATH/DIRECTORY SETUP
#==================================================================

#Data input directories
dir_InputData = '.' + os.path.sep + 'INPUT' + os.path.sep
dir_TrainingData = dir_InputData + 'TRAIN' + os.path.sep
dir_TestingData = dir_InputData + 'TEST' + os.path.sep
dir_ImpData = dir_InputData + 'IMP' + os.path.sep

#Results directories
dir_Results = '.' + os.path.sep + 'RESULTS' + os.path.sep
dir_TrainingResults = dir_Results + 'TRAIN' + os.path.sep
dir_TrainingResultsImages = dir_TrainingResults + 'Images' + os.path.sep
dir_TestingResults = dir_Results + 'TEST' + os.path.sep
dir_TestingResultsImages = dir_TestingResults + 'Images' + os.path.sep
dir_ImpResults = dir_Results + 'IMP'+ os.path.sep
dir_ImpResultsImages = dir_ImpResults + 'Images' + os.path.sep

#Check general directory structure
if not os.path.exists(dir_Results): sys.exit('Error - dir_Results does not exist')

#Input data directories
if not os.path.exists(dir_InputData): sys.exit('Error - dir_InputData does not exist')
if not os.path.exists(dir_TrainingData) and trainingModel: sys.exit('Error - dir_TrainingData does not exist')
if not os.path.exists(dir_TestingData) and testingModel: sys.exit('Error - dir_InputData does not exist')
if not os.path.exists(dir_ImpData) and impModel: sys.exit('Error - dir_ImpData does not exist')

#Results directories - reset results folders for new runs
if not os.path.exists(dir_Results): sys.exit('Error - dir_Results does not exist')

#As needed, reset the results' directories
if trainingModel:
    if os.path.exists(dir_TrainingResults): shutil.rmtree(dir_TrainingResults)
    os.makedirs(dir_TrainingResults)
    os.makedirs(dir_TrainingResultsImages)    
    
if testingModel:
    if os.path.exists(dir_TestingResults): shutil.rmtree(dir_TestingResults)
    os.makedirs(dir_TestingResults)

if impModel:
    if os.path.exists(dir_ImpResults): shutil.rmtree(dir_ImpResults)
    os.makedirs(dir_ImpResults)
    dir_ImpDataFinal = dir_ImpData + impSampleName + os.path.sep
    if os.path.exists(dir_ImpDataFinal): shutil.rmtree(dir_ImpDataFinal)
    os.makedirs(dir_ImpDataFinal)

if animationGen:
    dir_Animations = dir_TestingResults + 'Animations/'
    dir_AnimationVideos = dir_Animations + 'Videos/'
    dir_mzResults = dir_TestingResults + 'mzResults/'
    
    if os.path.exists(dir_Animations): shutil.rmtree(dir_Animations)    
    os.makedirs(dir_Animations)
    if os.path.exists(dir_AnimationVideos): shutil.rmtree(dir_AnimationVideos)    
    os.makedirs(dir_AnimationVideos)
    if os.path.exists(dir_mzResults): shutil.rmtree(dir_mzResults)    
    os.makedirs(dir_mzResults)

#AESTHETIC SETUP
#==================================================================
#Clear the screen
os.system('cls' if os.name=='nt' else 'clear')

#Determine console size if applicable
if consoleRunning:
    consoleRows, consoleColumns = os.popen('stty size', 'r').read().split()
else:
    consoleRows, consoleColumns = 40, 40
