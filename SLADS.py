#====================================================================
#PROGRAM INFORMATION
#====================================================================
#PROGRAM NAME:      SLADS
#
#DATE CREATED:	    4 October 2019
#
#DATE MODIFIED:	    10 December 2020
#
#VERSION NUM:	    0.7.4
#
#LICENSE:           GNU General Public License v3.0
#
#DESCRIPTION:	    Multichannel implementation of SLADS (Supervised Learning 
#                   Algorithm for Dynamic Sampling with additional constraint to
#                   select groups of points along a single axis.
#
#AUTHORS:           David Helminiak	EECE, Marquette University
#                   Dong Hye Ye		EECE, Marquette University
#
#COLLABORATORS:	    Julia Laskin	CHEM, Purdue University
#                   Hang Hu		CHEM, Purdue University
#
#FUNDING:	    This project has received funding and was programmed for:
#               NIH Grant 1UG3HL145593-01
#
#GLOBAL
#CHANGELOG:     0.1     Multithreading adjustments to pointwise SLADS
#               0.1.1    Line constraints, concatenation, pruning, and results organization
#               0.2     Line bounded constraints addition
#               0.3     Complete code rewrite, computational improvements
#               0.4     Class/function segmentation
#               0.5     Overhead reduction; switch multiprocessing package
#               0.6     Modifications for Nano-DESI microscope integration
#               0.6.1   Model robustness and reduction of memory overhead
#               0.6.2   Model loading and animation production patches
#               0.6.3   Start/End point selection with Canny
#               0.6.4   Custom knn metric, SSIM calc, init computations
#               0.6.5   Clean variables and resize to physical
#               0.6.6   SLADS-NET NN, PSNR, and multi-config
#               0.6.7   Clean asymmetric implementation with density features
#               0.6.8   Fixed RD generation, added metrics, and Windows compatible
#               0.7     CNN/Unet/RBDN with dynamic window size
#               0.7.1   c value selection performed before model training
#               0.7.2   Remove custom pkg. dependency, use NN resize, recon+measured input
#               0.7.3   Start/End line patch, SLADS(-Net) options, normalization optimization
#               0.6.9   Do not use -- Original SLADS(-Net) variations for comparison with 0.7.3
#               0.7.4   CPU compatibility patch, removal of NaN values
#               ~0.8    Multichannel integration
#               ~0.9    Tissue segmentation
#               ~1.0    Initial release
#====================================================================

#==================================================================
#MAIN PROGRAM
#==================================================================
#Current version information
versionNum="0.7.4"

#Import all involved external libraries (just once!)
exec(open("./CODE/EXTERNAL.py").read())

#For each of the configuration files that are present, run SLADS
for configFileName in natsort.natsorted(glob.glob('./CONFIG_*.py')):

    #Load in variable definitions from the configuration file
    exec(open(configFileName).read())

    #Import basic SLADS functions
    exec(open("./CODE/DEFS.py").read())

    #Setup directories and internal variables
    exec(open("./CODE/INTERNAL.py").read())

    sectionTitle("\n \
     ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄\n \
    ▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌\n \
    ▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀\n \
    ▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌\n \
    ▐░█▄▄▄▄▄▄▄▄▄ ▐░▌          ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄\n \
    ▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌\n \
     ▀▀▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌ ▀▀▀▀▀▀▀▀▀█░▌\n \
              ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌          ▐░▌\n \
     ▄▄▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌\n \
    ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌\n \
     ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀\n \
    Author(s):\tDavid Helminiak\t\tEECE Marquette University\n \
    \t\tDong Hye Ye\t\tEECE Marquette University\n \
    Licence:\tGNU General Public License v3.0\n \
    Version:\t"+versionNum+"\n \
    Config:\t"+os.path.splitext(os.path.basename(configFileName).split('_')[1])[0])

    destResultsFolder = './RESULTS_'+os.path.splitext(os.path.basename(configFileName).split('_')[1])[0]
    
    #If the destination exists, output an error, or delete the folder
    if preventResultsOverwrite:
        sys.exit('Error! - The destination results folder already exists')
    else:
        if os.path.exists(destResultsFolder): shutil.rmtree(destResultsFolder)

    #If a SLADS model needs to be trained
    if trainingModel:
        
        #Import any specfic training function and class definitions
        exec(open("./CODE/TRAINING.py").read())

        #Obtain the file paths for the intended training data
        trainSamplePaths = natsort.natsorted(glob.glob(dir_TrainingData + '/*'), reverse=False)

        #If the dataset has not been generated then generate and find the best c for one, otherwise load the best c previously determined
        if not loadTrainingDataset:
            sectionTitle('TRAINING DATABASE GENERATION')
            
            #Perform initial computations
            trainingSamples, trainingDatabase = initTrain(trainSamplePaths)
            
            sectionTitle('DETERMINING BEST C VALUE')
            
            #Optimize the c value
            bestC, cValues, bestCIndex = findBestC(trainingSamples, trainingDatabase)
        
        else:
            bestC = np.load(dir_TrainingResults + 'bestC.npy', allow_pickle=True).item()
            bestCIndex = np.load(dir_TrainingResults + 'bestCIndex.npy', allow_pickle=True).item()
            cValues = np.load(dir_TrainingResults + 'cValues.npy', allow_pickle=True)

        trainingSamples = pickle.load(open(dir_TrainingResults + 'trainingSamples.p', "rb" ))
        trainingDatabase = pickle.load(open(dir_TrainingResults + 'trainingDatabase.p', "rb" ))

        sectionTitle('TRAINING MODEL(S)')

        #Train model(s) for the given database and index of the best c value
        model = trainModel(trainingDatabase, cValues, bestCIndex)

    #If a new model shouldn't be generated, then the best c value should have already been selected
    if not trainingModel:
        bestC = np.load(dir_TrainingResults + 'bestC.npy', allow_pickle=True).item()
        bestCIndex = np.load(dir_TrainingResults + 'bestCIndex.npy', allow_pickle=True).item()
        cValues = np.load(dir_TrainingResults + 'cValues.npy', allow_pickle=True)
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net':
            model = np.load(dir_TrainingResults+'model_cValue_'+str(cValues[bestCIndex])+'.npy', allow_pickle=True).item()
        elif erdModel == 'DLADS':
            model = tf.keras.models.load_model(dir_TrainingResults+'model_cValue_'+str(bestC))

    #If a model needs to be tested
    if testingModel:
        sectionTitle('PERFORMING TESTING')

        #Import any specific testing function and class definitions
        exec(open("./CODE/TESTING.py").read())

        #Obtain the file pats for the intended testing data
        testSamplePaths = natsort.natsorted(glob.glob(dir_TestingData + '/*'), reverse=False)

        #Perform testing
        testSLADS(testSamplePaths, model, bestC)

    #If Leave-One-Out Cross Validation is to be performed
    if LOOCV:
        sectionTitle('PERFORMING LOOCV')
        sys.exit('Error! - LOOCV is not implemented at this time')

    #If a model is to be used in an implementation
    if impModel:
        sectionTitle('RUNNING EXPERIMENTAL MODEL')

        #Import any specific implementation function and class definitions
        exec(open("./CODE/EXPERIMENTAL.py").read())

        #Begin performing an implementation
        performImplementation(model, bestC)

    #Copy the results folder and the config file into it
    resultCopy = shutil.copytree('./RESULTS', destResultsFolder)
    configCopy = shutil.copy(configFileName, destResultsFolder+'/'+os.path.basename(configFileName))

    #Shutdown the ray server
    ray.shutdown()

    #AFTER INTENDED PROCEDURES (TRAINING/TESTING) HAVE BEEN PERFORMED
    sectionTitle('PROGRAM COMPLETE')

#Shutdown python kernel
exit()
