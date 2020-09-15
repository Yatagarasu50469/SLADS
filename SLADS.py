#====================================================================
#PROGRAM INFORMATION
#====================================================================
#PROGRAM NAME:      SLADS
#
#DATE CREATED:	    4 October 2019
#
#DATE MODIFIED:	    14 July 2020
#
#VERSION NUM:	    0.6.8
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
#               ~0.7	CNN with dynamic window size
#               ~0.8	Multichannel integration
#               ~0.9	Tissue segmentation
#               ~1.0	Initial release
#====================================================================

#==================================================================
#MAIN PROGRAM
#==================================================================
#Current version information
versionNum="0.6.8"

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

        sectionTitle('PERFORMING INTITIAL TRAINING COMPUTATIONS')
        
        #Perform initial computations
        trainingSamples, trainingDatabase = initTrain(trainSamplePaths)

        sectionTitle('TRAINING MODEL(S)')

        #Generate SLADS model(s) for the given training database
        trainingModels = trainModel(trainingDatabase)

        sectionTitle('DETERMINING BEST MODEL')

        #Identify the best Model; saves to training results
        bestC, bestModel = findBestC(trainingSamples, trainingModels)

    #Load in the best model information if training wasn't performed
    if not trainingModel:
        bestC = np.load(dir_TrainingResults + 'bestC.npy', allow_pickle=True).item()
        bestModel = np.load(dir_TrainingResults + 'bestModel.npy', allow_pickle=True).item()

    #If a SLADS model needs to be tested
    if testingModel:
        sectionTitle('PERFORMING TESTING')

        #Import any specific testing function and class definitions
        exec(open("./CODE/TESTING.py").read())

        #Obtain the file pats for the intended testing data
        testSamplePaths = natsort.natsorted(glob.glob(dir_TestingData + '/*'), reverse=False)

        #Perform testing
        testSLADS(testSamplePaths, bestC, bestModel)

    #If Leave-One-Out Cross Validation is to be performed
    if LOOCV:
        sectionTitle('PERFORMING LOOCV')
        sys.exit('Error! - LOOCV is not implemented at this time')

    #If a SLADS model is to be used in an implementation
    if impModel:
        sectionTitle('PERFORMING SLADS')

        #Import any specific implementation function and class definitions
        exec(open("./CODE/EXPERIMENTAL.py").read())

        #Begin performing an implementation
        performImplementation(bestC, bestModel)

    sectionTitle('DUPLICATING RESULTS')
    
    #Copy the results folder
    destination = shutil.copytree('./RESULTS', destResultsFolder)
    
    #Pause to ensure system had time to finish the copy
    time.sleep(10)

    #Attempt to shutdown multiprocessing server
    try:
        ray.shutdown()
    except:
        sys.error('Error! - Ray failed to shutdown correctly')

    #AFTER INTENDED PROCEDURES (TRAINING/TESTING) HAVE BEEN PERFORMED
    sectionTitle('PROGRAM COMPLETE')

