#====================================================================
#PROGRAM INFORMATION
#====================================================================
#PROGRAM NAME:      SLADS
#
#DATE CREATED:	    4 October 2019
#
#DATE MODIFIED:	    24 March 2021
#
#VERSION NUM:	    0.8.2
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
#CHANGELOG:     0.1.0   Multithreading adjustments to pointwise SLADS
#               0.1.1    Line constraints, concatenation, pruning, and results organization
#               0.2.0   Line bounded constraints addition
#               0.3.0   Complete code rewrite, computational improvements
#               0.4.0   Class/function segmentation
#               0.5.0   Overhead reduction; switch multiprocessing package
#               0.6.0   Modifications for Nano-DESI microscope integration
#               0.6.1   Model robustness and reduction of memory overhead
#               0.6.2   Model loading and animation production patches
#               0.6.3   Start/End point selection with Canny
#               0.6.4   Custom knn metric, SSIM calc, init computations
#               0.6.5   Clean variables and resize to physical
#               0.6.6   SLADS-NET NN, PSNR, and multi-config
#               0.6.7   Clean asymmetric implementation with density features
#               0.6.8   Fixed RD generation, added metrics, and Windows compatible
#               0.7.0   CNN/Unet/RBDN with dynamic window size
#               0.7.1   c value selection performed before model training
#               0.7.2   Remove custom pkg. dependency, use NN resize, recon+measured input
#               0.7.3   Start/End line patch, SLADS(-Net) options, normalization optimization
#               0.6.9   Do not use -- Original SLADS(-Net) variations for comparison with 0.7.3
#               0.7.4   CPU compatibility patch, removal of NaN values
#               0.7.5   c value selection performed before training database generation
#               0.8.0   Raw MSI file integration (Thermo .raw, Agilent .d), only Windows compatible
#               0.8.1   Model simplification, method cleanup, mz tolerance/standard patch
#               0.8.2   Multichannel, fixed groupwise, square pixels, accelerated RD, altered visuals/metrics
#               ~0.8.3  GAN 
#               ~0.8.4  Custom adversarial network
#               ~0.9.0  Multimodal integration
#               ~1.0.0  Initial release
#====================================================================

#==================================================================
#MAIN PROGRAM
#==================================================================
#Current version information
versionNum='0.8.2'

#Import all involved external libraries (just once!)
exec(open("./CODE/EXTERNAL.py").read())

#For each of the configuration files that are present, run SLADS
for configFileName in natsort.natsorted(glob.glob('./CONFIG_*.py')):

    #Import basic definitions
    exec(open("./CODE/DEFS.py").read())

    #Load in variable definitions from the configuration file
    exec(open(configFileName).read())

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
            
            #Import training/validation data
            sectionTitle('IMPORTING TRAINING SAMPLES')
            trainingSamples = importInitialData(trainSamplePaths)
            
            #Optimize the c value
            sectionTitle('OPTIMIZING C VALUE')
            optimalC = optimizeC(trainingSamples)
            
            #Generate a training database for the optimal c value and training samples
            sectionTitle('GENERATING TRAINING DATASET')
            trainingDatabase = generateTrainingData(trainingSamples, optimalC)
            
        else:
            optimalC = np.load(dir_TrainingResults + 'optimalC.npy', allow_pickle=True).item()
            trainingDatabase = pickle.load(open(dir_TrainingResults + 'trainingDatabase.p', "rb" ))

        #Train model(s) for the given database and c value
        sectionTitle('PERFORMING TRAINING')
        model = trainModel(trainingDatabase, optimalC)

    #If a new model shouldn't be generated, then the best c value should have already been selected
    if not trainingModel:
        optimalC = np.load(dir_TrainingResults + 'optimalC.npy', allow_pickle=True).item()
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net':
            model = np.load(dir_TrainingResults+'model_cValue_'+str(optimalC)+'.npy', allow_pickle=True).item()
        elif erdModel == 'DLADS':
            model = tf.keras.models.load_model(dir_TrainingResults+'model_cValue_'+str(optimalC), custom_objects={'PSNR':PSNR})

    #If a model needs to be tested
    if testingModel:
        sectionTitle('PERFORMING TESTING')

        #Import any specific testing function and class definitions
        exec(open("./CODE/TESTING.py").read())

        #Obtain the file pats for the intended testing data
        testSamplePaths = natsort.natsorted(glob.glob(dir_TestingData + '/*'), reverse=False)

        #Perform testing
        testSLADS(testSamplePaths, model, optimalC)

    #If Leave-One-Out Cross Validation is to be performed
    if LOOCV:
        sectionTitle('PERFORMING LOOCV')
        sys.exit('Error! - LOOCV is not implemented at this time')

    #If a model is to be used in an implementation
    if impModel:
        sectionTitle('IMPLEMENTING MODEL')

        #Import any specific implementation function and class definitions
        exec(open("./CODE/EXPERIMENTAL.py").read())

        #Begin performing an implementation
        performImplementation(model, optimalC)

    #Copy the results folder and the config file into it
    resultCopy = shutil.copytree('./RESULTS', destResultsFolder)
    configCopy = shutil.copy(configFileName, destResultsFolder+'/'+os.path.basename(configFileName))

    #Shutdown the ray server
    ray.shutdown()

    #AFTER INTENDED PROCEDURES (TRAINING/TESTING) HAVE BEEN PERFORMED
    sectionTitle('PROGRAM COMPLETE')

#Shutdown python kernel
exit()
