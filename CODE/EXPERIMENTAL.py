#==================================================================
#IMPLEMENTATION SLADS SPECIFIC
#==================================================================

def performImplementation(dataSampleName):

    #Load the best model (if it has not been passed along from the training method)
    bestC = np.load(dir_TrainingResults + 'bestC.npy')
    bestTheta = np.load(dir_TrainingResults + 'bestTheta.npy')

    images = []
    massRanges = []
    for imageFileName in natsort.natsorted(glob.glob(testingSampleFolder + '/*.' + 'csv'), reverse=False):
        images.append(np.nan_to_num(np.loadtxt(imageFileName, delimiter=',')))
        massRanges.append([os.path.basename(imageFileName)[2:10], os.path.basename(imageFileName)[11:19]])
    
    #Read in the width and height
    height, width = images[0].shape
    
    #Create a new maskObject
    maskObject = MaskObject(width, height, measurementPercs=[])
    
    #How should the mz ranges be weighted (all equal for now)
    mzWeights = np.ones(len(images))/len(images)
    
    #Define information as a new Sample object
    impSample = Sample(dataSampleName, images, massRanges, maskObject, mzWeights, dir_ImpResults)
    

    sys.error('Error! - Implementation functionality for SLADS has not fully been incorporated at this time.')
    
    #Modify runSLADS regarding iterNum and simulationFlag checking
    runSLADS(info, impSample, maskObject, bestTheta, stopPerc, simulationFlag=False, trainPlotFlag=False, animationFlag=animationGen)

