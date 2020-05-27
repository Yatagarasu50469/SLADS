#==================================================================
#TRAINING SLADS SPECIFIC  
#==================================================================

#Sample information object for training the regression model
class TrainingSample:
    def __init__(self, name, images, maskObject, massRanges, measurementPerc, polyFeatures, reconImage, orderForRD, RD):
        self.name = name
        self.images = images
        self.maskObject = maskObject
        self.massRanges = massRanges
        self.measurementPerc = measurementPerc
        self.polyFeatures = polyFeatures
        self.reconImage = reconImage
        self.orderForRD = orderForRD
        self.RD = RD


def cGaussian_parhelper(cNum, sigma, windowSize, orderForRD, imgAsBlocksOnlyUnmeasured):
    
    #For each of the selected unmeasured points calculate the captured "area"
    temp = np.zeros((windowSize[0]*windowSize[1], len(orderForRD)))
    for index in range(0,len(orderForRD)): temp[:,index] = imgAsBlocksOnlyUnmeasured[orderForRD[index],:]*generateGaussianKernel(sigma[orderForRD[index]], windowSize)
    
    return cNum, np.sum(temp, axis=0)

#Given a set of file names, perform the initial setup for generating SLADS Model(s)
def initTrain(sortedTrainingSampleFolders):
    
    #Set function for the pool
    with contextlib.redirect_stdout(None):
        cGaussian_parFunction = ray.remote(cGaussian_parhelper)
        time.sleep(1)

    #Create a set of training samples for each possible c Value
    trainingDatabase = []
    for cNum in range(0,len(cValues)): trainingDatabase.append([])

    #For each physical sample, generate a training example for
    trainingSamples = []
    for sampleNum in tqdm(range(0,len(sortedTrainingSampleFolders)), desc = 'Training Samples', position=0, ascii=True):
        trainingSampleFolder = sortedTrainingSampleFolders[sampleNum]

        dataSampleName = os.path.basename(trainingSampleFolder)
        images = []
        massRanges = []

        #Import the sample's pixel aspect ratio (width, height)
        aspectRatio = np.loadtxt(trainingSampleFolder+'/aspect.txt', delimiter=',')

        images = []
        massRanges = []
        #Import each of the images according to their mz range order
        for imageFileName in natsort.natsorted(glob.glob(trainingSampleFolder + '/*.csv'), reverse=False):
            image = np.nan_to_num(np.loadtxt(imageFileName, delimiter=','))
            height, width = image.shape
            
            #Whichever dimension is the smaller leave alone, but resize the other according to the aspect ratio
            if resizeAspect:
                if width > height:
                    image = cv2.resize((image), (int(round((aspectRatio[0]/aspectRatio[1])*height)), height), interpolation = cv2.INTER_NEAREST)
                elif height > width:
                    image = cv2.resize((image), (width, int(round((aspectRatio[0]/aspectRatio[1])*width))), interpolation = cv2.INTER_NEAREST)
        
            images.append(image)
            massRanges.append([os.path.basename(imageFileName)[2:10], os.path.basename(imageFileName)[11:19]])
        maskObject = MaskObject(images[0].shape[1], images[0].shape[0], measurementPercs, numMasks)

        #Save copies of the measurement masks for validation; DEBUG
        #for measurementPercNum in range(0,len(measurementPercs)):
            #for maskNum in range(0,numMasks):
                #plt.imshow(maskObject.percMasks[measurementPercNum][maskNum], aspect='auto', cmap='gray')
                #plt.savefig(dir_TrainingResultsImages+ 'mask_'+ dataSampleName + '_percentage_' + str(measurementPercs[measurementPercNum]) + '_variation_' + str(maskNum) + '.png', bbox_inches='tight')
                #plt.close()

        #Weight images equally
        mzWeights = np.ones(len(images))/len(images)

        #Define a new sample
        sample = Sample(dataSampleName, images, massRanges, maskObject, mzWeights, dir_TrainingResults)

        #Append the basic information for each of the provided samples for use in determining best c Value
        trainingSamples.append(sample)

        #For each of the measurement percentages, extract features and initial RD values for each image
        for measurementPercNum in tqdm(range(0,len(measurementPercs)), desc = 'Measurement %', position=1, leave=False, ascii=True):
            measurementPerc = measurementPercs[measurementPercNum]
            for maskNum in tqdm(range(0,numMasks), desc = 'Masks', position=2, leave=False, ascii=True):
                
                #Set mask in maskObject and extract needed measurement indices
                maskObject.mask = maskObject.percMasks[measurementPercNum][maskNum]
                maskObject.measuredIdxs = maskObject.measuredIdxsList[measurementPercNum][maskNum]
                maskObject.unMeasuredIdxs = maskObject.unMeasuredIdxsList[measurementPercNum][maskNum]

                #Find neighbor information
                neighborIndices, neighborWeights, neighborDistances = findNeighbors(info, maskObject, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)

                #Calculate the sigma values for each possible c
                sigmaValues = [(neighborDistances[:,0]/c) for c in cValues]
                #for c in cValues: sigmaValues.append(neighborDistances[:,0]/c)

                #Flatten 2D mask array to 1D
                maskVect = np.ravel(maskObject.mask)

                #Use all of the unmeasured points for RD approximation
                orderForRD = np.arange(0,len(maskObject.unMeasuredIdxs)).tolist()
                
                #Set the measured images for the sample
                for imageNum in range(0,len(sample.measuredImages)):
                    temp = np.asarray(sample.images[imageNum]).copy()
                    temp[maskObject.mask == 0] = 0
                    sample.measuredImages[imageNum] = temp.copy()

                #Compute reconstruction
                reconImage = computeRecons(info, sample, maskObject, True)

                #Determine the feature vector for the reconstruction
                polyFeatures = computeFeatures(maskObject, sample, info, reconImage)

                #Reset the measured images and measurement lists for the sample/mask to a blank state
                sample.measuredImages = [np.zeros([maskObject.width, maskObject.height]) for rangeNum in range(0,len(massRanges))]
                maskObject.measuredIdxs = []
                maskObject.unMeasuredIdxs = []

                #Compute the difference between the original and reconstructed images
                RDPP = computeDifference(sample.avgImage, reconImage, info.imageType)

                #Convert differences to int
                RDPP.astype(int)

                #Pad with zeros
                RDPPWithZeros = np.pad(RDPP, [(int(np.floor(windowSize[0]/2)), ), (int(np.floor(windowSize[1]/2)), )], mode='constant')

                #Convert into a series of blocks and isolate unmeasured points in those blocks
                imgAsBlocks = viewW(RDPPWithZeros, (windowSize[0],windowSize[1])).reshape(-1,windowSize[0]*windowSize[1])[:,::1]
                imgAsBlocksOnlyUnmeasured = imgAsBlocks[np.logical_not(maskVect),:]

                #Add static parameters to shared pool memory
                orderForRD_id = ray.put(orderForRD)
                imgAsBlocksOnlyUnmeasured_id = ray.put(imgAsBlocksOnlyUnmeasured)
                windowSize_id = ray.put(windowSize)

                #Perform pool function and extract variables from the results
                idens = [cGaussian_parFunction.remote(cNum, sigmaValues[cNum], windowSize_id, orderForRD_id, imgAsBlocksOnlyUnmeasured_id) for cNum in range(0, len(cValues))]
                for result in tqdm(parIterator(idens), total=len(idens), position=3, desc='c Values', leave=False, ascii=True):
                    trainingDatabase[result[0]].append(TrainingSample(dataSampleName, images, maskObject, massRanges, measurementPerc, polyFeatures, reconImage, orderForRD, result[1]))

    #Save the basic trainingSamples data for finding the best C
    #pickle.dump(trainingSamples, open(dir_TrainingResults + 'trainingSamples.p', 'wb'))

    return trainingSamples, trainingDatabase

#Given a training database, create SLADS Model(s), noting there exists a single training sample for numCValues*numMeasurementPercs*numTrainingSamples
def trainModel(trainingDatabase):

    #Find a SLADS model for each of the c values
    trainingModels = []
    for cNum in tqdm(range(0,len(cValues)), desc = 'c Values', leave = True, ascii=True): #For each of the proposed c values
        trainingDataset = trainingDatabase[cNum]
        for sampleNum in tqdm(range(0,len(trainingDataset)), desc = 'Training Data', leave = False, ascii=True):
            trainingSample = trainingDataset[sampleNum]      

            if sampleNum == 0: #First loop
                if info.imageType == 'C':
                    bigPolyFeatures = trainingSample.polyFeatures
                    bigRD = trainingSample.RD
                elif info.imageType == 'D':
                    bigPolyFeatures = np.column_stack((trainingSample.polyFeatures[:,0:25], trainingSample.polyFeatures[:,26]))
                    bigRD = trainingSample.RD
            else: #Subsequent loops
                if info.imageType == 'C':
                    bigPolyFeatures = np.row_stack((bigPolyFeatures, trainingSample.polyFeatures))
                    bigRD = np.append(bigRD, trainingSample.RD)
                elif info.imageType == 'D':
                    tempPolyFeatures = np.column_stack((trainingSample.polyFeatures[:,0:25], trainingSample.polyFeatures[:,26]))
                    bigPolyFeatures = np.row_stack((bigPolyFeatures, tempPolyFeatures))
                    bigRD = np.append(bigRD, trainingSample.RD)

        #Create least-squares regression model
        regr = linear_model.LinearRegression(normalize=True, n_jobs=num_threads)
        regr.fit(bigPolyFeatures, bigRD)
        trainingModels.append(regr)

    #Save the end models and the matched cValue order array
    np.save(dir_TrainingResults + 'cValues', cValues)
    np.save(dir_TrainingResults + 'trainedModels', trainingModels)

    return trainingModels

#Given SLADS Model(s) determine the c value that minimizes the total distortion in scanned samples
def findBestC(trainingSamples, trainingModels):
    
    #Set function for the pool
    with contextlib.redirect_stdout(None):
        parFunction = ray.remote(runSLADS)
        time.sleep(1)

    #Add constant static parameters to shared pool memory
    info_id = ray.put(info)
    trainingSamples_id = ray.put(trainingSamples)
    stopPerc_id = ray.put(stopPerc)
    simulationFlag_id = ray.put(True)
    trainPlotFlag_id = ray.put(False)
    animationFlag_id = ray.put(False)
    bestCFlag_id = ray.put(True)
    tqdmHide_id = ray.put(False)

    #For each of the proposed c values determine which produces the overall minimal amount of toal distortion
    areaUnderCurveList = []
    for cNum in tqdm(range(0, len(cValues)), desc = 'c Values', position=0, leave=True, ascii=True):
        
        #Add revelant static parameter to shared pool memory
        trainingModel_id = ray.put(trainingModels[cNum])
        
        #Initialize rolling sum for total distortion levels
        areaUnderCurve = 0
        
        #DEBUG: Serial operation
        #for sampleNum in tqdm(range(0, len(trainingSamples)), desc = 'Training Samples', leave=False, ascii=True):
        #    areaResult = runSLADS(info, trainingSamples, trainingModels[cNum], stopPerc, sampleNum, True, False, False, False, True)
        #    areaUnderCurve += areaResult
    
        #Perform pool function and extract variables from the results
        idens = [parFunction.remote(info_id, trainingSamples_id, trainingModel_id, stopPerc_id, sampleNum, simulationFlag_id, trainPlotFlag_id, animationFlag_id, tqdmHide_id, bestCFlag_id) for sampleNum in range(0, len(trainingSamples))]
        for areaResult in tqdm(parIterator(idens), total=len(idens), desc='Training Samples', position=1, leave=False, ascii=True): areaUnderCurve += areaResult

        #Append the total distortion sum to a list corresponding to the c values
        areaUnderCurveList.append(areaUnderCurve)
    
    #Select the c value and corresponding model that minimizes the summed total distortion across the samples
    bestIndex = np.argmin(areaUnderCurveList)
    bestC = cValues[bestIndex]
    bestModel = trainingModels[bestIndex]

    # Find the Threshold on stopping condition that corresponds to the desired total distortion (TD) value set above
    if findStopThresh:   
        sys.error('ERROR! - Automatic determination of a stopping threshold has not yet been fully implemented!')
        #threshold = findStoppingThreshold(trainingDataPath,NumImagesForSLADS,Best_c,PercentageInitialMask,DesiredTD,reconPercVector,SizeImage)
        #np.save(dir_TrainingResults + 'foundThreshold', threshold) 

    #Save the best model
    np.save(dir_TrainingResults + 'bestC', bestC)
    print('bestC: ' + str(bestC))
    np.save(dir_TrainingResults + 'bestModel', bestModel)

    return bestC, bestModel





