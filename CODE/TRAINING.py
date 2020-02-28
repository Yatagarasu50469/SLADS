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

def imFeature_parhelper(measuredValues, neighborIndices, info, maskObject, unMeasuredIdxs, measuredIdxs, neighborWeights, neighborDistances, orderForRD):
    
    #Find neighborhood values
    neighborValues = findNeighborValues(measuredValues, neighborIndices)
    
    #Compute reconstructions
    reconValues, reconImage = computeRecons(info, maskObject, unMeasuredIdxs, measuredIdxs, neighborValues, neighborWeights, measuredValues)
    
    #Compute features
    polyFeatures = computeFeatures(unMeasuredIdxs, maskObject.area, neighborValues, neighborWeights, neighborDistances, info, reconValues, reconImage)
    
    #Extract set of the polyFeatures
    polyFeatures = polyFeatures[orderForRD,:]
    
    return reconImage, polyFeatures

def cGaussian_parhelper(cNum, sigma, windowSize, orderForRD, imgAsBlocksOnlyUnmeasured):
    
    #For each of the selected unmeasured points calculate the captured "area"
    temp = np.zeros((windowSize[0]*windowSize[1], len(orderForRD)))
    for index in range(0,len(orderForRD)): temp[:,index] = imgAsBlocksOnlyUnmeasured[orderForRD[index],:]*generateGaussianKernel(sigma[orderForRD[index]], windowSize)
    RD = np.sum(temp, axis=0)
    
    return cNum, RD

#Given a set of file names, perform the initial setup for generating SLADS Model(s)
def initTrain(sortedTrainingSampleFolders):
    
    #Set function for the pool
    with contextlib.redirect_stdout(None):
        imFeature_parFunction = ray.remote(imFeature_parhelper)
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

        #Import each of the images according to their mz range order
        for imageFileName in natsort.natsorted(glob.glob(trainingSampleFolder + '/*.' + 'csv'), reverse=False):
            images.append(np.nan_to_num(np.loadtxt(imageFileName, delimiter=',')))
            massRanges.append([os.path.basename(imageFileName)[2:10], os.path.basename(imageFileName)[11:19]])
        maskObject = MaskObject(images[0].shape[1], images[0].shape[0], measurementPercs, numMasks)
        #for measurementPercNum in range(0,len(measurementPercs)):
            #for maskNum in range(0,numMasks):
                #plt.imshow(maskObject.percMasks[measurementPercNum][maskNum], aspect='auto', cmap='gray')
                #plt.savefig(dir_TrainingResultsImages+ 'mask_'+ dataSampleName + '_percentage_' + str(measurementPercs[measurementPercNum]) + '_variation_' + str(maskNum) + '.png', bbox_inches='tight')
                #plt.close()

        #Weight images equally
        mzWeights = np.ones(len(images))/len(images)

        #Append the basic information for each of the provided samples for use in determining best c Value
        trainingSamples.append(Sample(dataSampleName, images, massRanges, maskObject, mzWeights, dir_TrainingResults))

        #For each of the measurement percentages, extract features and initial RD values for each image
        for measurementPercNum in tqdm(range(0,len(measurementPercs)), desc = 'Measurement %', position=1, leave=False, ascii=True):
            measurementPerc = measurementPercs[measurementPercNum]
            for maskNum in tqdm(range(0,numMasks), desc = 'Masks', position=2, leave=False, ascii=True):
                
                #Retreive relevant mask information
                mask = maskObject.percMasks[measurementPercNum][maskNum]
                measuredIdxs = maskObject.measuredIdxsList[measurementPercNum][maskNum]
                unMeasuredIdxs = maskObject.unMeasuredIdxsList[measurementPercNum][maskNum]

                #Find neighbor information
                neighborIndices, neighborWeights, neighborDistances = findNeighbors(info, measuredIdxs, unMeasuredIdxs)

                #Calculate the sigma values for each possible c
                sigmaValues = []
                for c in cValues: sigmaValues.append(neighborDistances/c)

                #Flatten 2D mask array to 1D
                maskVect = np.ravel(mask)

                #Use all of the unmeasured points for RD approximation
                orderForRD = np.arange(0,len(unMeasuredIdxs)).tolist()

                #Setup holding arrays for reconImages and polyFeatures
                reconImageList = []
                polyFeaturesList = []

                #Add static parameters to shared pool memory
                neighborIndices_id = ray.put(neighborIndices)
                info_id = ray.put(info)
                maskObject_id = ray.put(maskObject)
                unMeasuredIdxs_id = ray.put(unMeasuredIdxs)
                measuredIdxs_id = ray.put(measuredIdxs)
                neighborWeights_id = ray.put(neighborWeights)
                neighborDistances_id = ray.put(neighborDistances)
                orderForRD_id = ray.put(orderForRD)

                #Perform pool function and extract variables from the results
                idens = [imFeature_parFunction.remote(np.asarray(images[imNum])[mask==1], neighborIndices_id, info_id, maskObject_id, unMeasuredIdxs_id, measuredIdxs_id, neighborWeights_id, neighborDistances_id, orderForRD_id) for imNum in range(0, len(images))]
                for result in tqdm(parIterator(idens), total=len(idens), position=3, desc='Poly Features', leave=False, ascii=True):
                    reconImageList.append(result[0])
                    polyFeaturesList.append(result[1])

                #Average the reconstructions according to the mzWeights
                reconImage = np.average(np.asarray(reconImageList), axis=0, weights=mzWeights)

                #Average the ground-truth according to the mzWeights
                avgImage = np.average(np.asarray(images), axis=0, weights=mzWeights)

                #Average the polyFeatures according to the mzWeights
                polyFeatures = np.average(np.asarray(polyFeaturesList), axis=0, weights=mzWeights)

                #Compute the difference between the original and reconstructed images
                RDPP = computeDifference(avgImage, reconImage, info.imageType)

                #Convert differences to int
                RDPP.astype(int)

                #Pad with zeros
                RDPPWithZeros = np.pad(RDPP, [(int(np.floor(windowSize[0]/2)), ), (int(np.floor(windowSize[1]/2)), )], mode='constant')

                #Convert into a series of blocks and isolate unmeasured points in those blocks
                imgAsBlocks = viewW(RDPPWithZeros, (windowSize[0],windowSize[1])).reshape(-1,windowSize[0]*windowSize[1])[:,::1]
                imgAsBlocksOnlyUnmeasured = imgAsBlocks[np.logical_not(maskVect),:]

                #Add additional static parameters to shared pool memory
                imgAsBlocksOnlyUnmeasured_id = ray.put(imgAsBlocksOnlyUnmeasured)
                windowSize_id = ray.put(windowSize)

                #Perform pool function and extract variables from the results
                idens = [cGaussian_parFunction.remote(cNum, sigmaValues[cNum], windowSize_id, orderForRD_id, imgAsBlocksOnlyUnmeasured_id) for cNum in range(0, len(cValues))]
                for result in tqdm(parIterator(idens), total=len(idens), position=3, desc='c Values', leave=False, ascii=True):
                    trainingDatabase[result[0]].append(TrainingSample(dataSampleName, images, maskObject, massRanges, measurementPerc, polyFeatures, reconImage, orderForRD, result[1]))

                #Manually run garbage collection
                gc.collect()

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

    #Manually run garbage collection
    gc.collect()
    
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
    tqdmHide_id = ray.put(True)

    #For each of the proposed c values determine which produces the overall minimal amount of toal distortion
    areaUnderCurveList = []
    for cNum in tqdm(range(0, len(cValues)), desc = 'c Values', position=0, leave=True, ascii=True):
        
        #Add revelant static parameter to shared pool memory
        trainingModel_id = ray.put(trainingModels[cNum])
        
        #Initialize rolling sum for total distortion levels
        areaUnderCurve = 0
        
        #Perform pool function and extract variables from the results
        idens = [parFunction.remote(info_id, trainingSamples_id, trainingModel_id, stopPerc_id, sampleNum, simulationFlag_id, trainPlotFlag_id, animationFlag_id, tqdmHide_id) for sampleNum in range(0, len(trainingSamples))]

        for result in tqdm(parIterator(idens), total=len(idens), desc='Training Samples', position=1, leave=False, ascii=True):
            areaUnderCurve += np.trapz(result.TDList, result.percMeasuredList)

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





