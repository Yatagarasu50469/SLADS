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

#Given a set of file names, perform the initial setup for generating SLADS Model(s)
def initTrain(sortedTrainingSampleFolders):
    
    #Set function for the pool
    with contextlib.redirect_stdout(None):
        cGaussian_parFunction = ray.remote(cGaussian_parhelper)
        time.sleep(1)

    #Create a set of training samples for each possible c Value
    trainingDatabase = [[] for cNum in range(0,len(cValues))]
    RDPSNR_trainingResults = [[] for cNum in range(0,len(cValues))]
    perc_trainingResults = [[] for cNum in range(0,len(cValues))]

    #For each physical sample, generate a training example for
    trainingSamples = []
    for trainingSampleFolder in tqdm(sortedTrainingSampleFolders, desc = 'Training Samples', position=0, ascii=True):

        #Obtain the name of the training sample
        dataSampleName = os.path.basename(trainingSampleFolder)

        #Import each of the images according to their mz range order
        images, massRanges, imageHeight, imageWidth = readScanData(trainingSampleFolder + '/')

        #Create a mask object
        maskObject = MaskObject(imageWidth, imageHeight, measurementPercs, numMasks)

        #Weight images equally
        mzWeights = np.ones(len(images))/len(images)

        #Define a new sample
        sample = Sample(dataSampleName, images, massRanges, maskObject, mzWeights, dir_TrainingResults)

        #Append the basic information for each of the provided samples for use in determining best c Value
        trainingSamples.append(sample)

        #Determine averaged ground truth image
        avgGroundTruthImage = np.average(np.asarray(images), axis=0, weights=sample.mzWeights)

        #Save a copy of the averaged ground-truth
        saveLocation = dir_TrainingResultsImages + 'groundTruth_' + sample.name + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(avgGroundTruthImage, cmap='hot', aspect='auto')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()

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
                sample.measuredImages = [np.zeros([maskObject.imageHeight, maskObject.imageWidth]) for rangeNum in range(0,len(massRanges))]
                maskObject.measuredIdxs = []
                maskObject.unMeasuredIdxs = []

                #Compute the difference between the original and reconstructed images
                RDPP = computeDifference(avgGroundTruthImage, reconImage, info.imageType)

                #Convert differences to int
                RDPP.astype(int)

                #Pad with zeros as needed for splitting into multiple blocks
                RDPPWithZeros = np.pad(RDPP, [(int(np.floor(windowSize[0]/2)), ), (int(np.floor(windowSize[1]/2)), )], mode='constant')

                #Convert into a series of blocks and isolate unmeasured points in those blocks
                imgAsBlocksOnlyUnmeasured = viewW(RDPPWithZeros, (windowSize[0],windowSize[1])).reshape(-1,windowSize[0]*windowSize[1])[:,::1][np.logical_not(maskVect),:]

                #Add static parameters to shared pool memory
                imgAsBlocksOnlyUnmeasured_id = ray.put(imgAsBlocksOnlyUnmeasured)

                #Perform pool function and extract variables from the results
                idens = [cGaussian_parFunction.remote(cNum, sigmaValues[cNum], windowSize, orderForRD, imgAsBlocksOnlyUnmeasured_id) for cNum in range(0, len(cValues))]
                for result in tqdm(parIterator(idens), total=len(idens), position=3, desc='c Values', leave=False, ascii=True):
                    trainingDatabase[result[0]].append(TrainingSample(dataSampleName, images, maskObject, massRanges, measurementPerc, polyFeatures, reconImage, orderForRD, result[1]))
                    
                    RDImage = np.zeros((maskObject.imageHeight, maskObject.imageWidth))
                    RDImage[maskObject.unMeasuredIdxsList[measurementPercNum][maskNum][:,0], maskObject.unMeasuredIdxsList[measurementPercNum][maskNum][:,1]] = result[1]
                    measuredImage = avgGroundTruthImage*maskObject.mask
                    
                    #Borderless
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(maskObject.mask, aspect='auto', cmap='gray')
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(dir_TrainingResultsImages + 'c_' + str(cValues[result[0]]) + '_mask_'+ dataSampleName + '_percentage_' + str(measurementPercs[measurementPercNum]) + '_variation_' + str(maskNum) + '.png', bbox_inches=extent)
                    plt.close()

                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(RDImage, aspect='auto')
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(dir_TrainingResultsImages + 'c_' + str(cValues[result[0]]) + '_rd_'+ dataSampleName + '_percentage_' + str(measurementPercs[measurementPercNum]) + '_variation_' + str(maskNum) + '.png', bbox_inches=extent)
                    plt.close()
                    
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(measuredImage, aspect='auto', cmap='hot')
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(dir_TrainingResultsImages + 'c_' + str(cValues[result[0]]) + '_measured_'+ dataSampleName + '_percentage_' + str(measurementPercs[measurementPercNum]) + '_variation_' + str(maskNum) + '.png', bbox_inches=extent)
                    plt.close()
                    
                    #Visualize and save copy of the training sample
                    saveLocation = dir_TrainingResultsImages + 'training_ ' + 'c_' + str(cValues[result[0]]) + '_var_' + str(maskNum) + '_' + dataSampleName + '_perc_' + str(round(measurementPerc, 4))+ '.png'

                    f = plt.figure(figsize=(20,5))
                    f.subplots_adjust(top = 0.7)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle('c: ' + str(cValues[result[0]]) + '  Variation: ' + str(maskNum) + '\nSample: ' + dataSampleName + '  Percent Sampled: ' + str(round(measurementPerc, 4)), fontsize=20, fontweight='bold', y = 0.95)

                    reconPSNR = compare_psnr(avgGroundTruthImage, reconImage, data_range=reconImage.max() - reconImage.min())
                    RDPSNR_trainingResults[result[0]].append(reconPSNR)
                    perc_trainingResults[result[0]].append(measurementPerc)
                    
                    ax = plt.subplot2grid(shape=(1,5), loc=(0,0))
                    ax.imshow(maskObject.mask, cmap='gray', aspect='auto')
                    ax.set_title('Mask', fontsize=15)

                    ax = plt.subplot2grid(shape=(1,5), loc=(0,1))
                    ax.imshow(measuredImage, cmap='hot', aspect='auto')
                    ax.set_title('Measured', fontsize=15)

                    ax = plt.subplot2grid(shape=(1,5), loc=(0,2))
                    ax.imshow(avgGroundTruthImage, cmap='hot', aspect='auto')
                    ax.set_title('Ground-Truth', fontsize=15)

                    ax = plt.subplot2grid(shape=(1,5), loc=(0,3))
                    ax.imshow(reconImage, cmap='hot', aspect='auto')
                    ax.set_title('Recon - PSNR: ' + str(round(reconPSNR, 4)), fontsize=15)

                    ax = plt.subplot2grid(shape=(1,5), loc=(0,4))
                    ax.imshow(RDImage, aspect='auto')
                    ax.set_title('RD', fontsize=15)

                    plt.savefig(saveLocation)
                    plt.close()
            
    #Note that the PSNR curves here are always going to be the same regardless of c value, since the measurement masks were generated independent of the c value
    for cNum in tqdm(range(0, len(cValues)), desc = 'Pool Setup', leave=False, ascii=True):

        RDPSNR_data = np.asarray(RDPSNR_trainingResults[cNum]).reshape((len(sortedTrainingSampleFolders), len(measurementPercs)))
        RDPerc_data = np.asarray(perc_trainingResults[cNum]).reshape((len(sortedTrainingSampleFolders), len(measurementPercs)))
        
        #Extract percentage results at the specified precision
        percents, trainingPSNR_mean = percResults(RDPSNR_data, RDPerc_data, precision)
        
        np.savetxt(dir_TrainingResultsImages+'trainingAveragePSNR_Percentage.csv', np.transpose([percents, trainingPSNR_mean]), delimiter=',')
        
        font = {'size' : 18}
        plt.rc('font', **font)
        f = plt.figure(figsize=(20,8))
        ax1 = f.add_subplot(1,1,1)    
        ax1.plot(percents, trainingPSNR_mean, color='black') 
        ax1.set_xlabel('% Pixels Measured')
        ax1.set_ylabel('Average PSNR')
        plt.savefig(dir_TrainingResultsImages + 'trainingAveragePSNR_Percentage_c_' + str(cValues[cNum]) + '.png')
        plt.close()
    
    return trainingSamples, trainingDatabase

#Given a training database, create SLADS Model(s), noting there exists a single training sample for numCValues*numMeasurementPercs*numTrainingSamples
def trainModel(trainingDatabase):
    
    regMSEs = []
    regR2Scores = []

    #Find a SLADS model for each of the c values
    trainingModels = []
    for cNum in tqdm(range(0,len(cValues)), desc = 'c Values', leave = True, ascii=True): #For each of the proposed c values
        trainingDataset = trainingDatabase[cNum]
        
        #Build model input datasets
        for sampleNum in tqdm(range(0,len(trainingDataset)), desc = 'Training Data', leave = False, ascii=True):
            trainingSample = trainingDataset[sampleNum]                 

            if sampleNum == 0: #First loop
                bigPolyFeatures = trainingSample.polyFeatures
                bigRD = trainingSample.RD
            else: #Subsequent loops
                bigPolyFeatures = np.row_stack((bigPolyFeatures, trainingSample.polyFeatures))
                bigRD = np.append(bigRD, trainingSample.RD)
        
        #Create regression model based on user selection - all data has been normalized prior
        if regModel == 'LS':
            regr = linear_model.LinearRegression()
        elif regModel == 'NN':
            regr = nnr(activation='identity', solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 5), random_state=1, max_iter=500)
        else:
            sys.exit('Error! - The chosen regression model is not supported/implemented.')

        #Fit the training data
        regr.fit(bigPolyFeatures, bigRD)
        
        #Evaluation of training data fit
        regMSEs.append(np.mean([mean_squared_error(bigPolyFeatures[:,index], bigRD) for index in range(bigPolyFeatures.shape[1])]))
        regR2Scores.append(np.mean([r2_score(bigPolyFeatures[:,index], bigRD) for index in range(bigPolyFeatures.shape[1])]))
        
        #Save the trained model
        trainingModels.append(regr)

    #Save evaluation of the training fit
    np.savetxt(dir_TrainingResults + 'regMSEs.csv', regMSEs, delimiter=',')
    np.savetxt(dir_TrainingResults + 'regR2Scores.csv', regR2Scores, delimiter=',')

    #Save the end models and the matched cValue order array
    np.save(dir_TrainingResults + 'cValues', cValues)
    np.save(dir_TrainingResults + 'trainedModels', trainingModels)

    return trainingModels

def bestC_parhelper(info_id, trainingSamples_id, trainingModel_id, stopPerc_id, sampleNum, simulationFlag_id, trainPlotFlag_id, animationFlag_id, tqdmHide_id, bestCFlag_id):
    result = runSLADS(info_id, trainingSamples_id, trainingModel_id, stopPerc_id, sampleNum, simulationFlag_id, trainPlotFlag_id, animationFlag_id, tqdmHide_id, bestCFlag_id)
    return result.complete(0, [])

#Given SLADS Model(s) determine the c value that minimizes the total distortion in scanned samples
def findBestC(trainingSamples, trainingModels):
    
    #Set function for the pool
    with contextlib.redirect_stdout(None):
        parFunction = ray.remote(bestC_parhelper)
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
        
        #Perform pool function and extract variables from the results
        idens = [parFunction.remote(info_id, trainingSamples_id, trainingModel_id, stopPerc_id, sampleNum, simulationFlag_id, trainPlotFlag_id, animationFlag_id, tqdmHide_id, bestCFlag_id) for sampleNum in range(0, len(trainingSamples))]
        for areaResult in tqdm(parIterator(idens), total=len(idens), desc='Training Samples', position=1, leave=False, ascii=True): 
            areaUnderCurve += areaResult

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





