#==================================================================
#TRAINING SLADS SPECIFIC  
#==================================================================

#Sample information object for training the regression model
class TrainingSample:
    def __init__(self, name, avgGroundTruthImage, maskObject, massRanges, measurementPerc, maskNum, cValue, reconImage, measuredImage, RDImage, RDValues, polyFeatures):
        self.name = name
        self.avgGroundTruthImage = avgGroundTruthImage
        self.maskObject = maskObject
        self.massRanges = massRanges
        self.maskNum = maskNum
        self.cValue = cValue
        self.measurementPerc = measurementPerc
        self.measuredImage = measuredImage
        self.reconImage = reconImage
        self.RDImage = RDImage
        self.RDValues = RDValues
        self.polyFeatures = polyFeatures

#Extract model performance/statistics from the validation data, must be run prior to epochDisplay
def epochCalculate(model, inputValidationTensors, outputValidationTensors, validationShapes, totalValidationPSNR):
    
    #If there are no validation tensors, then return empty data
    if len(inputValidationTensors) == 0: return [], 0, []
    
    psnrList = []
    ERDImages = []
    
    #Determine statistics for all validation samples
    for index in range(0, len(inputValidationTensors)):
        
        #Make predictions and retreive other data
        pred_ERD = model.predict(inputValidationTensors[index], steps=1)[0,:,:,0]
        ERDImages.append(pred_ERD)
        RD = outputValidationTensors[index][0,:,:,0].numpy()
        
        #Revert to original shapes
        pred_ERD = resize(pred_ERD, (validationShapes[index]), order=0)
        RD = resize(RD, (validationShapes[index]), order=0)
        
        #Determine statistics
        psnrList.append(compare_psnr(RD, pred_ERD, data_range=pred_ERD.max() - pred_ERD.min()))
    
    #Compute average PSNR and store for visualization and early stopping criteria
    avgPSNR = np.average(psnrList)
    totalValidationPSNR.append(avgPSNR)
    
    return psnrList, avgPSNR, ERDImages

#Visualize the network's current training progression/status
def epochDisplay(epoch, trainingDatabase, inputValidationTensors, outputValidationTensors, validationShapes, totalValidationPSNR, totalTrainingLosses, totalValidationLosses, cValue, bestCIndex, numTraining, patience, maxPatience, bestPSNR, bestEpoch, psnrList, avgPSNR, ERDImages):
    
    #If there are no validation tensors, then just save a plot of the training losses
    f = plt.figure()
    ax = f.add_subplot(111)
    if len(inputValidationTensors) == 0:
        if epoch < 1:
            ax.plot()
            ax.set_yscale('log')
            ax.set_title('Training Loss/MSE: N/A', fontsize=15, fontweight='bold')
        else:
            ax.plot(totalTrainingLosses, label='Training Loss')
            ax.legend(loc='upper right')
            ax.set_yscale('log')
            ax.set_title('Training Loss/MSE: '+str(round(totalTrainingLosses[-1],8)), fontsize=15, fontweight='bold')
        
        #Save resulting plot
        f.savefig(dir_TrainingModelResults + 'c_' + str(cValue) + '_epoch_' +str(epoch) + '.png', bbox_inches='tight')
        plt.close()
        
        return 0
    
    f = plt.figure(figsize=(35,25))
    f.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.suptitle('Epoch: '+str(epoch)+'\nPatience: '+str(patience)+'/'+str(maxPatience)+'   Best PSNR: '+str(round(bestPSNR, 4))+' at Epoch: '+str(bestEpoch), fontsize=20, fontweight='bold', y = 0.92)

    sampleLocations = [0, len(measurementPercs)-1]

    #Visualize first and last measurement percentage of the first validation sample
    for rowNum in range(1, len(sampleLocations)+1):
        
        inputValidationTensor = inputValidationTensors[sampleLocations[rowNum-1]]
        outputValidationTensor = outputValidationTensors[sampleLocations[rowNum-1]]
        validationTensorShape = validationShapes[sampleLocations[rowNum-1]]
        
        #Retrieve data
        RD = outputValidationTensor[0,:,:,0].numpy()
        measuredImage = inputValidationTensor[0,:,:,0].numpy()
        reconValueImage = inputValidationTensor[0,:,:,1].numpy()
        pred_ERD = ERDImages[sampleLocations[rowNum-1]]
        PSNR = psnrList[sampleLocations[rowNum-1]]
        
        #Revert to original shapes
        pred_ERD = resize(pred_ERD, (validationTensorShape), order=0)
        RD = resize(RD, (validationTensorShape), order=0)
        measuredImage = resize(measuredImage, (validationTensorShape), order=0)
        reconValueImage = resize(reconValueImage, (validationTensorShape), order=0)
        
        ax = plt.subplot2grid((3,4), (rowNum,0))
        ax.imshow(np.nan_to_num(measuredImage), aspect='auto', cmap='hot')
        ax.set_title('Input: Measured Values', fontsize=15, fontweight='bold')

        ax = plt.subplot2grid((3,4), (rowNum,1))
        ax.imshow(np.nan_to_num(reconValueImage), aspect='auto', cmap='hot')
        ax.set_title('Input: Recon Values', fontsize=15, fontweight='bold')
        
        ax = plt.subplot2grid((3,4), (rowNum,2))
        im = ax.imshow(RD, aspect='auto', vmin=0.0)
        ax.set_title('RD', fontsize=15, fontweight='bold')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        ax = plt.subplot2grid((3,4), (rowNum,3))
        im = ax.imshow(pred_ERD, aspect='auto')
        ax.set_title('ERD PSNR: ' + str(round(PSNR,4)), fontsize=15, fontweight='bold')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    ax = plt.subplot2grid((3,2), (0,0))
    if epoch < 1:
        ax.plot()
        ax.set_yscale('log')
        ax.set_title('Training Loss/MSE: N/A Validation Loss/MSE: N/A ', fontsize=15, fontweight='bold')
    else:
        ax.plot(totalTrainingLosses, label='Training Loss')
        ax.plot(totalValidationLosses, label='Validation Loss')
        ax.legend(loc='upper right')
        ax.set_yscale('log')
        ax.set_title('Training Loss/MSE: '+str(round(totalTrainingLosses[-1],8))+'  Validation Loss/MSE: '+str(round(totalValidationLosses[-1],8)), fontsize=15, fontweight='bold')
    
    ax = plt.subplot2grid((3,2), (0,1))
    if epoch < 1:
        ax.plot()
        ax.set_yscale('log')
        ax.set_title('Avg. Validation PSNR: N/A ', fontsize=15, fontweight='bold')
    else:
        ax.plot(totalValidationPSNR)
        ax.set_yscale('log')
        ax.set_title('Avg. Validation PSNR: ' + str(round(avgPSNR,4)), fontsize=15, fontweight='bold')
    
    #Save resulting plot
    f.savefig(dir_TrainingModelResults + 'c_' + str(cValue) + '_epoch_' +str(epoch) + '.png', bbox_inches='tight')
    plt.close()
    
    return 0

#Generate training data for a given sample and initial conditions
def generateDataset(dataset, RDValues, percValues, sample, avgGroundTruthImage, percToScan, massRanges, dir_TrainingResultsImages, dataSampleName, maskNum, cValue, imageWidth, imageHeight, trainingDataPlot):

    #Initialize percMeasured, RDImage and maskObject prior to loop, indicate that the first iteration is yet to occur
    percMeasured = 0
    RDImage, maskObject = None, None
    firstIteration = True
    
    #For each of the measurement percentages, create a training sample, evaluate it, and note the results
    for measurementPerc in tqdm(measurementPercs, desc = '%', leave=False, ascii=True):

        #Until the next measurement percetnage has been reached, continue scanning
        while (round(percMeasured) < measurementPerc):

            #If this is the first iteration, then create a new mask objectl, set initial avgImage, deactivate first iteration flag
            if firstIteration:
                maskObject = MaskObject(imageWidth, imageHeight, measurementPerc, 'pointwise')
                sample.avgImage = avgGroundTruthImage*maskObject.mask
                firstIteration = False
            else:
                maskObject, newIdxs = findNewMeasurementIdxs(maskObject, sample, RDImage, measurementPerc-percMeasured, 'random')
                sample, maskObject = performMeasurements(sample, maskObject, newIdxs, True)
            
            #Update the percentage by what was actually measured
            percMeasured = (np.sum(maskObject.mask)/maskObject.area)*100

            #Compute reconstruction
            reconImage = performRecon(sample.avgImage, maskObject)
            
            #Calculate the RD Image
            RDImage = calcRD(maskObject, reconImage, cValue, avgGroundTruthImage)

            #Extract image of only measured values
            measuredImage = avgGroundTruthImage*maskObject.mask
        
            #Append the result into the training database
            if erdModel == 'SLADS-LS' or erdModel == 'SLADS-LS':
                polyFeatures = computePolyFeatures(maskObject, reconImage)
                RDValues = RDImage[np.where(maskObject.mask==0)]
                dataset.append(TrainingSample(sample.name, avgGroundTruthImage, copy.deepcopy(maskObject), massRanges, percMeasured, maskNum, cValue, reconImage, measuredImage, RDImage, RDValues, polyFeatures))
            elif erdModel == 'DLADS':
                dataset.append(TrainingSample(sample.name, avgGroundTruthImage, copy.deepcopy(maskObject), massRanges, percMeasured, maskNum, cValue, reconImage, measuredImage, RDImage, None, None))
            
            #Visualize and save data if desired
            if trainingDataPlot:
                saveLocation = dir_TrainingResultsImages+ 'training_c_' + str(cValue) + '_var_' + str(maskNum) + '_' + sample.name + '_perc_' + str(round(percMeasured, 4))+ '.png'
                
                f = plt.figure(figsize=(20,5))
                f.subplots_adjust(top = 0.7)
                f.subplots_adjust(wspace=0.15, hspace=0.2)
                plt.suptitle('c: ' + str(cValue) + '  Variation: ' + str(maskNum) + '\nSample: ' + sample.name + '  Percent Sampled: ' + str(round(percMeasured, 4)), fontsize=20, fontweight='bold', y = 0.95)
                
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
                ax.set_title('Recon - PSNR: ' + str(round(compare_psnr(avgGroundTruthImage, reconImage, data_range=reconImage.max() - reconImage.min()), 4)), fontsize=15)

                ax = plt.subplot2grid(shape=(1,5), loc=(0,4))
                ax.imshow(RDImage, aspect='auto')
                ax.set_title('RD', fontsize=15)

                plt.savefig(saveLocation)
                plt.close()

                #Borderless saves
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(maskObject.mask, aspect='auto', cmap='gray')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(dir_TrainingResultsImages + 'c_' + str(cValue) + '_mask_'+ sample.name + '_percentage_' + str(round(percMeasured, 4)) + '_variation_' + str(maskNum) + '.png', bbox_inches=extent)
                plt.close()

                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(RDImage, aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(dir_TrainingResultsImages + 'c_' + str(cValue) + '_rd_'+ sample.name + '_percentage_' + str(round(percMeasured, 4)) + '_variation_' + str(maskNum) + '.png', bbox_inches=extent)
                plt.close()
                
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(measuredImage, aspect='auto', cmap='hot')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(dir_TrainingResultsImages + 'c_' + str(cValue) + '_measured_'+ sample.name + '_percentage_' + str(round(percMeasured, 4)) + '_variation_' + str(maskNum) + '.png', bbox_inches=extent)
                plt.close()

    return dataset

#Given a set of file names, perform the initial setup for generating Model(s)
def initTrain(sortedTrainingSampleFolders):
    
    #Create a set of training samples for each possible c Value
    trainingDatabase = [[] for cNum in range(0,len(cValues))]
    RDPSNR_trainingResults = [[] for cNum in range(0,len(cValues))]
    perc_trainingResults = [[] for cNum in range(0,len(cValues))]

    #For each physical sample, generate a training example for
    trainingSamples = []
    for trainingSampleFolder in tqdm(sortedTrainingSampleFolders, desc = 'Training Samples', ascii=True):

        #Obtain the name of the training sample
        dataSampleName = os.path.basename(trainingSampleFolder)

        #Import each of the images according to their mz range order
        images, massRanges, imageHeight, imageWidth = readScanData(trainingSampleFolder + '/')

        #Weight images equally
        mzWeights = np.ones(len(images))/len(images)

        #Create a temporary mask object
        maskObject = MaskObject(imageWidth, imageHeight, initialPercToScan, scanMethod)

        #Define a new sample
        sample = Sample(dataSampleName, images, massRanges, maskObject, mzWeights, dir_TrainingResults)

        #Append the basic information for each of the provided samples for use in determining best c Value
        trainingSamples.append(sample)

        #Determine averaged ground truth image
        avgGroundTruthImage = np.average(np.asarray(images), axis=0, weights=sample.mzWeights)
        avgGroundTruthImage = MinMaxScaler().fit_transform(avgGroundTruthImage.reshape(-1, 1)).reshape(avgGroundTruthImage.shape)

        #Save a copy of the averaged ground-truth
        saveLocation = dir_TrainingResultsImages + 'groundTruth_' + sample.name + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(avgGroundTruthImage, cmap='hot', aspect='auto')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()

        #For the number of mask iterations specified, create and extract training databases for each of the c values
        for maskNum in tqdm(range(0,numMasks), desc = 'Masks', leave=False, ascii=True):
            for cNum in tqdm(range(0, len(cValues)), desc = 'C Values', leave=False, ascii=True):
            
                #Lists that must be generated for training
                dataset = []
                RDValues = []
                percValues = []

                #Generate dataset
                dataset = generateDataset(dataset, RDValues, percValues, sample, avgGroundTruthImage, percToScan, massRanges, dir_TrainingResultsImages, dataSampleName, maskNum, cValues[cNum], imageWidth, imageHeight, trainingDataPlot)
                
                #Combine the results into the database and evaluation lists
                trainingDatabase[cNum] = trainingDatabase[cNum]+dataset

    #Save the database and samples
    pickle.dump(trainingSamples, open(dir_TrainingResults + 'trainingSamples.p', 'wb'))
    pickle.dump(trainingDatabase, open(dir_TrainingResults + 'trainingDatabase.p', 'wb'))
    
    return trainingSamples, trainingDatabase

def findBestC(trainingSamples, trainingDatabase):
    
    #If there are more than one c value, determine which minimizes the total distortion in scanned samples, force pointwise scanning (large variation with current reconstruction)
    if len(cValues)>1:
        areaUnderCurveList = []
        for cNum in tqdm(range(0, len(cValues)), desc = 'Best C', position=0, leave=True, ascii=True):
            
            PSNRLists = []
            percLists = []
            for sampleNum in tqdm(range(0, len(trainingSamples)), desc = 'Samples', leave=False, ascii=True):
                result = runSLADS(trainingSamples, None, 'pointwise', cValues[cNum], percToScan, stopPerc, sampleNum, simulationFlag=True, trainPlotFlag=True, animationFlag=False, tqdmHide=False, oracleFlag=True, bestCFlag=True)
                PSNRList, percMeasuredList = result.complete(None)
                PSNRLists.append(PSNRList)
                percLists.append(percMeasuredList)
        
            #Extract percentage results at the specified precision
            percents, trainingPSNR_mean = percResults(PSNRLists, percLists, precision)

            #Compute and save area under the PSNR curve
            areaUnderCurve = np.trapz(trainingPSNR_mean, percents)
            areaUnderCurveList.append(areaUnderCurve)
        
            #Save data and visualize/save the averaged curve for the given c value
            np.savetxt(dir_TrainingResultsImages+'trainingAveragePSNR_Percentage_c_' + str(cValues[cNum]) + '.csv', np.transpose([percents, trainingPSNR_mean]), delimiter=',')
            font = {'size' : 18}
            plt.rc('font', **font)
            f = plt.figure(figsize=(20,8))
            ax1 = f.add_subplot(1,1,1)
            ax1.plot(percents, trainingPSNR_mean, color='black')
            ax1.set_xlabel('% Pixels Measured')
            ax1.set_ylabel('Average PSNR (dB)')
            ax1.set_title('Area Under Curve: ' + str(areaUnderCurve), fontsize=15, fontweight='bold')
            plt.savefig(dir_TrainingResultsImages + 'trainingAveragePSNR_Percentage_c_' + str(cValues[cNum]) + '.png')
            plt.close()

        #Select the c value and corresponding model that maximizes the PSNR across the samples
        bestCIndex = np.argmax(areaUnderCurveList)
        bestC = cValues[bestCIndex]
    else:
        bestCIndex = 0
        bestC = cValues[bestCIndex]

    #Save the best c value
    np.save(dir_TrainingResults + 'bestC', bestC)
    np.save(dir_TrainingResults + 'bestCIndex', bestCIndex)
    np.save(dir_TrainingResults + 'cValues', cValues)
    print('bestC: ' + str(bestC))

    return bestC, cValues, bestCIndex

#Given a training database, create Model(s), noting there exists a single training sample for numCValues*numMeasurementPercs*numTrainingSamples
def trainModel(trainingDatabase, cValues, bestCIndex):
    
    if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net':
        firstFlag = True
        for trainingSample in tqdm(trainingDatabase[bestCIndex], desc = 'Setup', leave=True, ascii=True):

            #Stack polyFeatures for the regression
            if firstFlag:
                bigPolyFeatures = trainingSample.polyFeatures
                bigRD = trainingSample.RDValues
                firstFlag = False
            else:
                bigPolyFeatures = np.row_stack((bigPolyFeatures, trainingSample.polyFeatures))
                bigRD = np.append(bigRD, trainingSample.RDValues)

        #Create regression model based on user selection
        if erdModel == 'SLADS-LS':
            model = linear_model.LinearRegression()
        elif erdModel == 'SLADS-Net':
            model = nnr(activation='identity', solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 5), random_state=1, max_iter=500)
        model.fit(bigPolyFeatures, bigRD)

        #Save the model
        np.save(dir_TrainingResults+'model_cValue_'+str(cValues[bestCIndex]), model)

        return model
            
    elif erdModel == 'DLADS':

        #Create arrays to hold training data, network inputs, and metadata
        inputTensors = []
        outputTensors = []
        imagesShapes = []
        
        for trainingSample in tqdm(trainingDatabase[bestCIndex], desc = 'Setup', leave=True, ascii=True):
            
            #Extract data of interest from the database
            reconImage = trainingSample.reconImage
            maskObject = trainingSample.maskObject
            
            #Compute PSNR between reconstruction and the ground truth
            reconPSNR = compare_psnr(trainingSample.avgGroundTruthImage, reconImage, data_range=reconImage.max() - reconImage.min())
            
            #Use nan values in measured image
            measuredImage = np.empty((maskObject.mask.shape))
            measuredImage[np.where(maskObject.mask)] = trainingSample.avgGroundTruthImage[np.where(maskObject.mask)]
            measuredImage[np.where(maskObject.mask==0)] = np.nan

            #Compute features of the reconstruction
            featureImage = featureExtractor(maskObject, measuredImage, reconImage)
            
            #Extract/create the input/output data
            inputImage = featureImage
            outputImage = trainingSample.RDImage
            
            #Convert to symmetric tensors
            inputTensor, originalShape = makeTensor(inputImage, False)
            outputTensor, originalShape = makeTensor(outputImage, True)
            
            inputTensors.append(inputTensor)
            outputTensors.append(outputTensor)
            imagesShapes.append(originalShape)
        
        #Determine the number of samples to be used for training, rest are used for validation
        numTraining = (int(len(trainingSamples)*trainingSplit)*(len(measurementPercs)))

        #If there is no validation set, then use the training data for visualization/validation/early-cutoff values
        if numTraining == len(trainingDatabase[bestCIndex]):
            inputTrainingTensors = inputTensors
            outputTrainingTensors = outputTensors
            trainingShapes = imagesShapes
            inputValidationTensors = []
            outputValidationTensors = []
            validationShapes = []
        else:
            inputTrainingTensors = inputTensors[:numTraining]
            outputTrainingTensors = outputTensors[:numTraining]
            trainingShapes = imagesShapes[:numTraining]
            inputValidationTensors = inputTensors[numTraining:]
            outputValidationTensors = outputTensors[numTraining:]
            validationShapes = imagesShapes[numTraining:]

        #Go through the training tensors and split according to shape
        trainingShapes = [inputTrainingTensor.shape.as_list() for inputTrainingTensor in inputTrainingTensors]
        uniqueShapes = np.unique(trainingShapes, axis=0)
        inputShapeBuckets = [[inputTrainingTensors[index] for index in np.where(np.sum(uniqueShape==trainingShapes, axis=1)==uniqueShapes.shape[1])[0]] for uniqueShape in uniqueShapes]
        outputShapeBuckets = [[outputTrainingTensors[index] for index in np.where(np.sum(uniqueShape==trainingShapes, axis=1)==uniqueShapes.shape[1])[0]] for uniqueShape in uniqueShapes]

        #Form batches containing only single shapes
        globalInputBatches, globalOutputBatches = [], []
        for shapeBucketNum in range(0, len(inputShapeBuckets)):
            inputBatches = [inputShapeBuckets[shapeBucketNum][i:i + batchSize] for i in range(0, len(inputShapeBuckets[shapeBucketNum]), batchSize)]
            outputBatches = [outputShapeBuckets[shapeBucketNum][i:i + batchSize] for i in range(0, len(outputShapeBuckets[shapeBucketNum]), batchSize)]
            for batch in inputBatches: globalInputBatches.append(batch)
            for batch in outputBatches: globalOutputBatches.append(batch)

        #Initialize model optimizer and loss functions
        trainOptimizer = tf.keras.optimizers.Nadam(learning_rate=learningRate)

        #Define and compile the network appropriate for the number of features
        try:
            if modelDef == 'cnn': model = cnn(numStartFilters, inputTrainingTensors[0].shape[3])
            if modelDef == 'unet': model = unet(numStartFilters, inputTrainingTensors[0].shape[3])
            if modelDef == 'rbdnWithNormalization': model = rbdnWithNormalization(numStartFilters, inputTrainingTensors[0].shape[3])
        except:
            if modelDef == 'cnn': model = cnn(numStartFilters, 1)
            if modelDef == 'unet': model = unet(numStartFilters, 1)
            if modelDef == 'rbdnWithNormalization': model = rbdnWithNormalization(numStartFilters, 1)

        #Use MSE for model regression loss
        model.compile(optimizer=trainOptimizer, loss='mean_squared_error', metrics=['mse'])

        #Hold training progression results
        totalTrainingLosses = []
        totalValidationLosses = []
        totalTrainingMSEs = []
        totalValidationMSEs = []
        totalValidationPSNR = []

        #Calculate results for the validation data
        psnrList, currentValidationPSNR, ERDImages = epochCalculate(model, inputValidationTensors, outputValidationTensors, validationShapes, totalValidationPSNR)
        
        #Set initial values for visuals
        patience, bestPSNR, bestEpoch = 0, 0, 0
        
        #Perform an initial visualization if enabled
        if trainingProgressionVisuals: epochDisplay(0, trainingDatabase, inputValidationTensors, outputValidationTensors, validationShapes, totalValidationPSNR, totalTrainingLosses, totalValidationLosses, cValues[bestCIndex], bestCIndex, numTraining, patience, maxPatience, bestPSNR, bestEpoch, psnrList, currentValidationPSNR, ERDImages)

        #Train the network
        for epoch in tqdm(range(1,numEpochs+1), desc='Epoch', leave=True, ascii=True):
            epochTrainingLosses = []
            epochTrainingMSEs = []
            epochValidationLosses = []
            epochValidationMSEs = []
            
            #Training loop
            for index in tqdm(range(0,len(globalInputBatches)), desc='Train Samples', leave=False, ascii=True):
                history = model.train_on_batch(globalInputBatches[index], globalOutputBatches[index])
                epochTrainingLosses.append(history[0])
                epochTrainingMSEs.append(history[1])
            totalTrainingLosses.append(np.mean(epochTrainingLosses))
            totalTrainingMSEs.append(np.mean(epochTrainingMSEs))

            #If there are validation samples
            if len(inputValidationTensors) > 0:
                #Validation loop - check evaluation output dictionary with "print(model.metrics_names)"
                for index in tqdm(range(0,len(inputValidationTensors)), desc='Val. Samples', leave=False, ascii=True):
                    results = model.evaluate(inputValidationTensors[index], outputValidationTensors[index], verbose=0)
                    epochValidationLosses.append(results[0]) #loss
                    epochValidationMSEs.append(results[1]) #mse
                totalValidationLosses.append(np.mean(epochValidationLosses))
                totalValidationMSEs.append(np.mean(epochValidationMSEs))

            #Calculate results for the validation data
            psnrList, currentValidationPSNR, ERDImages = epochCalculate(model, inputValidationTensors, outputValidationTensors, validationShapes, totalValidationPSNR)
            
            #If the number of minimum epochs has been reached, but the current result is worse than the best, increase the counter for early cutoff
            if earlyCutoff and (epoch >= minimumEpochs) and (currentValidationPSNR <= bestPSNR):
                patience += 1

            #If there is a new best result and the minimum number of epochs has been reached, save the new model, reset the patience counter, and produce a display if not already done
            if ((epoch > minimumEpochs) and (currentValidationPSNR > bestPSNR)) or (epoch == minimumEpochs):
                model.save(dir_TrainingResults+'model_cValue_'+str(cValues[bestCIndex]))
                patience = 0
                bestPSNR = currentValidationPSNR
                bestEpoch = epoch

                #If displaying training progression, then have it do so even if its not due for this iteration
                if trainingProgressionVisuals:
                    epochDisplay(epoch, trainingDatabase, inputValidationTensors, outputValidationTensors, validationShapes, totalValidationPSNR, totalTrainingLosses, totalValidationLosses, cValues[bestCIndex], bestCIndex, numTraining, patience, maxPatience, bestPSNR, bestEpoch, psnrList, currentValidationPSNR, ERDImages)
            
            #Otherwise, check if a visualization should be done for this iteration
            elif trainingProgressionVisuals and (epoch % trainingVizSteps == 0):
                epochDisplay(epoch, trainingDatabase, inputValidationTensors, outputValidationTensors, validationShapes, totalValidationPSNR, totalTrainingLosses, totalValidationLosses, cValues[bestCIndex], bestCIndex, numTraining, patience, maxPatience, bestPSNR, bestEpoch, psnrList, currentValidationPSNR, ERDImages)
            
            #If the model has not improved within maxPatience (+1 to ensure final visualization) epochs then stop training
            if earlyCutoff and (patience >= maxPatience+1): return model

