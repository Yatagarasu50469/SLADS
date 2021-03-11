#==================================================================
#TRAINING SLADS SPECIFIC  
#==================================================================

#Tensorflow callback object to check early stopping criteria and visualize the network's current training progression/status
class EpochEnd(keras.callbacks.Callback):
    def __init__(self, maxPatience, minimumEpochs, trainingProgressionVisuals, trainingVizSteps, noValFlag, vizInputValTensors, vizInputValImages, vizOutputValImages, vizValShapes, dir_TrainingModelResults):
        self.maxPatience = maxPatience
        self.patience = 0
        self.bestWeights = None
        self.bestEpoch = 0
        self.bestPSNR = -np.inf
        self.stopped_epoch = 0
        self.minimumEpochs = minimumEpochs
        self.trainingProgressionVisuals = trainingProgressionVisuals
        self.trainingVizSteps = trainingVizSteps
        self.noValFlag = noValFlag
        self.train_lossList = []
        self.train_psnrList = []
        self.vizInputValTensors = vizInputValTensors
        self.vizInputValImages = vizInputValImages
        self.vizOutputValImages = vizOutputValImages
        self.vizValShapes = vizValShapes
        self.dir_TrainingModelResults = dir_TrainingModelResults
        if not self.noValFlag:
            self.val_lossList = []
            self.val_psnrList = []

    def on_epoch_end(self, epoch, logs=None):

        #Early stopping criteria
        if self.noValFlag:
            currentPSNR = logs.get('PSNR')
        else:
            currentPSNR = logs.get('val_PSNR')
        if (currentPSNR > self.bestPSNR) and (epoch >= self.minimumEpochs):
            self.patience = 0
            self.bestPSNR = currentPSNR
            self.bestEpoch = epoch
            self.bestWeights = self.model.get_weights()
        elif (epoch >= self.minimumEpochs):
            self.patience += 1
            if self.patience >= self.maxPatience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.bestWeights)

        #Store model convergence progress
        self.train_lossList.append(logs.get('loss'))
        self.train_psnrList.append(logs.get('PSNR'))
        if not self.noValFlag:
            self.val_lossList.append(logs.get('val_loss'))
            self.val_psnrList.append(logs.get('val_PSNR'))
            
        #Perform visualization as needed/specified
        if trainingProgressionVisuals and ((epoch == 0) or ((epoch+1) % trainingVizSteps == 0) or (self.stopped_epoch == epoch) or (self.bestEpoch == epoch)):

            #If there are no validation tensors, then just save a plot of the training losses
            if self.noValFlag:
                if averageReconInput: 
                    f = plt.figure(figsize=(35,10))
                else:
                    f = plt.figure(figsize=(30,10))
                    
                f.subplots_adjust(top = 0.80)
                f.subplots_adjust(wspace=0.2, hspace=0.2)
                
                #Plot losses
                ax = plt.subplot2grid((1,2), (0,0))
                ax.plot(self.train_lossList, label='Training')
                ax.plot(self.val_lossList, label='Validation')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                ax.set_title('Training Loss/MSE: ' + str(round(self.train_lossList[-1,8])), fontsize=15, fontweight='bold')
                
                #Plot PSNR
                ax = plt.subplot2grid((1,2), (0,1))
                ax.plot(self.train_psnrList, label='Training')
                ax.plot(self.val_psnrList, label='Validation')
                ax.legend(loc='lower right', fontsize=14)
                ax.set_yscale('log')
                ax.set_title('Training PSNR: ' + str(round(self.train_psnrList[-1,8])), fontsize=15, fontweight='bold')
            else:
                if averageReconInput: 
                    f = plt.figure(figsize=(35,25))
                else:
                    f = plt.figure(figsize=(30,25))
                    
                f.subplots_adjust(top = 0.88)
                f.subplots_adjust(wspace=0.2, hspace=0.2)
                
                #Plot losses
                ax = plt.subplot2grid((3,2), (0,0))
                ax.plot(self.train_lossList, label='Training')
                ax.plot(self.val_lossList, label='Validation')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                ax.set_title('Training Loss/MSE: ' + str(round(self.train_lossList[-1],8)) + '     Validation Loss/MSE: ' + str(round(self.val_lossList[-1],8)), fontsize=15, fontweight='bold')
                
                #Plot PSNR
                ax = plt.subplot2grid((3,2), (0,1))
                ax.plot(self.train_psnrList, label='Training')
                ax.plot(self.val_psnrList, label='Validation')
                ax.legend(loc='lower right', fontsize=14)
                ax.set_yscale('log')
                ax.set_title('Training PSNR: ' + str(round(self.train_psnrList[-1],8)) + '     Validation PSNR: ' + str(round(self.val_psnrList[-1],8)), fontsize=15, fontweight='bold')
                
                #Show a validation sample result at min and max sampling percentages (variables provided through callback initialization)
                for imageNum in range(0, len(self.vizOutputValImages)):
                    outputTensor = self.model.predict(self.vizInputValTensors[imageNum], steps=1)
                    ERD = resize(outputTensor[0,:,:,0], (self.vizValShapes[imageNum]), order=0)
                    ERD_PSNR = compare_psnr(self.vizOutputValImages[imageNum], ERD, data_range=1)
                    
                    if averageReconInput:
                        ax = plt.subplot2grid((3,4), (imageNum+1,0))
                        ax.imshow(self.vizInputValImages[imageNum][:,:,0], aspect='auto', cmap='hot', vmin=0, vmax=1)
                        ax.set_title('Input: Measured Values', fontsize=15, fontweight='bold')

                        ax = plt.subplot2grid((3,4), (imageNum+1,1))
                        ax.imshow(self.vizInputValImages[imageNum][:,:,1], aspect='auto', cmap='hot', vmin=0, vmax=1)
                        ax.set_title('Input: Recon Values', fontsize=15, fontweight='bold')

                        ax = plt.subplot2grid((3,4), (imageNum+1,2))
                        im = ax.imshow(self.vizOutputValImages[imageNum], aspect='auto', vmin=0, vmax=1)
                        ax.set_title('RD', fontsize=15, fontweight='bold')
                        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

                        ax = plt.subplot2grid((3,4), (imageNum+1,3))
                        im = ax.imshow(ERD, aspect='auto', vmin=0, vmax=1)
                        ax.set_title('ERD - PSNR: ' + str(round(ERD_PSNR,4)), fontsize=15, fontweight='bold')
                        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

                    else:
                        ax = plt.subplot2grid((3,2), (imageNum+1,0))
                        im = ax.imshow(self.vizOutputValImages[imageNum], aspect='auto', vmin=0, vmax=1)
                        ax.set_title('RD', fontsize=15, fontweight='bold')
                        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

                        ax = plt.subplot2grid((3,2), (imageNum+1,1))
                        im = ax.imshow(ERD, aspect='auto', vmin=0, vmax=1)
                        ax.set_title('ERD - PSNR: ' + str(round(ERD_PSNR,4)), fontsize=15, fontweight='bold')
                        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

            plt.suptitle('Epoch: '+str(epoch)+'     Patience: '+str(self.patience)+'/'+str(self.maxPatience)+'\nBest PSNR: '+str(round(self.bestPSNR, 4))+' at Epoch: '+str(self.bestEpoch), fontsize=20, fontweight='bold', y = 0.92)
            
            #Save resulting plot
            f.savefig(self.dir_TrainingModelResults + 'epoch_' +str(epoch) + '.png', bbox_inches='tight')
            plt.close()

def importInitialData(sortedTrainingSampleFolders):

    #For each sample, generate an object containing its relevant data
    trainingSamples = []
    for trainingSampleFolder in tqdm(sortedTrainingSampleFolders, desc = 'Training Samples', ascii=True):

        #Read all available scan data into a sample object
        sample = Sample(trainingSampleFolder, ignoreMissingLines=True)
        sample.readScanData(lineRevistMethod)
        
        #Save a visual of the averaged ground-truth
        saveLocation = dir_TrainingResultsImages + 'groundTruth_' + sample.name + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(sample.avgGroundTruthImage, cmap='hot', aspect='auto')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()
        
        #Save a visual of the normalization image
        if sample.normMethod != 'none':
            saveLocation = dir_TrainingResultsImages + 'norm_' + sample.name + '.png'
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
            plt.axis('off')
            plt.imshow(sample.normArray, cmap='hot', aspect='auto')
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(saveLocation, bbox_inches=extent)
            plt.close()

        #Append the basic sample information for use in determining an optimal c value
        trainingSamples.append(sample)
        
    #Save the samples database
    pickle.dump(trainingSamples, open(dir_TrainingResults + 'trainingSamples.p', 'wb'))
    
    return trainingSamples

#Given a set of training samples, determine an optimal c value
def optimizeC(trainingSamples):

    #If there are more than one c value, determine which minimizes the total distortion in the samples, force this optimization to be performed with pointwise scanning
    if len(cValues)>1:
        areaUnderCurveList = []
        for cNum in tqdm(range(0, len(cValues)), desc = 'c Values', position=0, leave=True, ascii=True):
            
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
    else:
        bestCIndex = 0
        
    #Save the best c value
    np.save(dir_TrainingResults + 'optimalC', cValues[bestCIndex])
    print('Optimal C: ' + str(cValues[bestCIndex]))

    return cValues[bestCIndex]

#Given a set of samples and a chosen c value, generate a training database
def generateTrainingData(samples, optimalC):
    
    trainingDatabase = []
    for sample in tqdm(samples, desc = 'Samples', leave=True, ascii=True):
        
        #For the number of mask iterations specified, create and extract training databases for each of the c values
        for maskNum in tqdm(range(0,numMasks), desc = 'Masks', leave=False, ascii=True):
            
            #Create a new instance of the sample
            newSample = Sample(sample.sampleFolder, ignoreMissingLines=True)
            newIdxs = np.transpose(np.where(sample.initialMask == 1))
            
            #For each of the measurement percentages, update a training sample, evaluate it, and store the results; prepare to normalize RD image series
            for measurementPerc in tqdm(measurementPercs, desc = '%', leave=False, ascii=True):

                #Until the next measurement percentage has been reached, continue scanning
                while (round(newSample.percMeasured) < measurementPerc):

                    #If this is the first iteration, then create a new mask object and deactivate first iteration flag, otherwise pointwise scan randomly
                    if not firstIteration: newIdxs = findNewMeasurementIdxs(newSample, newSample.RDImage, measurementPerc-newSample.percMeasured, 'random')
                    else: firstIteration = False
                    
                    #Perform measurements
                    newSample.performMeasurements(newIdxs, True)

                    #Update the percentage by what was actually measured
                    newSample.percMeasured = (np.sum(newSample.mask)/newSample.area)*100
                    
                    #Compute reconstruction
                    newSample.avgReconImage = computeRecon(sample.avgSquareMeasuredImage, sample)
                    newSample.avgReconImage = resize(newSample.avgReconImage, tuple(sample.finalDim), order=0)
                    
                    #Calculate the RD Image
                    newSample.RDImage = computeRD(newSample, optimalC)
                    
                    #Append the result into the training database
                    newSample.polyFeatures = computePolyFeatures(newSample, newSample.reconImage)
                    newSample.RDValues = newSample.RDImage[np.where(newSample.maskObject.mask==0)]
                    trainingDatabase.append(copy.deepcopy(newSample))
                    
                    #Visualize and save data if desired
                    if trainingDataPlot:
                        saveLocation = dir_TrainingResultsImages+ 'training_c_' + str(optimalC) + '_var_' + str(maskNum) + '_' + newSample.name + '_perc_' + str(round(newSample.percMeasured, 4))+ '.png'
                        
                        f = plt.figure(figsize=(20,5))
                        f.subplots_adjust(top = 0.7)
                        f.subplots_adjust(wspace=0.15, hspace=0.2)
                        plt.suptitle('c: ' + str(optimalC) + '  Variation: ' + str(maskNum) + '\nSample: ' + newSample.name + '  Percent Sampled: ' + str(round(newSample.percMeasured, 4)), fontsize=20, fontweight='bold', y = 0.95)
                        
                        ax = plt.subplot2grid(shape=(1,5), loc=(0,0))
                        ax.imshow(newSample.maskObject.mask, cmap='gray', aspect='auto')
                        ax.set_title('Mask', fontsize=15)

                        ax = plt.subplot2grid(shape=(1,5), loc=(0,1))
                        ax.imshow(newSample.avgMeasuredImage, cmap='hot', aspect='auto')
                        ax.set_title('Measured', fontsize=15)

                        ax = plt.subplot2grid(shape=(1,5), loc=(0,2))
                        ax.imshow(newSample.avgGroundTruthImage, cmap='hot', aspect='auto')
                        ax.set_title('Ground-Truth', fontsize=15)

                        ax = plt.subplot2grid(shape=(1,5), loc=(0,3))
                        ax.imshow(newSample.avgReconImage, cmap='hot', aspect='auto')
                        ax.set_title('Recon - PSNR: ' + str(round(compare_psnr(newSample.avgGroundTruthImage, newSample.avgReconImage, data_range=1), 4)), fontsize=15)

                        ax = plt.subplot2grid(shape=(1,5), loc=(0,4))
                        ax.imshow(newSample.RDImage, aspect='auto')
                        ax.set_title('RD', fontsize=15)
                        plt.savefig(saveLocation)
                        plt.close()
                        
                        #Borderless saves
                        fig=plt.figure()
                        ax=fig.add_subplot(1,1,1)
                        plt.axis('off')
                        plt.imshow(newSample.maskObject.mask, aspect='auto', cmap='gray')
                        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_mask_'+ newSample.name + '_percentage_' + str(round(newSample.percMeasured, 4)) + '_variation_' + str(maskNum) + '.png', bbox_inches=extent)
                        plt.close()
                        
                        fig=plt.figure()
                        ax=fig.add_subplot(1,1,1)
                        plt.axis('off')
                        plt.imshow(newSample.RDImage, aspect='auto')
                        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_rd_'+ newSample.name + '_percentage_' + str(round(newSample.percMeasured, 4)) + '_variation_' + str(maskNum) + '.png', bbox_inches=extent)
                        plt.close()
                        
                        fig=plt.figure()
                        ax=fig.add_subplot(1,1,1)
                        plt.axis('off')
                        plt.imshow(newSample.avgMeasuredImage, aspect='auto', cmap='hot')
                        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_measured_'+ newSample.name + '_percentage_' + str(round(newSample.percMeasured, 4)) + '_variation_' + str(maskNum) + '.png', bbox_inches=extent)
                        plt.close()
                        
                        fig=plt.figure()
                        ax=fig.add_subplot(1,1,1)
                        plt.axis('off')
                        plt.imshow(newSample.avgReconImage, aspect='auto', cmap='hot')
                        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_reconstruction_'+ newSample.name + '_percentage_' + str(round(newSample.percMeasured, 4)) + '_variation_' + str(maskNum) + '.png', bbox_inches=extent)
                        plt.close()
                        
            
    #Save the complete databases
    pickle.dump(trainingDatabase, open(dir_TrainingResults + 'trainingDatabase.p', 'wb'))

    return trainingDatabase

#Given a training database, train a regression model
def trainModel(trainingDatabase, optimalC):
    
    if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net':
        firstFlag = True
        for trainingSample in tqdm(trainingDatabase, desc = 'Setup', leave=True, ascii=True):

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
        np.save(dir_TrainingResults+'model_cValue_'+str(optimalC), model)

        return model
            
    elif erdModel == 'DLADS':
    
        #Determine the number of samples to be used for training, rest are used for validation
        numTraining = int(len(trainingDatabase)/(len(measurementPercs)*numMasks)*trainingSplit)*(len(measurementPercs)*numMasks)
        
        #Create lists of input/output images
        inputImages, outputImages, imagesShapes = [], [], []
        for trainingSample in tqdm(trainingDatabase, desc = 'Setup', leave=True, ascii=True):
            
            #Compute feature image for network input
            
            if averageReconInput: 
                inputImage, originalShape = makeCompatible(featureExtractor(trainingSample.maskObject, trainingSample.avgMeasuredImage, trainingSample.reconImage))
            else:
                inputImage, originalShape = makeCompatible(np.stack(trainingSample.measuredmzImages, axis=-1))
            
            outputImage, originalShape = makeCompatible(trainingSample.RDImage)
            
            #Add to lists
            inputImages.append(inputImage)
            outputImages.append(outputImage)
            imagesShapes.append(originalShape)

        #If there is a validation set, then split it from the training set, else indicate such with a flag
        noValFlag = False
        if numTraining == len(trainingDatabase): 
            noValFlag = True
        else:
            trainInputImages = inputImages[:numTraining]
            trainOutputImages = outputImages[:numTraining]
            trainingShapes = imagesShapes[:numTraining]
            valInputImages = inputImages[numTraining:]
            valOutputImages = outputImages[numTraining:]
            valShapes = imagesShapes[numTraining:]
        
        #Determine the number of channels in the input images
        try:
            numChannels = trainInputImages[0].shape[3]
        except:
            numChannels = 1

        #Create and compile the chosen model/optimizer
        if modelDef == 'cnn': model = cnn(numStartFilters, numChannels)
        if modelDef == 'unet': model = unet(numStartFilters, numChannels)
        if modelDef == 'mlp': model = mlp(numStartFilters, numChannels)
        trainOptimizer = tf.keras.optimizers.Nadam(learning_rate=learningRate)
        model.compile(optimizer=trainOptimizer, loss='mean_squared_error', metrics=[PSNR])
        
        #Transform image sets into tensorflow datasets
        trainData = tf.data.Dataset.from_generator(lambda: iter(zip(trainInputImages, trainOutputImages)), output_types=(tf.float32, tf.float32), output_shapes=([1,None,None,numChannels], [1,None,None,1]))
        if not noValFlag: valData = tf.data.Dataset.from_generator(lambda: iter(zip(valInputImages, valOutputImages)), output_types=(tf.float32, tf.float32), output_shapes=([1,None,None,numChannels], [1,None,None,1]))
        
        #Extract validation input/output images desired for visualization
        vizInputValTensors, vizInputValImages, vizOutputValImages, vizValShapes = [], [], [], []
        if not noValFlag:
            sampleLocations = [0, numMasks+len(measurementPercs)-2]
            for sampleLocation in sampleLocations:
                vizInputValTensors.append(valInputImages[sampleLocation])
                if averageReconInput: vizInputValImages.append(resize(valInputImages[sampleLocation][0,:,:,:], (valShapes[sampleLocation]), order=0))
                vizValShapes.append(valShapes[sampleLocation])
                vizOutputValImages.append(resize(valOutputImages[sampleLocation][0,:,:,0], (valShapes[sampleLocation]), order=0))
                
        #Perform training
        t0 = time.time()
        if not noValFlag: 
            history = model.fit(trainData, epochs=numEpochs, callbacks=[EpochEnd(maxPatience, minimumEpochs, trainingProgressionVisuals, trainingVizSteps, noValFlag, vizInputValTensors, vizInputValImages, vizOutputValImages, vizValShapes, dir_TrainingModelResults)], validation_data=valData, validation_freq=1, verbose=1, shuffle=True)
        else:
            history = model.fit(trainData, epochs=numEpochs, callbacks=[EpochEnd(maxPatience, minimumEpochs, trainingProgressionVisuals, trainingVizSteps, noValFlag, vizInputValTensors, vizInputValImages, vizOutputValImages, vizValShapes, dir_TrainingModelResults)], verbose=1, shuffle=True)
        print('Total Training Time: ', time.time()-t0)

        #Save the final model and weights
        model.save(dir_TrainingResults+'model_cValue_'+str(optimalC))
        
        #Write out the training history to a .csv
        pd.DataFrame(history.history).to_csv(dir_TrainingResults+'history.csv')
        
        return model
