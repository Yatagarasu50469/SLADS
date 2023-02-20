#==================================================================
#TRAINING SPECIFIC METHOD AND CLASS DEFINITIONS
#==================================================================

#Perform identical data augmentation steps on an a set of inputs with num channels and an output with one channel
class DataAugmentation(Layer):
    def __init__(self, numChannels=None):
        super().__init__()
        self.numChannels = numChannels
        #RandomCrop(64, 64),
        self.augmentLayer = tf.keras.Sequential([
            RandomFlip('horizontal_and_vertical'),
            RandomRotation(factor = (-0.125, 0.125), fill_mode='constant', interpolation='nearest', fill_value=0.0),
            RandomTranslation(height_factor=(-0.25, 0.25), width_factor=(-0.25, 0.25), fill_mode = 'constant', interpolation='nearest', fill_value=0.0)
        ])
        
    #Convert training/validation sample(s) in ragged tensors to regular tensors and perform augmentation; MUST set training=True for functionality
    #def __call__(self, inputs, outputs): return tf.split(self.augmentLayer(tf.concat([inputs.to_tensor(), tf.expand_dims(outputs.to_tensor(), -1)], -1), training=True), [self.numChannels,1], axis=-1)
    def __call__(self, inputs, outputs): return tf.split(self.augmentLayer(tf.concat([inputs.to_tensor(), outputs.to_tensor()], -1), training=True), [self.numChannels,1], axis=-1)

#Convert training/validation sample(s) in ragged tensors to regular tensors
class RaggedPassthrough(Layer):
    def __init__(self): super().__init__()
    def call(self, inputs, outputs): return inputs.to_tensor(), outputs.to_tensor()
    
#Tensorflow callback object to check early stopping criteria and visualize the network's current training progression/status
class EpochEnd(Callback):
    def __init__(self, maxPatience, minimumEpochs, trainingProgressionVisuals, trainingVizSteps, noValFlag, vizSamples, vizSampleData, dir_TrainingModelResults):
        self.maxPatience = maxPatience
        self.patience = 0
        self.bestWeights = None
        self.bestEpoch = 0
        self.bestLoss = np.inf
        self.stopped_epoch = 0
        self.minimumEpochs = minimumEpochs
        self.trainingProgressionVisuals = trainingProgressionVisuals
        self.trainingVizSteps = trainingVizSteps
        self.noValFlag = noValFlag
        self.train_lossList = []
        self.vizSamples = vizSamples
        self.vizSampleData = vizSampleData
        self.dir_TrainingModelResults = dir_TrainingModelResults
        if not self.noValFlag: self.val_lossList = []
        self.nanValue = False
        self.valLosses = []

    def on_epoch_end(self, epoch, logs=None):
        
        if np.isnan(logs.get('loss')): 
            self.model.stop_training = True
            self.nanValue = True
        
        #Store model convergence progress
        self.train_lossList.append(logs.get('loss'))
        if not self.noValFlag: 
            currentLoss = logs.get('val_loss')
            self.val_lossList.append(logs.get('val_loss'))
        else: 
            currentLoss = logs.get('loss')
        
        #Early stopping criteria
        if (currentLoss < self.bestLoss) and (epoch >= self.minimumEpochs):
            self.patience = 0
            self.bestLoss = currentLoss
            self.bestEpoch = epoch
            self.bestWeights = copy.deepcopy(self.model.get_weights())
        elif (epoch >= self.minimumEpochs):
            self.patience += 1
            if self.patience >= self.maxPatience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.bestWeights)
                
        #Perform visualization as needed/specified
        if trainingProgressionVisuals and ((epoch == 0) or (epoch % trainingVizSteps == 0) or (self.stopped_epoch == epoch) or (self.bestEpoch == epoch)):

            #If there are no validation tensors, then just save a plot of the training losses
            if self.noValFlag:
                f = plt.figure(figsize=(20,10))
                f.subplots_adjust(top = 0.80)
                f.subplots_adjust(wspace=0.2, hspace=0.2)
                
                #Plot losses
                ax = plt.subplot2grid((1,1), (0,0))
                ax.plot(self.train_lossList, label='Training')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                ax.set_title('Training Loss: ' + str(round(self.train_lossList[-1],8)), fontsize=15, fontweight='bold')
                
            else:
                f = plt.figure(figsize=(40,25))
                f.subplots_adjust(top = 0.88)
                f.subplots_adjust(wspace=0.2, hspace=0.2)
                
                #Plot losses
                ax = plt.subplot2grid((3,1), (0,0))
                ax.plot(self.train_lossList, label='Training')
                ax.plot(self.val_lossList, label='Validation')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                ax.set_title('Training Loss: ' + str(round(self.train_lossList[-1],8)) + '     Validation Loss: ' + str(round(self.val_lossList[-1],8)), fontsize=15, fontweight='bold')
                
                #Show a validation sample result at min and max sampling percentages (variables provided through callback initialization)
                for vizSampleNum in range(0, len(self.vizSamples)):
                
                    vizSample = self.vizSamples[vizSampleNum]
                    sampleBatch = makeCompatible([prepareInput(vizSample, self.vizSampleData, chanNum) for chanNum in range(0, len(vizSample.chanReconImages))])
                    squareERD = np.mean(self.model(sampleBatch, training=False)[:,:,:,0].numpy(), axis=0)
                    squareRD = vizSample.squareRD
                    
                    maxRangeValue = np.max([squareRD, squareERD])
                    ERD_PSNR = compare_psnr(squareRD, squareERD, data_range=maxRangeValue)
                    ERD_SSIM = compare_ssim(squareRD, squareERD, data_range=maxRangeValue)
                    
                    if np.isnan(ERD_PSNR) or np.isnan(ERD_SSIM): 
                        self.model.stop_training = True
                        self.nanValue = True
                    
                    ax = plt.subplot2grid((3,2), (vizSampleNum+1,0))
                    im = ax.imshow(squareRD, aspect='auto', vmin=0)
                    ax.set_title('RD', fontsize=15, fontweight='bold')
                    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                    
                    ax = plt.subplot2grid((3,2), (vizSampleNum+1,1))
                    im = ax.imshow(squareERD, aspect='auto', vmin=0)
                    ax.set_title('ERD - PSNR: ' + str(round(ERD_PSNR,4)) + ' SSIM: ' + str(round(ERD_SSIM,4)), fontsize=15, fontweight='bold')
                    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                    
            plt.suptitle('Epoch: '+str(epoch)+'     Patience: '+str(self.patience)+'/'+str(self.maxPatience)+'\nBest Loss: '+str(round(self.bestLoss, 5))+' at Epoch: '+str(self.bestEpoch), fontsize=20, fontweight='bold', y = 0.92)
            
            #Save resulting plot
            f.savefig(self.dir_TrainingModelResults + 'epoch_' +str(epoch) + '.png', bbox_inches='tight')
            plt.close()

#Read in training and validation data; do not run this section with parallelization, makes optimizeC behavior inconsistent
def importInitialData(sortedSampleFolders):
    if consistentSeed: 
        np.random.seed(0)
        random.seed(0)
    trainingValidationSampleData = np.asarray([SampleData(sampleFolder, initialPercToScan, stopPercTrain, 'pointwise', lineRevist, True, True, True, False, False, False, False, True) for sampleFolder in tqdm(sortedSampleFolders, desc='Imports', leave=True, ascii=asciiFlag)], dtype='object')
    pickle.dump(trainingValidationSampleData, open(dir_TrainingResults + 'trainingValidationSampleData.p', 'wb'))
    return trainingValidationSampleData

#Given a set of samples, determine an optimal c value
def optimizeC(sampleDataSet):
    
    #If there are more than one c value, determine which maximizes the progressive PSNR in the samples; force pointwise acquisition (done during initial import!)
    if len(cValues)>1:
        
        #Set bestCFlag to true and datagenFlag to false for each of the sampleData that will be used
        for sampleNum in range(0, len(sampleDataSet)): 
            sampleDataSet[sampleNum].bestCFlag = True
            sampleDataSet[sampleNum].datagenFlag = False
        
        t0 = time.time()
        if parallelization:
            
            #Setup an actor to hold global sampling progress across multiple processes
            samplingProgress_Actor = SamplingProgress_Actor.remote()
            
            #Setup sampling jobs and determine total amount of work that is going to be done
            futures, maxProgress = [], 0.0
            for cNum in range(0, len(cValues)):
                for sampleNum in range(0, len(sampleDataSet)):
                    futures.append((sampleDataSet[sampleNum], cValues[cNum], False, percToScanC, percToVizC, lineVisitAll, None, True, samplingProgress_Actor, 1.0))
                    maxProgress += sampleDataSet[sampleNum].stopPerc
            maxProgress = round(maxProgress, 2)
            
            #Initialize a global progress bar and start parallel sampling operations
            pbar = tqdm(total=maxProgress, desc = '% Sampled', leave=True, ascii=asciiFlag)
            computePool = Pool(numberCPUS)
            results = computePool.starmap_async(runSampling, futures)
            computePool.close()
            
            #While some results have yet to be returned, regularly update the global progress bar, then obtain results and purge/reset ray
            pbar.n = 0
            pbar.refresh()
            while (True):
                pbar.n = np.clip(round(ray.get(samplingProgress_Actor.getCurrent.remote()),2), 0, maxProgress)
                pbar.refresh()
                if results.ready(): 
                    pbar.n = maxProgress
                    pbar.refresh()
                    pbar.close()
                    break
                time.sleep(0.1)
            computePool.join()
            results = np.split(np.asarray(results.get(), dtype='object'), len(cValues))
            del samplingProgress_Actor
            resetRay(numberCPUS)
        else:
            results = []
            for cNum in tqdm(range(0, len(cValues)), desc='c Value Sampling', leave=True, ascii=asciiFlag):
                results.append([runSampling(sampleDataSet[sampleNum], cValues[cNum], False, percToScanC, percToVizC, lineVisitAll, None, False) for sampleNum in tqdm(range(0, len(sampleDataSet)), desc='Samples', leave=False, ascii=asciiFlag)])
        
        areaUnderCurveList, allRDTimesList, dataPrintout = [], [], [['','Average', '', 'Standard Deviation']]
        for cNum in tqdm(range(0, len(cValues)), desc='Evaluation', leave=True, ascii=asciiFlag):
        
            #Double check that results were split correctly according to cValue
            if np.sum(np.diff([results[cNum][index].cValue for index in range(0, len(results[cNum]))]))>0: sys.exit('Error! - Results for c values were not split correctly.')
            
            #Compute and save area under the PSNR curve
            for result in tqdm(results[cNum], desc='Samples', leave=False, ascii=asciiFlag): result.complete()
            AUC = [np.trapz(result.chanAvgPSNRList, result.percsMeasured) for result in results[cNum]]
            areaUnderCurveList.append(np.mean(AUC))
            
            #Extract RD computation times
            allRDTimes = np.concatenate([result.computeRDTimes for result in results[cNum]])
            
            #Save information for output to file
            dataPrintout.append(['c Value', cValues[cNum]])
            dataPrintout.append(['PSNR (dB) Area Under Curve for Targeted Channels:', np.mean(AUC), '+/-', np.std(AUC)])
            dataPrintout.append(['Average RD Compute Time (s)', np.mean(allRDTimes), '+/-', np.std(allRDTimes)])
            dataPrintout.append([])
            
            #Extract percentage results at the specified precision
            percents, trainMetricAvg = percResults([result.chanAvgPSNRList for result in results[cNum]], [result.percsMeasured for result in results[cNum]], precision)
            
            #Visualize/save the averaged curve for the given c value
            np.savetxt(dir_TrainingResults+'optimizationCurve_c_' + str(cValues[cNum]) + '.csv', np.transpose([percents, trainMetricAvg]), delimiter=',')
            font = {'size' : 18}
            plt.rc('font', **font)
            f = plt.figure(figsize=(20,8))
            ax1 = f.add_subplot(1,1,1)
            #ax1.plot(results[cNum][0].percsMeasured, results[cNum][0].chanAvgPSNRList, color='black')
            ax1.plot(percents, trainMetricAvg, color='black')
            ax1.set_xlabel('% Measured')
            ax1.set_ylabel('Average Reconstruction PSNR (dB) of Targeted Channels')
            ax1.set_title('Area Under Curve: ' + str(areaUnderCurveList[-1]), fontsize=15, fontweight='bold')
            plt.savefig(dir_TrainingResults + 'optimizationCurve_c_' + str(cValues[cNum]) + '.png')
            plt.close()
                
        #Save the AUC scores and RD computation times
        pd.DataFrame(dataPrintout).to_csv(dir_TrainingResults + 'cValueOptimization.csv')
        
        #Select the c value and corresponding model that maximizes the PSNR across the samples' progression
        bestCIndex = np.argmax(areaUnderCurveList)
        
        #Reset bestCFlag and datagenFlag back to False and True for each of the sampleData used
        for sampleNum in range(0, len(sampleDataSet)): 
            sampleDataSet[sampleNum].bestCFlag = False
            sampleDataSet[sampleNum].datagenFlag = True
        
    else: bestCIndex = 0
        
    #Save the final c value
    np.save(dir_TrainingResults + 'optimalC', cValues[bestCIndex])
    print('Final c Value: ' + str(cValues[bestCIndex]))
    
    return cValues[bestCIndex]

#Given a set of samples and a chosen c value, generate a training/validation database(s)
def genTrainValDatabases(trainingValidationSampleData, optimalC):
    
    #Use a common rng seed if enabled
    if consistentSeed: 
        np.random.seed(0)
        random.seed(0)
    
    #Create training and validation data save locations for different masks
    trainSaveLocations, valSaveLocations = [], []
    for maskNum in tqdm(range(0,numMasks), desc = 'Masks', leave=False, ascii=asciiFlag, disable=parallelization):
        trainSaveLocations.append(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_samplingVariation_' + str(maskNum) + os.path.sep)
        valSaveLocations.append(dir_ValidationTrainingResultsImages + 'c_' + str(optimalC) + '_samplingVariation_' + str(maskNum) + os.path.sep)
        if os.path.exists(trainSaveLocations[-1]): shutil.rmtree(trainSaveLocations[-1])
        os.makedirs(trainSaveLocations[-1])
        if os.path.exists(valSaveLocations[-1]): shutil.rmtree(valSaveLocations[-1])
        os.makedirs(valSaveLocations[-1])
    
    #If parallelization, setup an actor to hold global sampling progress across multiple processes
    if parallelization: samplingProgress_Actor = SamplingProgress_Actor.remote()
    
    #For the number of mask iterations specified, create new masks (should not be done in parallel) and scan them with the specified method
    t0 = time.time()
    valThresh = int(math.floor(trainingSplit*len(trainingValidationSampleData)))-1
    if valThresh < 0: valThresh = 0
    results, futures, maxProgress, trainSampleBoolList, sampleDataIndexList, sampleDataIndexList, sampleDataIndex = [], [], 0.0, [], [], [], -1
    for index in tqdm(range(0, len(trainingValidationSampleData)), desc = 'Samples', leave=True, ascii=asciiFlag, disable=parallelization):
        
        #Indicate if this is a training or validation set sample; reset sampleDataIndex when threshold has been reached
        if index <= valThresh: trainSampleBool = True
        elif trainSampleBool: sampleDataIndex, trainSampleBool = -1, False
        sampleDataIndex+=1
        
        for maskNum in tqdm(range(0,numMasks), desc = 'Masks', leave=False, ascii=asciiFlag, disable=parallelization):
            
            #Make a copy of the sampleData with a new initial measurement mask
            sampleData = copy.deepcopy(trainingValidationSampleData[index])
            sampleData.generateInitialSets('random')

            #Store training/validation sampleData index to identify the corresponding reference during training procedure
            sampleDataIndexList.append(sampleDataIndex)

            #Change location and a boolean flag for results/visuals depending on if sample belongs to training or validation sets
            trainSampleBoolList.append(trainSampleBool)
            if trainSampleBool: saveLocation = trainSaveLocations[maskNum]
            else: saveLocation = valSaveLocations[maskNum]
            
            #If parallel, then add job to list, otherwise just run and collect the result
            if parallelization: 
                futures.append((sampleData, optimalC, False, 1, None, lineVisitAll, saveLocation, True, samplingProgress_Actor, 1.0))
                maxProgress+=sampleData.stopPerc
            else: 
                results.append(runSampling(sampleData, optimalC, False, 1, None, lineVisitAll, saveLocation, False))
    maxProgress = round(maxProgress, 2)
    
    #If parallel, initialize a global progress bar, start jobs, and wait for results, regularly updating progress bar
    if parallelization: 
    
        #Initialize a global progress bar and start parallel sampling operations
        pbar = tqdm(total=maxProgress, desc = '% Sampled', leave=True, ascii=asciiFlag)
        computePool = Pool(numberCPUS)
        results = computePool.starmap_async(runSampling, futures)
        computePool.close()
        
        #While some results have yet to be returned, regularly update the global progress bar, then retrieve results and purge/reset ray
        pbar.n = 0
        pbar.refresh()
        while (True):
            pbar.n = np.clip(round(ray.get(samplingProgress_Actor.getCurrent.remote()),2), 0, maxProgress)
            pbar.refresh()
            pbar.refresh()
            if results.ready(): 
                pbar.n = maxProgress
                pbar.refresh()
                pbar.close()
                break
            time.sleep(0.1)
        computePool.join()
        results = results.get()
        del samplingProgress_Actor
        resetRay(numberCPUS)
    
    #Get timing data for RD generation, average, and save
    allRDTimes = np.concatenate([result.computeRDTimes for result in results])
    dataPrintout = [['','Average', '', 'Standard Deviation']]
    dataPrintout.append(['RD Compute Time (s)', np.mean(allRDTimes), '+/-', np.std(allRDTimes)])
    pd.DataFrame(dataPrintout).to_csv(dir_TrainingResults + 'trainingValidation_RDTimes.csv')
    
    #Reference a result, call for result completion/printout, and sort into either training or validation sets
    trainingDatabase, validationDatabase = [], []
    for index in tqdm(range(0, len(results)), desc='Processing', leave=True, ascii=asciiFlag):
        result = results[index]
        result.complete()
        for sample in results[index].samples: 
            
            #TODO: To save memory and storage space remove sample variables not needed for training here!
            #del sample.iteration, sample.percMeasured, sample.squareERD, sample.squareERDS, sample.mask
            
            #Store the index for finding the sampleData corresponding to the sample in trainingSampleData/validationSampleData during training
            sample.sampleDataIndex = sampleDataIndexList[index]
            if trainSampleBoolList[index]: trainingDatabase.append(sample)
            else: validationDatabase.append(sample)
    
    #Store the complete databases to disk
    pickle.dump(trainingDatabase, open(dir_TrainingResults + 'trainingDatabase.p', 'wb'))
    pickle.dump(validationDatabase, open(dir_TrainingResults + 'validationDatabase.p', 'wb'))
    
    return trainingDatabase, validationDatabase

#Given a training database, train a regression model
def trainModel(trainingDatabase, validationDatabase, trainingSampleData, validationSampleData, modelName):

    #Verify that there is some data allocated for training
    if len(trainingDatabase) == 0: sys.exit('Error! - No training data available.')

    #If consistency in the random generator is desired for comparisons, then reset seed
    if consistentSeed: 
        tf.random.set_seed(0)
        np.random.seed(0)
        random.seed(0)
    
    if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net':
        
        #Extract polyFeatures and squareRDValues for each input channel in the sample
        polyFeatureStack, squareRDValueStack = [], []
        for sample in trainingDatabase:
            for channelNum in range(0, len(sample.polyFeatures)):
                polyFeatureStack.append(sample.polyFeatures[channelNum])
                squareRDValueStack.append(sample.squareRDValues[channelNum])
        
        #Collapse the stacks for regression
        bigPolyFeatures = np.row_stack(polyFeatureStack)
        bigRD = np.concatenate(squareRDValueStack)
        
        #Create and save regression model based on user selection
        if erdModel == 'SLADS-LS': model = linear_model.LinearRegression()
        elif erdModel == 'SLADS-Net': model = nnr(activation='identity', solver='adam', alpha=1e-4, hidden_layer_sizes=(50, 5), random_state=1, max_iter=500)
        model.fit(bigPolyFeatures, bigRD)
        np.save(dir_TrainingResults + modelName, model)
            
    elif erdModel == 'DLADS' or erdModel == 'GLANDS':
        
        #Setup a distribution strategy (batchSize already has been adjusted to accomodate strategy)
        devices=tf.config.experimental.list_physical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy()
        
        #Create training and validation datasets compatible with tensorflow models
        trainInputImages, trainOutputImages = [], []
        for sample in tqdm(trainingDatabase, desc = 'Training Data Setup', leave=True, ascii=asciiFlag):
            for chanNum in range(0, len(sample.squareRDs)):
                trainInputImages.append(tf.convert_to_tensor(prepareInput(sample, trainingSampleData[sample.sampleDataIndex], chanNum).astype(np.float32)))
                trainOutputImages.append(tf.convert_to_tensor(np.expand_dims(sample.squareRDs[chanNum], -1).astype(np.float32)))
        trainCount = len(trainInputImages)
        trainSteps = trainCount//batchSize
        trainData = tf.data.Dataset.from_tensor_slices((tf.ragged.stack(trainInputImages), tf.ragged.stack(trainOutputImages)))
        
        #If there is not a validation set then indicate such, otherwise create required lists
        if len(validationDatabase)<=0: 
            noValFlag = True
            vizSamples, vizSampleData = None, None
        else:
            noValFlag = False
            valInputImages, valOutputImages = [], []
            for sample in tqdm(validationDatabase, desc = 'Validation Data Setup', leave=True, ascii=asciiFlag):
                for chanNum in range(0, len(sample.squareRDs)):
                    valInputImages.append(tf.convert_to_tensor(prepareInput(sample, validationSampleData[sample.sampleDataIndex], chanNum).astype(np.float32)))
                    valOutputImages.append(tf.convert_to_tensor(np.expand_dims(sample.squareRDs[chanNum], -1).astype(np.float32)))
            valCount = len(valInputImages)
            valSteps = valCount//batchSize
            valData = tf.data.Dataset.from_tensor_slices((tf.ragged.stack(valInputImages), tf.ragged.stack(valOutputImages)))
            
            #Extract lowest and highest density from the first validation sample for visualization during training; assumes 1% spacing
            vizSamples = [validationDatabase[0], validationDatabase[len(np.arange(initialPercToScanTrain, stopPercTrain))]]
            vizSampleData = validationSampleData[0]
        
        #Determine the number of channels in the input images
        if len(trainInputImages[0].shape) > 2: numChannels = trainInputImages[0].shape[2]
        else: numChannels = 1
        
        #Setup shard policy to avoid AUTO shard messages
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
         
        #Set dynamic resource tuning option
        AUTOTUNE = tf.data.AUTOTUNE
         
        #Setup training dataset for model
        trainData = trainData.repeat()
        trainData = trainData.shuffle(trainCount, seed=0, reshuffle_each_iteration=False)
        trainData = trainData.batch(batchSize)
        if augTrainData: trainData = trainData.map(DataAugmentation(numChannels), num_parallel_calls=AUTOTUNE, deterministic=True)
        else: trainData = trainData.map(RaggedPassthrough(),num_parallel_calls=AUTOTUNE, deterministic=True)
        trainData = trainData.prefetch(AUTOTUNE)
        trainData = trainData.with_options(options)
        
        #Setup validation dataset for model if applicable
        if not noValFlag: 
            valData = valData.repeat()
            valData = valData.shuffle(valCount, seed=0, reshuffle_each_iteration=False)
            valData = valData.batch(batchSize)
            valData = valData.map(RaggedPassthrough(), num_parallel_calls=AUTOTUNE, deterministic=True)
            valData = valData.prefetch(AUTOTUNE)
            valData = valData.with_options(options)
        
        #Select optimizer
        if optimizer == 'Nadam': trainOptimizer = tf.keras.optimizers.Nadam(learning_rate=learningRate)
        elif optimizer == 'Adam': trainOptimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
        elif optimizer == 'RMSProp': trainOptimizer = tf.keras.optimizers.RMSprop(learning_rate=learningRate)
        elif optimizer == 'SGD': trainOptimizer = tf.keras.optimizers.SGD(learning_rate=learningRate)
        
        #Given the specified computational scope create the model and select a loss function
        with strategy.scope(): 
            if modelDef=='unet': model = unet(numStartFilters, numChannels)
            elif modelDef=='custom': model = custom(numStartFilters, numChannels)
            if lossFunc == 'MAE': model.compile(optimizer=trainOptimizer, loss='mean_absolute_error')
            elif lossFunc == 'MSE': model.compile(optimizer=trainOptimizer, loss='mean_squared_error')
        
        #Setup callback object for visualizing training convergence
        epochEndCallback = EpochEnd(maxPatience, minimumEpochs, trainingProgressionVisuals, trainingVizSteps, noValFlag, vizSamples, vizSampleData, dir_TrainingModelResults)
        tqdmCallback = tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False, overall_bar_format='Epochs |{bar}| {n_fmt}/{total_fmt} | ETA: {remaining} | {rate_fmt} | {desc}', epoch_bar_format='Steps  |{bar}| {n_fmt}/{total_fmt} | ETA: {remaining} | {desc}', metrics_separator=' | ')

        #Train the model, timing the overall operation
        t0 = time.time()
        if not noValFlag: history = model.fit(trainData, epochs=numEpochs, callbacks=[epochEndCallback, tqdmCallback], validation_data=valData, steps_per_epoch=trainSteps, validation_steps=valSteps, verbose=0)
        else: history = model.fit(trainData, epochs=numEpochs, callbacks=[epochEndCallback, tqdmCallback], steps_per_epoch=trainSteps, verbose=0)
        t1 = time.time()
        print('Model Training Time: ' + str(datetime.timedelta(seconds=(t1-t0))))
        
        #Save the final model and weights; do not include optimizer to save space
        model.save(dir_TrainingResults + modelName, include_optimizer=False)
        
        #Write out the training history to a .csv
        pd.DataFrame(history.history).to_csv(dir_TrainingResults+'history.csv')
        
        #del model, trainData, epochEndCallback, history, trainSteps, batchSize, strategy, trainOptimizer, trainDataset, trainInputImages, trainOutputImages, trainingDatabase
        #if not noValFlag: del valSteps, valData, valDataset, valInputImages, valOutputImages, validationDatabase
