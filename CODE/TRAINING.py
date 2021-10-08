#==================================================================
#TRAINING SLADS SPECIFIC  
#==================================================================

#Tensorflow callback object to check early stopping criteria and visualize the network's current training progression/status
class EpochEnd(keras.callbacks.Callback):
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

    def on_epoch_end(self, epoch, logs=None):

        if np.isnan(logs.get('loss')): 
            self.model.stop_training = True
            self.nanValue = True
        
        #Early stopping criteria
        currentLoss = logs.get('val_loss')
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

        #Store model convergence progress
        self.train_lossList.append(logs.get('loss'))
        if not self.noValFlag: self.val_lossList.append(logs.get('val_loss'))
            
        #Perform visualization as needed/specified
        if trainingProgressionVisuals and ((epoch == 0) or (epoch % trainingVizSteps == 0) or (self.stopped_epoch == epoch) or (self.bestEpoch == epoch)):

            #If there are no validation tensors, then just save a plot of the training losses
            if self.noValFlag:
                f = plt.figure(figsize=(40,10))
                f.subplots_adjust(top = 0.80)
                f.subplots_adjust(wspace=0.2, hspace=0.2)
                
                #Plot losses
                ax = plt.subplot2grid((1,2), (0,0))
                ax.plot(self.train_lossList, label='Training')
                ax.plot(self.val_lossList, label='Validation')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                ax.set_title('Training Loss: ' + str(round(self.train_lossList[-1,8])), fontsize=15, fontweight='bold')
                
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
                    
                    ERD = self.model(makeCompatible(prepareInput(vizSample)), training=False)[0,:,:,0].numpy()
                    ERD[vizSample.squareMask==1] = 0
                    
                    RD = (vizSample.squareRD-np.min(vizSample.squareRD))/(np.max(vizSample.squareRD)-np.min(vizSample.squareRD))
                    ERD = (ERD-np.min(ERD))/(np.max(ERD)-np.min(ERD))

                    ERD_PSNR = compare_psnr(RD, ERD, data_range=1)
                    ERD_SSIM = compare_ssim(RD, ERD, data_range=1)
                    
                    if np.isnan(ERD_PSNR) or np.isnan(ERD_SSIM): 
                        self.model.stop_training = True
                        self.nanValue = True
                    
                    ax = plt.subplot2grid((3,3), (vizSampleNum+1,0))
                    if not sysLogNorm: im = ax.imshow(vizSample.squaremzAvgReconImage, aspect='auto', cmap='hot', vmin=0, vmax=np.max(vizSample.squaremzAvgReconImage))
                    if sysLogNorm: im = ax.imshow(vizSample.squaremzAvgReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(self.vizSampleData.squaremzAvgImage)+3*np.std(self.vizSampleData.squaremzAvgImage), base=10, vmin=np.min(self.vizSampleData.squaremzAvgImage), vmax=np.max(self.vizSampleData.squaremzAvgImage)))
                    ax.set_title('Average Reconstruction', fontsize=15, fontweight='bold')
                    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                    
                    ax = plt.subplot2grid((3,3), (vizSampleNum+1,1))
                    im = ax.imshow(RD, aspect='auto', vmin=0, vmax=1)
                    ax.set_title('RD', fontsize=15, fontweight='bold')
                    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                    
                    ax = plt.subplot2grid((3,3), (vizSampleNum+1,2))
                    im = ax.imshow(ERD, aspect='auto', vmin=0, vmax=1)
                    ax.set_title('ERD - PSNR: ' + str(round(ERD_PSNR,4)) + ' SSIM: ' + str(round(ERD_SSIM,4)), fontsize=15, fontweight='bold')
                    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                    
            plt.suptitle('Epoch: '+str(epoch)+'     Patience: '+str(self.patience)+'/'+str(self.maxPatience)+'\nBest Loss: '+str(round(self.bestLoss, 5))+' at Epoch: '+str(self.bestEpoch), fontsize=20, fontweight='bold', y = 0.92)
            
            #Save resulting plot
            f.savefig(self.dir_TrainingModelResults + 'epoch_' +str(epoch) + '.png', bbox_inches='tight')
            plt.close()

#Read in training and validation data
def importInitialData(sortedSampleFolders):
        
    #Make sure sample mask initialization is consistent, particularly important for c value optimization
    #DO NOT RUN this section in parallel, makes optimizeC inconsistent, forcing seed in SampleData harms training generalization
    if consistentSeed: np.random.seed(0)
    trainingValidationSampleData = np.asarray([SampleData(sampleFolder, initialPercToScan, stopPerc, 'pointwise', RDMethod, True, lineRevist, True) for sampleFolder in tqdm(sortedSampleFolders, desc='Reading', leave=True, ascii=True)], dtype='object')
    
    #Save relevant images
    trainDataFlag, valDataFlag = True, False
    for index in range(0, len(trainingValidationSampleData)):
    
        #Determine if result data should go into training or validation sets
        if index >= int(trainingSplit*len(trainingValidationSampleData)): trainDataFlag, valDataFlag = False, True
        sampleData = trainingValidationSampleData[index]
        
        #Save a visual of the averaged ground-truth
        if trainDataFlag: saveLocation = dir_TrainingResultsImages + 'groundTruth_' + sampleData.name + '.png'
        if valDataFlag: saveLocation = dir_ValidationTrainingResultsImages + 'groundTruth_' + sampleData.name + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(np.mean(sampleData.mzImages, axis=0), cmap='hot', aspect='auto')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()
        
        #Save a visual of the TIC
        if trainDataFlag: saveLocation = dir_TrainingResultsImages + 'TIC_' + sampleData.name + '.png'
        if valDataFlag: saveLocation = dir_ValidationTrainingResultsImages + 'TIC_' + sampleData.name + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(sampleData.TIC, cmap='hot', aspect='auto')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()
        
        #Save a visual of the normalization image used, if it was not the TIC
        if sampleData.mzMonoValue!=-1:
            if trainDataFlag: saveLocation = dir_TrainingResultsImages + 'mzMono_' + sampleData.name + '.png'
            if valDataFlag: saveLocation = dir_ValidationTrainingResultsImages + 'mzMono_' + sampleData.name + '.png'
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
            plt.axis('off')
            plt.imshow(sampleData.mzMono, cmap='hot', aspect='auto')
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(saveLocation, bbox_inches=extent)
            plt.close()

    #Save the samples database
    pickle.dump(trainingValidationSampleData, open(dir_TrainingResults + 'trainingValidationSampleData.p', 'wb'))
    
    return trainingValidationSampleData

#Given a set of training samples, determine an optimal c value
def optimizeC(trainingSampleData):
    
    #If there are more than one c value, determine which minimizes the total distortion in the samples, force this optimization to be performed with pointwise scanning
    if len(cValues)>1:
        
        if parallelization:
            t0 = time.time()
            futures = []
            for cNum in range(0, len(cValues)):
                for sampleNum in range(0, len(trainingSampleData)):
                    futures.append(runSLADS_parhelper.remote(trainingSampleData[sampleNum], cValues[cNum], False, 1, None, True, False, lineVisitAll, False, None, False))
            results = np.split(np.asarray(ray.get(futures), dtype='object'), len(cValues))
            print('Completed in: '+str(time.time()-t0))
        else:
            results = []
            for cNum in tqdm(range(0, len(cValues)), desc='c Values', leave=True, ascii=True):
                results.append([runSLADS(trainingSampleData[sampleNum], cValues[cNum], False, 1, None, True, False, lineVisitAll, False, None, False) for sampleNum in tqdm(range(0, len(trainingSampleData)), desc='Samples', leave=False, ascii=True)])
        areaUnderCurveList = []
        
        for cNum in range(0, len(cValues)):
        
            #Double check that results were split correctly according to cValue
            if np.sum(np.diff([results[cNum][index].cValue for index in range(0, len(results[cNum]))]))>0: Tracer()()
            
            #Extract percentage results at the specified precision
            percents, trainingmzMetric_mean = percResults([result.cSelectionList for result in results[cNum]], [result.percsMeasured for result in results[cNum]], precision)

            #Compute and save area under the PSNR curve
            areaUnderCurveList.append(np.trapz(trainingmzMetric_mean, percents))
        
            #Save data and visualize/save the averaged curve for the given c value
            np.savetxt(dir_TrainingResultsImages+'optimizationCurve_c_' + str(cValues[cNum]) + '.csv', np.transpose([percents, trainingmzMetric_mean]), delimiter=',')
            font = {'size' : 18}
            plt.rc('font', **font)
            f = plt.figure(figsize=(20,8))
            ax1 = f.add_subplot(1,1,1)
            ax1.plot(percents, trainingmzMetric_mean, color='black')
            ax1.set_xlabel('% Measured')
            ax1.set_ylabel('Average PSNR of mz Reconstructions (dB)')
            #if legacyFlag: ax1.set_ylabel('Average Total Distortion of mz Reconstructions')
            ax1.set_title('Area Under Curve: ' + str(areaUnderCurveList[-1]), fontsize=15, fontweight='bold')
            plt.savefig(dir_TrainingResultsImages + 'optimizationCurve_c_' + str(cValues[cNum]) + '.png')
            plt.close()
            
        #Select the c value and corresponding model that maximizes the PSNR across the samples' progression
        bestCIndex = np.argmax(areaUnderCurveList)
    else:
        bestCIndex = 0
        
    #Save the final c value
    np.save(dir_TrainingResults + 'optimalC', cValues[bestCIndex])
    print('Final c Value: ' + str(cValues[bestCIndex]))
    
    return cValues[bestCIndex]

#Visualize multiple sample progression steps at once
@ray.remote
def visualizeTraining_parhelper(sample, result, maskNum, trainDataFlag, valDataFlag, parallel):
    return visualizeTraining_serial(sample, result, maskNum, trainDataFlag, valDataFlag, parallel)

#Visualize single sample progression step
def visualizeTraining_serial(sample, result, maskNum, trainDataFlag, valDataFlag, parallel):

    #If in a parallel thread, re-import libraries inside of thread to set plotting backend as non-interactive
    if parallel:
        import matplotlib
        matplotlib.use('agg')

    if trainDataFlag: saveLocation = dir_TrainingResultsImages+ 'training_c_' + str(optimalC) + '_variation_' + str(maskNum) + '_' + result.sampleData.name + '_percentage_' + str(round(sample.percMeasured, 5))+ '.png'
    if valDataFlag: saveLocation = dir_ValidationTrainingResultsImages+ 'validation_c_' + str(optimalC) + '_variation_' + str(maskNum) + '_' + result.sampleData.name + '_percentage_' + str(round(sample.percMeasured, 5))+ '.png'
    
    f = plt.figure(figsize=(20,5))
    f.subplots_adjust(top = 0.7)
    f.subplots_adjust(wspace=0.15, hspace=0.2)
    plt.suptitle('c: ' + str(optimalC) + '  Variation: ' + str(maskNum) + '\nSample: ' + result.sampleData.name + '  Percent Sampled: ' + str(round(sample.percMeasured, 5)), fontsize=20, fontweight='bold', y = 0.95)
    
    ax = plt.subplot2grid(shape=(1,5), loc=(0,0))
    ax.imshow(sample.mask, cmap='gray', aspect='auto')
    ax.set_title('Mask', fontsize=15)

    ax = plt.subplot2grid(shape=(1,5), loc=(0,1))
    if not sysLogNorm: ax.imshow(sample.mzAvgImage, cmap='hot', aspect='auto')
    if sysLogNorm: ax.imshow(sample.mzAvgImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.mzAvgImage)+3*np.std(result.sampleData.mzAvgImage), base=10, vmin=np.min(result.sampleData.mzAvgImage), vmax=np.max(result.sampleData.mzAvgImage)))
    ax.set_title('Measured', fontsize=15)

    ax = plt.subplot2grid(shape=(1,5), loc=(0,2))
    if not sysLogNorm: ax.imshow(result.sampleData.mzAvgImage, cmap='hot', aspect='auto')
    if sysLogNorm: ax.imshow(result.sampleData.mzAvgImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.mzAvgImage)+3*np.std(result.sampleData.mzAvgImage), base=10, vmin=np.min(result.sampleData.mzAvgImage), vmax=np.max(result.sampleData.mzAvgImage)))
    ax.set_title('Ground-Truth', fontsize=15)

    ax = plt.subplot2grid(shape=(1,5), loc=(0,3))
    if not sysLogNorm: ax.imshow(sample.mzAvgReconImage, cmap='hot', aspect='auto')
    if sysLogNorm: ax.imshow(sample.mzAvgReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.mzAvgImage)+3*np.std(result.sampleData.mzAvgImage), base=10, vmin=np.min(result.sampleData.mzAvgImage), vmax=np.max(result.sampleData.mzAvgImage)))
    ax.set_title('Recon - PSNR: ' + str(round(compare_psnr(result.sampleData.mzAvgImage, sample.mzAvgReconImage, data_range=np.max(result.sampleData.mzAvgImage)), 5)), fontsize=15)

    ax = plt.subplot2grid(shape=(1,5), loc=(0,4))
    ax.imshow((sample.RD-np.min(sample.RD))/(np.max(sample.RD)-np.min(sample.RD)), aspect='auto')
    ax.set_title('RD', fontsize=15)
    plt.savefig(saveLocation)
    plt.close()
    
    #Borderless saves
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(sample.mask, aspect='auto', cmap='gray')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if trainDataFlag: plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_mask_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    if valDataFlag: plt.savefig(dir_ValidationTrainingResultsImages + 'c_' + str(optimalC) + '_mask_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    plt.close()
    
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow((sample.RD-np.min(sample.RD))/(np.max(sample.RD)-np.min(sample.RD)), aspect='auto')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if trainDataFlag: plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_rd_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    if valDataFlag: plt.savefig(dir_ValidationTrainingResultsImages + 'c_' + str(optimalC) + '_rd_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    plt.close()
    
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    if not sysLogNorm: plt.imshow(sample.mzAvgImage, aspect='auto', cmap='hot')
    if sysLogNorm: plt.imshow(sample.mzAvgImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.mzAvgImage)+3*np.std(result.sampleData.mzAvgImage), base=10, vmin=np.min(result.sampleData.mzAvgImage), vmax=np.max(result.sampleData.mzAvgImage)))
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if trainDataFlag: plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_measured_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    if valDataFlag: plt.savefig(dir_ValidationTrainingResultsImages + 'c_' + str(optimalC) + '_measured_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    plt.close()
    
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    if not sysLogNorm: plt.imshow(sample.mzAvgReconImage, aspect='auto', cmap='hot')
    if sysLogNorm: plt.imshow(sample.mzAvgReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.mzAvgImage)+3*np.std(result.sampleData.mzAvgImage), base=10, vmin=np.min(result.sampleData.mzAvgImage), vmax=np.max(result.sampleData.mzAvgImage)))
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if trainDataFlag: plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_reconstruction_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    if valDataFlag: plt.savefig(dir_ValidationTrainingResultsImages + 'c_' + str(optimalC) + '_reconstruction_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    plt.close()

#Given a set of samples and a chosen c value, generate a training database
def generateDatabases(trainingValidationSampleData, optimalC):
    
    #Create holding location for training and validation samples
    trainingDatabase, validationDatabase = [], []
    
    if consistentSeed: np.random.seed(0)
    
    #For the number of mask iterations specified, create new masks and scan them with the specified method
    if parallelization: futures = []
    else: results = []
    maskNumList = []
    for index in tqdm(range(0, len(trainingValidationSampleData)), desc = 'Samples', leave=True, ascii=True, disable=parallelization):
        for maskNum in tqdm(range(0,numMasks), desc = 'Masks', leave=False, ascii=True, disable=parallelization):
        
            #Note mask number for visualizations
            maskNumList.append(maskNum)
        
            #Create new mask
            trainingValidationSampleData[index].initialPercToScan = initialPercToScanTrain
            trainingValidationSampleData[index].stopPerc = stopPercTrain
            trainingValidationSampleData[index].generateInitialSets('random')
            
            #If parallel, then add job to list, otherwise just run and collect the result
            if parallelization: futures.append(runSLADS_parhelper.remote(copy.deepcopy(trainingValidationSampleData[index]), optimalC, False, 1, None, False, True, lineVisitAll, False, None, False))
            else: results.append(runSLADS(copy.deepcopy(trainingValidationSampleData[index]), optimalC, False, 1, None, False, True, lineVisitAll, False, None, False))
    
    #If parallel, start queue and wait for results
    if parallelization: 
        t0 = time.time()
        results = ray.get(futures)
        print('Completed in: '+str(time.time()-t0))
    
    #Seperate training and validation data from results
    trainDataFlag, valDataFlag = True, False
    for index in tqdm(range(0, len(results)), desc='Visualization/Set Separation', leave=True, ascii=True):
        
        #Determine if result data should go into training or validation sets
        if index >= int(trainingSplit*len(trainingValidationSampleData))*numMasks: trainDataFlag, valDataFlag = False, True
        
        #Append the result into the training or validation database
        for sample in results[index].samples:
            if trainDataFlag: trainingDatabase.append(copy.deepcopy(sample))
            if valDataFlag: validationDatabase.append(copy.deepcopy(sample))
        
        #Visualize and save data if desired, in parallel if specified
        if trainingDataPlot:
            if parallelization:
                futures = [visualizeTraining_parhelper.remote(sample, results[index], maskNumList[index], trainDataFlag, valDataFlag, parallelization) for sample in results[index].samples]
                _ = ray.get(futures)
            else:
                _ = [visualizeTraining_serial(sample, results[index], maskNumList[index], trainDataFlag, valDataFlag, parallelization) for sample in tqdm(results[index].samples, desc='% Measured', leave=False, ascii=True)]
                
    #Save the complete databases
    pickle.dump(trainingDatabase, open(dir_TrainingResults + 'trainingDatabase.p', 'wb'))
    pickle.dump(validationDatabase, open(dir_TrainingResults + 'validationDatabase.p', 'wb'))

    return trainingDatabase, validationDatabase

#Given a training database, train a regression model
def trainModel(trainingDatabase, validationDatabase, trainingSampleData, validationSampleData, optimalC):
    
    #If consistentcy in the random generator is desired for comparisons, then reset seed
    if consistentSeed: 
        tf.random.set_seed(0)
        np.random.seed(0)
        random.seed(0)
    
    if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net':
        
        bigPolyFeatures = np.row_stack([sample.polyFeatures for sample in trainingDatabase])
        bigRD = np.concatenate([sample.squareRDValues for sample in trainingDatabase])
        
        #Create and save regression model based on user selection
        if erdModel == 'SLADS-LS': model = linear_model.LinearRegression()
        elif erdModel == 'SLADS-Net': model = nnr(activation='identity', solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 5), random_state=1, max_iter=500)
        model.fit(bigPolyFeatures, bigRD)
        np.save(dir_TrainingResults+'model_cValue_'+str(optimalC), model)
        
        return model
            
    elif erdModel == 'DLADS':
        
        #Create lists of input/output images
        trainInputImages, trainOutputImages = [], []
        for sample in tqdm(trainingDatabase, desc = 'Training Data Setup', leave=True, ascii=True):
            trainInputImages.append(prepareInput(sample))
            trainOutputImages.append(sample.squareRD)

        #If there is a validation set then create respective lists
        if len(validationDatabase)<=0: 
            noValFlag = True
        else:
            noValFlag = False
            valInputImages, valOutputImages = [], []
            for sample in tqdm(validationDatabase, desc = 'Validation Data Setup', leave=True, ascii=True):
                valInputImages.append(prepareInput(sample))
                valOutputImages.append(sample.squareRD)
                
            #Extract lowest and highest density from the first validation sample for visualization during training; assumes 1% spacing
            vizSamples = [validationDatabase[0], validationDatabase[len(np.arange(initialPercToScanTrain,stopPercTrain))]]
            vizSampleData = validationSampleData[0]
                    
        #Determine the number of channels in the input images
        if len(trainInputImages[0].shape) > 2: numChannels = trainInputImages[0].shape[2]
        else: numChannels = 1
        
        #Specify data augmentation steps
        data_augmentation = tf.keras.Sequential([
        experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        experimental.preprocessing.RandomHeight(factor=(-0.25, 0.25)),
        experimental.preprocessing.RandomWidth(factor=(-0.25, 0.25)),
        experimental.preprocessing.RandomRotation(factor = (-0.125, 0.125), fill_mode='constant'),
        experimental.preprocessing.RandomTranslation(height_factor=(-0.25, 0.25), width_factor=(-0.25, 0.25), fill_mode = "constant"),
        ])
        
        #If data augmentation is to be performed
        augmentedSets = []
        if numAugTimes>0:
            
            #Combine input and output images, so they can be augmented the same way; expanding dims for network compatability
            augInputSets = [tf.expand_dims(np.dstack((trainInputImages[index], trainOutputImages[index])), 0) for index in range(0, len(trainInputImages))]
            
            #Setup samples for augmentation; do so in advance of calls, otherwise changes input/output random seeds...
            for _ in range(0, numAugTimes): augmentedSets = augmentedSets+[tf.squeeze(data_augmentation(image), axis=0).numpy() for image in augInputSets]
            
            #Split augmented input and output data
            for image in augmentedSets:
                trainInputImages.append(image[:,:,:numChannels])
                trainOutputImages.append(np.squeeze(image[:,:,numChannels:], axis=2))
            
        #f = plt.figure(figsize=(22,15))
        #j = 0
        #for i in range(10):
        #    if i >= 5: i, j = i-5, 2
        #    index = np.random.randint(0, high=len(trainInputImages))
        #    ax = plt.subplot2grid(shape=(4,5), loc=(0+j,i))
        #    ax.imshow(np.mean(trainInputImages[index], axis=-1), aspect='auto', cmap='hot')
        #    ax = plt.subplot2grid(shape=(4,5), loc=(1+j,i))
        #    ax.imshow(trainOutputImages[index], aspect='auto')

        f = plt.figure(figsize=(10,10))
        for i in range(9):
            index = np.random.randint(0, high=len(trainOutputImages))
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(trainOutputImages[index], aspect='auto')
        
        plt.savefig(dir_TrainingResults+'randomOutputSet.png')
        plt.close()
         
        #Transform image sets into tensorflow datasets and batch
        trainData = tf.data.Dataset.from_generator(lambda: iter(zip(trainInputImages, trainOutputImages)), output_types=(tf.float32, tf.float32), output_shapes=([None,None,numChannels], [None,None]))
        trainData = trainData.padded_batch(batchSize)
        if not noValFlag: 
            valData = tf.data.Dataset.from_generator(lambda: iter(zip(valInputImages, valOutputImages)), output_types=(tf.float32, tf.float32), output_shapes=([None,None,numChannels], [None,None]))
            valData = valData.batch(1)
        
        #While a model has not been trained, reinitialize
        trainingAttempts = 0
        while True:
            
            #Create model
            model = unet(numStartFilters, numChannels, batchSize)
            
            #Select optimizer
            if optimizer == 'Nadam': trainOptimizer = tf.keras.optimizers.Nadam(learning_rate=learningRate)
            elif optimizer == 'Adam': trainOptimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
            elif optimizer == 'RMSProp': trainOptimizer = tf.keras.optimizers.RMSprop(learning_rate=learningRate)
            
            #Select loss function
            if lossFunc == 'MAE': model.compile(optimizer=trainOptimizer, loss=tf.keras.losses.MeanAbsoluteError())
            elif lossFunc == 'MSE': model.compile(optimizer=trainOptimizer, loss='mean_squared_error')
            
            #Setup callback object
            epochEndCallback = EpochEnd(maxPatience, minimumEpochs, trainingProgressionVisuals, trainingVizSteps, noValFlag, vizSamples, vizSampleData, dir_TrainingModelResults)
            
            #Perform training
            t0 = time.time()
            if not noValFlag: history = model.fit(trainData, epochs=numEpochs, callbacks=[epochEndCallback], validation_data=valData, validation_freq=1, verbose=1, shuffle=True)
            else: history = model.fit(trainData, epochs=numEpochs, callbacks=[epochEndCallback], verbose=1, shuffle=True)
            
            #Check if training terminated due to nan, if not then break from loop, else restart training
            if not epochEndCallback.nanValue: break
            print('Restarting model training due to nan value')
            
            #Increment the number of training attempts and check if attempt should be canceled
            trainingAttempts += 1
            if trainingAttempts >= maxTrainingAttempts: sys.exit('Error! - Maximum number of training attempts have been performed.')
        
        print('Model Training Time: ' + str(datetime.timedelta(seconds=(time.time()-t0))))
        
        #Save the final model and weights
        model.save(dir_TrainingResults+'model_cValue_'+str(optimalC))
        
        #Write out the training history to a .csv
        pd.DataFrame(history.history).to_csv(dir_TrainingResults+'history.csv')
        
        return model
