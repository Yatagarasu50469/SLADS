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
        self.valLosses = []

    def on_epoch_end(self, epoch, logs=None):
        
        if np.isnan(logs.get('loss')): 
            self.model.stop_training = True
            self.nanValue = True
        
        #Early stopping criteria
        currentLoss = logs.get('val_loss')
        #if epoch >= self.minimumEpochs: 
        #    self.valLosses.append(currentLoss)
        #    if np.sum(currentLoss == self.valLosses[:-self.maxPatience])) >= self.maxPatience: 
        #        self.model.stop_training = True
        #        self.nanValue = True
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
                    sampleBatch = makeCompatible([prepareInput(vizSample, mzNum) for mzNum in range(0, len(vizSample.mzReconImages))])
                    squareERD = np.mean(self.model(sampleBatch, training=False)[:,:,:,0].numpy(), axis=0)
                    squareRD = vizSample.squareRD

                    maxRangeValue = np.max([squareRD, squareERD])
                    ERD_PSNR = compare_psnr(squareRD, squareERD, data_range=maxRangeValue)
                    ERD_SSIM = compare_ssim(squareRD, squareERD, data_range=maxRangeValue)
                    
                    if np.isnan(ERD_PSNR) or np.isnan(ERD_SSIM): 
                        self.model.stop_training = True
                        self.nanValue = True
                    
                    ax = plt.subplot2grid((3,3), (vizSampleNum+1,0))
                    #if not sysLogNorm: im = ax.imshow(vizSample.squaremzAvgReconImage, aspect='auto', cmap='hot', vmin=0, vmax=np.max(vizSample.squaremzAvgReconImage))
                    #if sysLogNorm: im = ax.imshow(vizSample.squaremzAvgReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(self.vizSampleData.squaremzAvgReconImage)+3*np.std(self.vizSampleData.squaremzAvgReconImage), base=10, vmin=np.min(self.vizSampleData.squaremzAvgReconImage), vmax=np.max(self.vizSampleData.squaremzAvgReconImage)))
                    #ax.set_title('Average Reconstruction', fontsize=15, fontweight='bold')
                    if not sysLogNorm: im = ax.imshow(vizSample.squareTICReconImage, aspect='auto', cmap='hot', vmin=0, vmax=np.max(vizSample.squareTICReconImage))
                    if sysLogNorm: im = ax.imshow(vizSample.squareTICReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(self.vizSampleData.squareTICReconImage)+3*np.std(self.vizSampleData.squareTICReconImage), base=10, vmin=np.min(self.vizSampleData.squareTICReconImage), vmax=np.max(self.vizSampleData.squareTICReconImage)))
                    ax.set_title('TIC Reconstruction', fontsize=15, fontweight='bold')
                    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                    
                    ax = plt.subplot2grid((3,3), (vizSampleNum+1,1))
                    im = ax.imshow(squareRD, aspect='auto', vmin=0)
                    ax.set_title('RD', fontsize=15, fontweight='bold')
                    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                    
                    ax = plt.subplot2grid((3,3), (vizSampleNum+1,2))
                    im = ax.imshow(squareERD, aspect='auto', vmin=0)
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
    trainingValidationSampleData = np.asarray([SampleData(sampleFolder, initialPercToScan, stopPerc, 'pointwise', lineRevist, False) for sampleFolder in tqdm(sortedSampleFolders, desc='Reading', leave=True, ascii=True)], dtype='object')
    
    #Save base visualizations
    for index in range(0, len(trainingValidationSampleData)):
    
        #Determine if result data should go into training or validation sets
        if index >= int(trainingSplit*len(trainingValidationSampleData)): trainDataFlag, valDataFlag = False, True
        else: trainDataFlag, valDataFlag = True, False
        sampleData = trainingValidationSampleData[index]
        
        #Save a visual of the ground-truth mz images
        for mzNum in range(0, len(sampleData.mzImages)):
            massRange = str(sampleData.mzRanges[mzNum][0]) + '-' + str(sampleData.mzRanges[mzNum][1])
            if trainDataFlag: saveLocation = dir_TrainingResultsImages + 'groundTruth_' + sampleData.name + '_mz_' + massRange + '.png'
            if valDataFlag: saveLocation = dir_ValidationTrainingResultsImages + 'groundTruth_' + sampleData.name + '_mz_'+ massRange + '.png'
            borderlessPlot(sampleData.mzImages[mzNum], 'hot', saveLocation)
            
        #Save a visual of the TIC
        if trainDataFlag: saveLocation = dir_TrainingResultsImages + 'TIC_' + sampleData.name + '.png'
        if valDataFlag: saveLocation = dir_ValidationTrainingResultsImages + 'TIC_' + sampleData.name + '.png'
        borderlessPlot(sampleData.TIC, 'hot', saveLocation)
        
        #Save the samples database
        pickle.dump(trainingValidationSampleData, open(dir_TrainingResults + 'trainingValidationSampleData.p', 'wb'))
        
    return trainingValidationSampleData

#Given a set of samples, determine an optimal c value
def optimizeC(trainingSampleData):
    
    #If there are more than one c value, determine which maximizes the progressive PSNR in the samples; force pointwise acquisition (done during initial import!)
    if len(cValues)>1:
        t0 = time.time()
        if parallelization:
            futures = []
            for cNum in range(0, len(cValues)):
                for sampleNum in range(0, len(trainingSampleData)):
                    futures.append((trainingSampleData[sampleNum], cValues[cNum], False, 1, None, True, False, lineVisitAll, False, None, False, False, False))
            p = Pool(numberCPUS)
            results = p.starmap(runSampling, futures)
            p.close()
            p.join()
            results = np.split(np.asarray(results, dtype='object'), len(cValues))
        else:
            results = []
            for cNum in tqdm(range(0, len(cValues)), desc='c Values', leave=True, ascii=True):
                results.append([runSampling(trainingSampleData[sampleNum], cValues[cNum], False, 1, None, True, False, lineVisitAll, False, None, False, False, False) for sampleNum in tqdm(range(0, len(trainingSampleData)), desc='Samples', leave=False, ascii=True)])
        print('Completed in: '+str(time.time()-t0))
        
        areaUnderCurveList, allRDTimesList, dataPrintout = [], [], [['','Average', '', 'Standard Deviation']]
        for cNum in range(0, len(cValues)):
        
            #Double check that results were split correctly according to cValue
            if np.sum(np.diff([results[cNum][index].cValue for index in range(0, len(results[cNum]))]))>0: sys.exit('Error! - Results for c values were not split correctly.')
            
            #Compute and save area under the PSNR curve
            AUC = [np.trapz(result.cSelectionList, result.percsMeasured) for result in results[cNum]]
            areaUnderCurveList.append(np.mean(AUC))
            
            #Extract RD computation times
            allRDTimes = np.concatenate([result.computeRDTimes for result in results[cNum]])
            
            #Save information for output to file
            dataPrintout.append(['c Value', cValues[cNum]])
            dataPrintout.append(['mz PSNR Area Under Curve:', np.mean(AUC), '+/-', np.std(AUC)])
            dataPrintout.append(['Average RD Compute Time (s)', np.mean(allRDTimes), '+/-', np.std(allRDTimes)])
            dataPrintout.append([''])
            
            #Extract percentage results at the specified precision
            percents, trainingmzMetric_mean = percResults([result.cSelectionList for result in results[cNum]], [result.percsMeasured for result in results[cNum]], precision)
            
            #Visualize/save the averaged curve for the given c value
            np.savetxt(dir_TrainingResults+'optimizationCurve_c_' + str(cValues[cNum]) + '.csv', np.transpose([percents, trainingmzMetric_mean]), delimiter=',')
            font = {'size' : 18}
            plt.rc('font', **font)
            f = plt.figure(figsize=(20,8))
            ax1 = f.add_subplot(1,1,1)
            ax1.plot(percents, trainingmzMetric_mean, color='black')
            ax1.set_xlabel('% Measured')
            ax1.set_ylabel('Average PSNR of mz Reconstructions (dB)')
            ax1.set_title('Area Under Curve: ' + str(areaUnderCurveList[-1]), fontsize=15, fontweight='bold')
            plt.savefig(dir_TrainingResults + 'optimizationCurve_c_' + str(cValues[cNum]) + '.png')
            plt.close()
                
        #Save the AUC scores and RD computation times
        pd.DataFrame(dataPrintout).to_csv(dir_TrainingResults + 'cValueOptimization.csv')
        
        #Select the c value and corresponding model that maximizes the PSNR across the samples' progression
        bestCIndex = np.argmax(areaUnderCurveList)
        
    else: bestCIndex = 0
        
    #Save the final c value
    np.save(dir_TrainingResults + 'optimalC', cValues[bestCIndex])
    print('Final c Value: ' + str(cValues[bestCIndex]))
    
    return cValues[bestCIndex]

#Visualize single sample progression step
def visualizeTraining_serial(sample, result, maskNum, trainDataFlag, valDataFlag):
    
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
    #if not sysLogNorm: ax.imshow(result.sampleData.mzAvgImage, cmap='hot', aspect='auto')
    #if sysLogNorm: ax.imshow(result.sampleData.mzAvgImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.mzAvgImage)+3*np.std(result.sampleData.mzAvgImage), base=10, vmin=np.min(result.sampleData.mzAvgImage), vmax=np.max(result.sampleData.mzAvgImage)))
    if not sysLogNorm: ax.imshow(result.sampleData.TIC, cmap='hot', aspect='auto')
    if sysLogNorm: ax.imshow(result.sampleData.TIC, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.TIC)+3*np.std(result.sampleData.TIC), base=10, vmin=np.min(result.sampleData.TIC), vmax=np.max(result.sampleData.TIC)))
    ax.set_title('Ground-Truth', fontsize=15)

    ax = plt.subplot2grid(shape=(1,5), loc=(0,3))
    #if not sysLogNorm: plt.imshow(sample.mzAvgReconImage, aspect='auto', cmap='hot')
    #if sysLogNorm: plt.imshow(sample.mzAvgReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.mzAvgImage)+3*np.std(result.sampleData.mzAvgImage), base=10, vmin=np.min(result.sampleData.mzAvgImage), vmax=np.max(result.sampleData.mzAvgImage)))
    if not sysLogNorm: ax.imshow(sample.TICReconImage, cmap='hot', aspect='auto')
    if sysLogNorm: ax.imshow(sample.TICReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.TIC)+3*np.std(result.sampleData.TIC), base=10, vmin=np.min(result.sampleData.TIC), vmax=np.max(result.sampleData.TIC)))
    ax.set_title('Recon - PSNR: ' + str(round(compare_psnr(result.sampleData.TIC, sample.TICReconImage, data_range=np.max(result.sampleData.TIC)), 5)), fontsize=15)
    ax = plt.subplot2grid(shape=(1,5), loc=(0,4))
    ax.imshow(sample.RD, aspect='auto')
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
    plt.imshow(sample.RD, aspect='auto')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if trainDataFlag: plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_rd_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    if valDataFlag: plt.savefig(dir_ValidationTrainingResultsImages + 'c_' + str(optimalC) + '_rd_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    plt.close()
    
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    #if not sysLogNorm: ax.imshow(result.sampleData.mzAvgImage, cmap='hot', aspect='auto')
    #if sysLogNorm: ax.imshow(result.sampleData.mzAvgImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.mzAvgImage)+3*np.std(result.sampleData.mzAvgImage), base=10, vmin=np.min(result.sampleData.mzAvgImage), vmax=np.max(result.sampleData.mzAvgImage)))
    if not sysLogNorm: ax.imshow(result.sampleData.TIC, cmap='hot', aspect='auto')
    if sysLogNorm: ax.imshow(result.sampleData.TIC, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.TIC)+3*np.std(result.sampleData.TIC), base=10, vmin=np.min(result.sampleData.TIC), vmax=np.max(result.sampleData.TIC)))
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if trainDataFlag: plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_measured_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    if valDataFlag: plt.savefig(dir_ValidationTrainingResultsImages + 'c_' + str(optimalC) + '_measured_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    plt.close()
    
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    #if not sysLogNorm: plt.imshow(sample.mzAvgReconImage, aspect='auto', cmap='hot')
    #if sysLogNorm: plt.imshow(sample.mzAvgReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.mzAvgImage)+3*np.std(result.sampleData.mzAvgImage), base=10, vmin=np.min(result.sampleData.mzAvgImage), vmax=np.max(result.sampleData.mzAvgImage)))
    if not sysLogNorm: ax.imshow(sample.TICReconImage, cmap='hot', aspect='auto')
    if sysLogNorm: ax.imshow(sample.TICReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(result.sampleData.TIC)+3*np.std(result.sampleData.TIC), base=10, vmin=np.min(result.sampleData.TIC), vmax=np.max(result.sampleData.TIC)))
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if trainDataFlag: plt.savefig(dir_TrainingResultsImages + 'c_' + str(optimalC) + '_reconstruction_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    if valDataFlag: plt.savefig(dir_ValidationTrainingResultsImages + 'c_' + str(optimalC) + '_reconstruction_'+ result.sampleData.name + '_variation_' + str(maskNum) + '_percentage_' + str(round(sample.percMeasured, 5)) + '.png', bbox_inches=extent)
    plt.close()

#Given a set of samples and a chosen c value, generate a training database
def generateDatabases(trainingValidationSampleData, optimalC):
    
    if consistentSeed: np.random.seed(0)
    
    #For the number of mask iterations specified, create new masks and scan them with the specified method
    t0 = time.time()
    results, maskNumList, futures = [], [], []
    for index in tqdm(range(0, len(trainingValidationSampleData)), desc = 'Samples', leave=True, ascii=True, disable=parallelization):
        for maskNum in tqdm(range(0,numMasks), desc = 'Masks', leave=False, ascii=True, disable=parallelization):
            
            #Make a copy of the sample
            sample = copy.deepcopy(trainingValidationSampleData[index])
            
            #Note mask number for visualizations
            maskNumList.append(maskNum)
        
            #Create new mask
            sample.initialPercToScan = initialPercToScanTrain
            sample.stopPerc = stopPercTrain
            sample.generateInitialSets('random')
            
            #If parallel, then add job to list, otherwise just run and collect the result
            if parallelization: futures.append((sample, optimalC, False, 1, None, False, True, lineVisitAll, False, None, True, False, False))
            else: results.append(runSampling(sample, optimalC, False, 1, None, False, True, lineVisitAll, False, None, True, False, False))
    
    #If parallel, start queue and wait for results
    if parallelization: 
        p = Pool(numberCPUS)
        results = p.starmap(runSampling, futures)
        p.close()
        p.join()
    print('Completed in: '+str(time.time()-t0))
    
    #Get timing data for RD generation, average, and save
    allRDTimes = np.concatenate([result.computeRDTimes for result in results])
    dataPrintout = [['','Average', '', 'Standard Deviation']]
    dataPrintout.append(['RD Compute Time (s)', np.mean(allRDTimes), '+/-', np.std(allRDTimes)])
    pd.DataFrame(dataPrintout).to_csv(dir_TrainingResults + 'trainingValuation_RDTimes.csv')
    
    #Seperate training and validation data from results
    trainDataFlag, valDataFlag = True, False
    trainingDatabase, validationDatabase = [], []
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
                results_id = ray.put(results[index])
                _ = ray.get([visualizeTraining_parhelper.remote(results_id, maskNumList[index], trainDataFlag, valDataFlag, indexes) for indexes in np.array_split(np.arange(0, len(results[index].samples)), numberCPUS)])
                
            else:
                _ = [visualizeTraining_serial(sample, results[index], maskNumList[index], trainDataFlag, valDataFlag) for sample in tqdm(results[index].samples, desc='% Measured', leave=False, ascii=True)]

    #Save the complete databases
    pickle.dump(trainingDatabase, open(dir_TrainingResults + 'trainingDatabase.p', 'wb'))
    pickle.dump(validationDatabase, open(dir_TrainingResults + 'validationDatabase.p', 'wb'))

    return trainingDatabase, validationDatabase

#Given a training database, train a regression model
def trainModel(trainingDatabase, validationDatabase, trainingSampleData, validationSampleData, modelName):
    
    if len(trainingDatabase) == 0: sys.exit('Error! - There was no training data supplied to train a model with; perhaps only one sample was provided?')

    #If consistentcy in the random generator is desired for comparisons, then reset seed
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
        
        #Create lists of input/output images
        trainInputImages, trainOutputImages = [], []
        for sample in tqdm(trainingDatabase, desc = 'Training Data Setup', leave=True, ascii=True):
            for mzNum in range(0, len(sample.squareRDs)):
                trainInputImages.append(prepareInput(sample, mzNum))
                trainOutputImages.append(sample.squareRDs[mzNum])

        #If there is a validation set then create respective lists
        if len(validationDatabase)<=0: 
            noValFlag = True
        else:
            noValFlag = False
            valInputImages, valOutputImages = [], []
            for sample in tqdm(validationDatabase, desc = 'Validation Data Setup', leave=True, ascii=True):
                for mzNum in range(0, len(sample.squareRDs)):
                    valInputImages.append(prepareInput(sample, mzNum))
                    valOutputImages.append(sample.squareRDs[mzNum])
                
            #Extract lowest and highest density from the first validation sample for visualization during training; assumes 1% spacing
            vizSamples = [validationDatabase[0], validationDatabase[len(np.arange(initialPercToScanTrain, stopPercTrain))]]
            vizSampleData = validationSampleData[0]
        
        #Determine the number of channels in the input images
        if len(trainInputImages[0].shape) > 2: numChannels = trainInputImages[0].shape[2]
        else: numChannels = 1

        #Create generators/iterators for the training and validation sets
        trainGen = DataGen(trainInputImages, trainOutputImages, numChannels, True, batchSize)
        valGen = DataGen(valInputImages, valOutputImages, numChannels, False, batchSize)

        #Setup for distributed GPU training
        strategy = tf.distribute.MirroredStrategy()

        #While a model has not been trained, reinitialize
        trainingAttempts = 0
        while True:
            
            #Select optimizer
            if optimizer == 'Nadam': trainOptimizer = tf.keras.optimizers.Nadam(learning_rate=learningRate)
            elif optimizer == 'Adam': trainOptimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
            elif optimizer == 'RMSProp': trainOptimizer = tf.keras.optimizers.RMSprop(learning_rate=learningRate)
            elif optimizer == 'SGD': trainOptimizer = tf.keras.optimizers.SGD(learning_rate=learningRate)
            
            #Given the specified computational scope
            with strategy.scope(): 

                #Create model
                model = unet(numStartFilters, numChannels)

                #Select loss function
                if lossFunc == 'MAE': model.compile(optimizer=trainOptimizer, loss='mean_absolute_error')
                elif lossFunc == 'MSE': model.compile(optimizer=trainOptimizer, loss='mean_squared_error')

            #Setup callback object
            epochEndCallback = EpochEnd(maxPatience, minimumEpochs, trainingProgressionVisuals, trainingVizSteps, noValFlag, vizSamples, vizSampleData, dir_TrainingModelResults)

            #Perform training
            t0 = time.time()
            if not noValFlag: history = model.fit(trainGen, epochs=numEpochs, callbacks=[epochEndCallback], validation_data=valGen, validation_freq=1, verbose=1, shuffle=True)
            else: sys.exit('Error! - Model training without validation data not presently functional.')
            #history = model.fit(trainGen, epochs=numEpochs, callbacks=[epochEndCallback], verbose=1, shuffle=True)
            
            #Check if training terminated due to nan, if not then break from loop, else restart training
            if not epochEndCallback.nanValue: break
            print('Restarting model training due to nan value')
            
            #Increment the number of training attempts and check if attempt should be canceled
            trainingAttempts += 1
            if trainingAttempts >= maxTrainingAttempts: sys.exit('Error! - Maximum number of training attempts have been performed.')
        
        print('Model Training Time: ' + str(datetime.timedelta(seconds=(time.time()-t0))))
        
        #Save the final model and weights; do not include optimizer to save space
        model.save(dir_TrainingResults + modelName, include_optimizer=False)
        
        #Write out the training history to a .csv
        pd.DataFrame(history.history).to_csv(dir_TrainingResults+'history.csv')

class DataGen(tf.keras.utils.Sequence):
    
    def __init__(self, inputs, outputs, numChannels, dataAug, batchSize=1):
        self.origInputs = inputs
        self.origOutputs = outputs
        self.numChannels = numChannels
        self.dataAug = dataAug
        self.batchSize = batchSize
        self.length = len(inputs)
        self.splitIndexes = np.array_split(np.arange(0, self.length), int(self.length/batchSize))
        self.data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomCrop(64,64),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor = (-0.125, 0.125), fill_mode='constant'),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.25, 0.25), width_factor=(-0.25, 0.25), fill_mode = "constant"),
        ])
        if self.dataAug: 
            self.comImages = [tf.expand_dims(np.dstack((self.origInputs[index], self.origOutputs[index])), 0) for index in range(0, self.length)]
            self.on_epoch_end()
        else: 
            if self.batchSize == 1: 
                self.inputs = [makeCompatible(inputImage) for inputImage in self.origInputs]
                self.outputs = [makeCompatible(outputImage) for outputImage in self.origOutputs]
            else:
                self.inputs, self.outputs = [], []
                for indexes in self.splitIndexes:
                    self.inputs.append(np.asarray([self.origInputs[index] for index in indexes]))
                    self.outputs.append(np.asarray([self.origOutputs[index] for index in indexes]))

    def on_epoch_end(self):
        if self.dataAug: 

            augImages = [tf.squeeze(self.data_augmentation(self.comImages[index]), axis=0).numpy() for index in range(0, self.length)]
            augInputs = [comImage[:,:,:self.numChannels] for comImage in augImages]
            augOutputs = [comImage[:,:,self.numChannels:] for comImage in augImages]

            if self.batchSize == 1:
                self.inputs = [makeCompatible(augInput) for augInput in augInputs]
                self.outputs = [makeCompatible(augOutput) for augOutput in augOutputs]
            else: 
                self.inputs, self.outputs = [], []
                for indexes in self.splitIndexes:
                    self.inputs.append(np.asarray([augInputs[index] for index in indexes]))
                    self.outputs.append(np.asarray([augOutputs[index] for index in indexes]))

    def __len__(self):
        return math.ceil(self.length/self.batchSize)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

