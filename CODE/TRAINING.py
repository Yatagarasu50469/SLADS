#==================================================================
#TRAINING
#==================================================================

#Read in training and validation data; do not run this section with parallelization, makes optimizeC behavior inconsistent
def importInitialData(sortedSampleFolders):
    if manualSeedValue != -1: 
        torch.manual_seed(manualSeedValue)
        np.random.seed(manualSeedValue)
        random.seed(manualSeedValue)
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
            if np.sum(np.diff([results[cNum][index].cValue for index in range(0, len(results[cNum]))]))>0: sys.exit('\nError - Results for c values were not split correctly.')
            
            #Compute and save area under the PSNR curve
            for result in tqdm(results[cNum], desc='Samples', leave=False, ascii=asciiFlag): result.complete()
            AUC = [np.trapz(result.chanAvgPSNRList, result.percsMeasured) for result in results[cNum]]
            areaUnderCurveList.append(np.mean(AUC))
            
            #Extract RD computation times
            allRDTimes = np.concatenate([result.avgTimesComputeRD for result in results[cNum]])
            
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
            plt.savefig(dir_TrainingResults + 'optimizationCurve_c_' + str(cValues[cNum]) + '.tiff')
            plt.close(f)
        
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
    if manualSeedValue != -1: 
        torch.manual_seed(manualSeedValue)
        np.random.seed(manualSeedValue)
        random.seed(manualSeedValue)
    
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
    allRDTimes = np.concatenate([result.avgTimesComputeRD for result in results])
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
    if erdModel != 'GLANDS' and len(trainingDatabase) == 0: sys.exit('\nError - No training data available.')
    elif len(trainingSampleData) == 0: sys.exit('\nError - No training data available.')

    #If consistency in the random generator is desired for comparisons, then reset seed
    if manualSeedValue != -1: 
        torch.manual_seed(manualSeedValue)
        np.random.seed(manualSeedValue)
        random.seed(manualSeedValue)
    
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
        
    elif erdModel == 'DLADS':
    
        #Create a DLADS model in training mode, load data, and perform training
        model = DLADS(True, gpus)
        model.loadData(trainingDatabase, validationDatabase)
        model.train()
        
    elif erdModel == 'GLANDS':
        
        #Create a GLANDS model in training mode, load data, and perform training
        sys.exit('\nError - GLANDS training procedure not yet defined')
    
    #Cleanup larger variables
    del model, trainingDatabase, validationDatabase