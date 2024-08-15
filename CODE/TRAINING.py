#==================================================================
#TRAINING
#==================================================================

#Read in training and validation data; do not run this section with parallelization
def importInitialData(sortedSampleFolders):
    resetRandom()
    trainingValidationSampleData = np.asarray([SampleData(sampleFolder, initialPercToScan, stopPercTrain, 'pointwise', lineRevist, True, True, True, False, False, False, False, True) for sampleFolder in tqdm(sortedSampleFolders, desc='Imports', leave=True, ascii=asciiFlag)], dtype='object')
    pickle.dump(trainingValidationSampleData, open(dir_TrainingResults + 'trainingValidationSampleData.p', 'wb'))
    return trainingValidationSampleData

#Given a set of samples, determine an optimal c value
def optimizeC(sampleDataset):
    
    #If there are more than one c value, determine which minimizes the progressive NRMSE in the samples; force pointwise acquisition (done during initial import!)
    if len(cValues)>1:
        
        #Set bestCFlag to true and datagenFlag to false for each of the sampleData that will be used
        for sampleDataNum in range(0, len(sampleDataset)): 
            sampleDataset[sampleDataNum].bestCFlag = True
            sampleDataset[sampleDataNum].datagenFlag = False
        
        t0 = time.perf_counter()
        if parallelization:
            
            #Setup an actor to hold global sampling progress across multiple processes
            samplingProgress_Actor = SamplingProgress_Actor.remote()
            
            #Setup sampling jobs and determine total amount of work that is going to be done
            futures, maxProgress = [], 0.0
            for cNum in range(0, len(cValues)):
                for sampleDataNum in range(0, len(sampleDataset)):
                    futures.append((sampleDataNum, sampleDataset[sampleDataNum], cValues[cNum], False, percToScanC, percToVizC, lineVisitAll, None, True, samplingProgress_Actor, 0.01))
                    maxProgress += sampleDataset[sampleDataNum].stopPerc
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
                pbar.n = np.clip(round(copy.deepcopy(ray.get(samplingProgress_Actor.getCurrent.remote())),2), 0, maxProgress)
                pbar.refresh()
                if results.ready(): 
                    pbar.n = maxProgress
                    pbar.refresh()
                    pbar.close()
                    break
                time.sleep(0.1)
            computePool.join()
            results = np.split(np.asarray(results.get().copy(), dtype='object'), len(cValues))
            
            del samplingProgress_Actor, futures
            resetRay(numberCPUS)
        else:
            results = []
            for cNum in tqdm(range(0, len(cValues)), desc='c Value Sampling', leave=True, ascii=asciiFlag):
                results.append([runSampling(sampleDataNum, sampleDataset[sampleDataNum], cValues[cNum], False, percToScanC, percToVizC, lineVisitAll, None, False) for sampleDataNum in tqdm(range(0, len(sampleDataset)), desc='Samples', leave=False, ascii=asciiFlag)])
        
        areaUnderCurveList, allRDTimesList, dataPrintout = [], [], [['','Average', '', 'Standard Deviation']]
        for cNum in tqdm(range(0, len(cValues)), desc='Evaluation', leave=True, ascii=asciiFlag):
        
            #Double check that results were split correctly according to cValue
            if np.sum(np.diff([results[cNum][index].cValue for index in range(0, len(results[cNum]))]))>0: sys.exit('\nError - Results for c values were not split correctly.')
            
            #Compute and save area under the NRMSE curve
            for result in tqdm(results[cNum], desc='Samples', leave=False, ascii=asciiFlag): result.complete(sampleDataset[result.sampleDataNum])
            if cOptMetric == 'NRMSE': AUC = [np.trapz(result.chanAvgNRMSEList, result.percsMeasured) for result in results[cNum]]
            elif cOptMetric == 'SSIM': AUC = [np.trapz(result.chanAvgSSIMList, result.percsMeasured) for result in results[cNum]]
            elif cOptMetric == 'PSNR': AUC = [np.trapz(result.chanAvgPSNRList, result.percsMeasured) for result in results[cNum]]
            else: sys.exit('\nError - Specified cOptMetric has not been implemented, please adjust the configuration to use either PSNR or NRMSE.')
            areaUnderCurveList.append(np.mean(AUC))
            
            #Extract RD computation times
            allRDTimes = np.concatenate([result.avgTimesComputeRD for result in results[cNum]])
            
            #Save information for output to file
            dataPrintout.append(['c Value', cValues[cNum]])
            if cOptMetric == 'NRMSE': dataPrintout.append(['NRMSE Area Under Curve for Targeted Channels:', np.mean(AUC), '+/-', np.std(AUC)])
            elif cOptMetric == 'SSIM':  dataPrintout.append(['SSIM Area Under Curve for Targeted Channels:', np.mean(AUC), '+/-', np.std(AUC)])
            elif cOptMetric == 'PSNR': dataPrintout.append(['PSNR Area Under Curve for Targeted Channels:', np.mean(AUC), '+/-', np.std(AUC)])
            dataPrintout.append(['Average RD Compute Time (s)', np.mean(allRDTimes), '+/-', np.std(allRDTimes)])
            dataPrintout.append([])
            
            #Extract percentage results at the specified precision
            if cOptMetric == 'NRMSE': percents, trainMetricAvg = percResults([result.chanAvgNRMSEList for result in results[cNum]], [result.percsMeasured for result in results[cNum]], precision)
            elif cOptMetric == 'SSIM': percents, trainMetricAvg = percResults([result.chanAvgSSIMList for result in results[cNum]], [result.percsMeasured for result in results[cNum]], precision)
            elif cOptMetric == 'PSNR': percents, trainMetricAvg = percResults([result.chanAvgPSNRList for result in results[cNum]], [result.percsMeasured for result in results[cNum]], precision)
            
            #Visualize/save the averaged curve for the given c value
            np.savetxt(dir_TrainingResults+'optimizationCurve_c_' + str(cValues[cNum]) + '.csv', np.transpose([percents, trainMetricAvg]), delimiter=',')
            font = {'size' : 18}
            plt.rc('font', **font)
            f = plt.figure(figsize=(15,8))
            ax1 = f.add_subplot(1,1,1)
            ax1.plot(percents, trainMetricAvg, color='black')
            ax1.set_xlabel('% Measured')
            if cOptMetric == 'NRMSE': ax1.set_ylabel('Average Reconstruction NRMSE of Targeted Channels')
            elif cOptMetric == 'SSIM': ax1.set_ylabel('Average Reconstruction SSIM of Targeted Channels')
            elif cOptMetric == 'PSNR': ax1.set_ylabel('Average Reconstruction PSNR of Targeted Channels')
            ax1.set_title('Area Under Curve: ' + str(areaUnderCurveList[-1]), fontsize=15, fontweight='bold')
            plt.savefig(dir_TrainingResults + 'optimizationCurve_c_' + str(cValues[cNum]) + '.tiff')
            plt.close(f)
        
        #Save the AUC scores and RD computation times
        pd.DataFrame(dataPrintout).to_csv(dir_TrainingResults + 'cValueOptimization.csv')
        
        #Select the c value and corresponding model that minimizes the target metric across the samples' progression
        
        if cOptMetric == 'NRMSE': bestCIndex = np.argmin(areaUnderCurveList)
        elif cOptMetric == 'SSIM': bestCIndex = np.argmax(areaUnderCurveList)
        elif cOptMetric == 'PSNR': bestCIndex = np.argmax(areaUnderCurveList)
        
        #Reset bestCFlag and datagenFlag back to False and True for each of the sampleData used
        for sampleNum in range(0, len(sampleDataset)): 
            sampleDataset[sampleNum].bestCFlag = False
            sampleDataset[sampleNum].datagenFlag = True
        
    else: bestCIndex = 0
    
    #Save the final c value
    np.save(dir_TrainingResults + 'optimalC', cValues[bestCIndex])
    print('Final c Value: ' + str(cValues[bestCIndex]))
    
    return cValues[bestCIndex]

#Given a set of samples and a chosen c value, generate a training/validation database(s)
def genTrainValDatabases(trainingValidationSampleData, optimalC):
    
    #Reset randomization
    resetRandom()
    
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
    t0 = time.perf_counter()
    valThresh = int(math.floor(trainingSplit*len(trainingValidationSampleData)))-1
    if valThresh < 0: valThresh = 0
    results, futures, maxProgress, trainSampleBoolList, sampleDataIndexList, sampleDataIndexList, sampleDataIndex = [], [], 0.0, [], [], [], -1
    for sampleDataNum in tqdm(range(0, len(trainingValidationSampleData)), desc = 'Samples', leave=True, ascii=asciiFlag, disable=parallelization):
        
        #Indicate if this is a training or validation set sample; reset sampleDataIndex when threshold has been reached
        if sampleDataNum <= valThresh: trainSampleBool = True
        elif trainSampleBool: sampleDataIndex, trainSampleBool = -1, False
        sampleDataIndex+=1
        
        for maskNum in tqdm(range(0,numMasks), desc = 'Masks', leave=False, ascii=asciiFlag, disable=parallelization):
            
            #Make a copy of the sampleData with a new initial measurement mask
            sampleData = copy.deepcopy(trainingValidationSampleData[sampleDataNum])
            sampleData.generateInitialSets('random')

            #Store training/validation sampleData index to identify the corresponding reference during training procedure
            sampleDataIndexList.append(sampleDataIndex)

            #Change location and a boolean flag for results/visuals depending on if sample belongs to training or validation sets
            trainSampleBoolList.append(trainSampleBool)
            if trainSampleBool: saveLocation = trainSaveLocations[maskNum]
            else: saveLocation = valSaveLocations[maskNum]
            
            #If parallel, then add job to list, otherwise just run and collect the result
            if parallelization: 
                futures.append((sampleDataNum, sampleData, optimalC, False, 1, None, lineVisitAll, saveLocation, True, samplingProgress_Actor, 0.01))
                maxProgress+=sampleData.stopPerc
            else: 
                results.append(runSampling(sampleDataNum, sampleData, optimalC, False, 1, None, lineVisitAll, saveLocation, False))
    
    #If parallel, initialize a global progress bar, start jobs, and wait for results, regularly updating progress bar
    if parallelization: 
        
        #Initialize a global progress bar and start parallel sampling operations
        maxProgress = round(maxProgress, 2)
        pbar = tqdm(total=maxProgress, desc = '% Sampled', leave=True, ascii=asciiFlag)
        computePool = Pool(numberCPUS)
        results = computePool.starmap_async(runSampling, futures)
        computePool.close()
        
        #While some results have yet to be returned, regularly update the global progress bar, then retrieve results and purge/reset ray
        pbar.n = 0
        pbar.refresh()
        while (True):
            pbar.n = np.clip(round(copy.deepcopy(ray.get(samplingProgress_Actor.getCurrent.remote())),2), 0, maxProgress)
            pbar.refresh()
            if results.ready(): 
                pbar.n = maxProgress
                pbar.refresh()
                pbar.close()
                break
            time.sleep(0.1)
        computePool.join()
        results = results.get().copy()
        del samplingProgress_Actor, futures
        resetRay(numberCPUS)
    
    #Get timing data for RD generation, average, and save
    allRDTimes = np.concatenate([result.avgTimesComputeRD for result in results])
    dataPrintout = [['','Average', '', 'Standard Deviation']]
    dataPrintout.append(['RD Compute Time (s)', np.mean(allRDTimes), '+/-', np.std(allRDTimes)])
    pd.DataFrame(dataPrintout).to_csv(dir_TrainingResults + 'trainingValidation_RDTimes.csv')
    
    #Reference a result, call for result completion/printout, and sort into either training or validation sets; storing index for later lookup
    trainingDatabase, validationDatabase = [], []
    for index in tqdm(range(0, len(results)), desc='Processing', leave=True, ascii=asciiFlag):
        result = results[index]
        result.complete(trainingValidationSampleData[result.sampleDataNum])
        for sample in results[index].samples: 
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
    
    #Delete any existing models from the training directory
    for file in [file for file in glob.glob(dir_TrainingData+'*') if 'model_' in file]: 
        if os.path.isfile(file): os.remove(file)
        else: shutil.rmtree(file)

    #If consistency in the random generator is desired for comparisons, then reset seed
    resetRandom()
    
    #Initiate the specified model
    if 'SLADS' in erdModel: model = SLADS(True)
    elif erdModel == 'DLADS': model = DLADS(True, gpus)
    elif erdModel == 'DLADS-TF': model = DLADS_TF(True, gpus)
    elif erdModel == 'DLADS-PY': model = DLADS_PY(True, gpus)
    elif erdModel == 'GLANDS': sys.exit('\nError - GLANDS training procedure not yet defined')
    else: sys.exit('\nError - Specified model type does not exist')
    
    #Load and train the specified model 
    model.loadData(trainingDatabase, validationDatabase)
    model.train()