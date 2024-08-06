#==================================================================
#SIMULATION
#==================================================================

#Given a set of sample paths, perform simulations using a trained SLADS Model
def simulateSampling(sortedSampleFolders, dir_Results, optimalC, modelName):
    
    #If consistency in the random generator is desired for comparisons, then reset seed
    resetRandom()
    
    #If results and testingSampleData already exist (needed for RAM OOM reduction development), can bypass sampling process (provided keepResultData was enabled previously)
    if not bypassSampling:
    
        #Load the samples, storing the complete databases to disk if indicated; sometimes useful/needed for post-processing and results writeup
        sampleDataset = [SampleData(sampleFolder, initialPercToScan, stopPerc, scanMethod, lineRevist, True, False, False, False, liveOutputFlag, False, False, False) for sampleFolder in tqdm(sortedSampleFolders, desc='Imports', leave=True, ascii=asciiFlag)]
        if storeTestingSampleData: pickle.dump(sampleDataset, open(dir_TestingResults + 'testingSampleData.p', 'wb'))
        
        #Ray sends warning as error message regarding size; temporarily redirect output during model loading to prevent spurious output(s); https://github.com/ray-project/ray/issues/43264
        with suppressSTD() if not debugMode else nullcontext():
            
            #Setup a model for each GPU available (provided there is a job for it)
            numJobs = len(sortedSampleFolders)
            if numGPUs > 0: models = [Model_Actor.remote(erdModel, dir_TrainingResults, modelName, gpus[gpuNum]) for gpuNum in range(0, np.clip(numGPUs, 0, numJobs))]
            else: models = [Model_Actor.remote(erdModel, dir_TrainingResults, modelName)]
            numModels = len(models)
        
        #Run algorithm for each of the samples, in parallel if possible, timing and storing metric progression for each; (1 less CPU in pool for model server deployment)
        if parallelization and numJobs>1: 
        
            #Initialize a global progress bar
            maxProgress = round(float(np.sum([sampleData.stopPerc for sampleData in sampleDataset])), 2)
            pbar = tqdm(total=maxProgress, desc = '% Sampled', leave=True, ascii=asciiFlag)
            samplingProgress_Actor = SamplingProgress_Actor.remote()
            
            #Start parallel sampling operations
            if numGPUs > 0: futures = [(sampleNum, sampleDataset[sampleNum], optimalC, models[sampleNum%numModels], percToScan, percToViz, lineVisitAll, dir_Results, True, samplingProgress_Actor, 0.01) for sampleNum in range(0, numJobs)]
            else: futures = [(sampleNum, sampleDataset[sampleNum], optimalC, models[0], percToScan, percToViz, lineVisitAll, dir_Results, True, samplingProgress_Actor, 0.01) for sampleNum in range(0, numJobs)]
            computePool = Pool(numberCPUS-numModels)
            results = computePool.starmap_async(runSampling, futures)
            computePool.close()
            
            #While some results have yet to be returned, regularly update the global progress bar, then retrieve results, and purge/reset ray (actors and server)
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
        else: results = [runSampling(sampleNum, sampleDataset[sampleNum], optimalC, models[0], percToScan, percToViz, lineVisitAll, dir_Results, False) for sampleNum in tqdm(range(0, numJobs), desc='Scanning', position=0, leave=True, ascii=asciiFlag)]
        
        #Pickle each result and sampleData to disk in order to help prevent OOM during processing
        resultLocations, sampleDataLocations = [], []
        for result in results:
            resultLocation = dir_TestingResults + sampleDataset[result.sampleDataNum].name + '_result.p'
            pickle.dump(result, open(resultLocation, 'wb'))
            resultLocations.append(resultLocation)
            sampleDataLocation = dir_TestingResults + sampleDataset[result.sampleDataNum].name + '_sampleData.p'
            pickle.dump(sampleDataset[result.sampleDataNum], open(sampleDataLocation, 'wb'))
            sampleDataLocations.append(sampleDataLocation)
        pickle.dump(resultLocations, open(dir_TestingResults + 'resultLocations.p', 'wb'))
        pickle.dump(sampleDataLocations, open(dir_TestingResults + 'sampleDataLocations.p', 'wb'))
        del results, sampleDataset
    else:
        sampleDataLocations = pickle.load(open(dir_TestingResults + 'sampleDataLocations.p', "rb" ))
        resultLocations = pickle.load(open(dir_TestingResults + 'resultLocations.p', "rb" ))
    
    #Perform completion/visualization routines
    chanAvgNRMSE_Results, allAvgNRMSE_Results, sumImageNRMSE_Results, ERD_NRMSE_Results = [], [], [], []
    chanAvgSSIM_Results, allAvgSSIM_Results, sumImageSSIM_Results, ERD_SSIM_Results = [], [], [], []
    quantityMeasured_Results, timeResults, allAvgTimesComputeERD, allAvgTimesComputeRecon, allAvgTimesFileLoad, allAvgTimesComputeIter, lastQuantityMeasured = [], [], [], [], [], [], []
    
    #Set the quantity measured metric for labeling the x axes
    xLabel = '% Measured'
    #elif scanMethod == 'linewise': xLabel = '% Lines Measured'
    
    for index in tqdm(range(0, len(resultLocations)), desc='Processing', position=0, leave=True, ascii=asciiFlag):
    
        #Load pickled result and sampleData from disk
        result = pickle.load(open(resultLocations[index], "rb" ))
        sampleData = pickle.load(open(sampleDataLocations[index], "rb" ))
    
        #Export metrics and perform final analysis for the result
        result.complete(sampleData)
        
        #Add individual results to rolling lists for averaging
        timeResults.append(result.finalTime)
        allAvgTimesFileLoad.append(result.avgTimeFileLoad)
        allAvgTimesComputeRecon.append(result.avgTimeComputeRecon)
        allAvgTimesComputeERD.append(result.avgTimeComputeERD)
        allAvgTimesComputeIter.append(result.avgTimeComputeIter)
        
        #If pointwise, then consider the percentage measured at each step; if linewise then consider the number of lines scanned
        #Since number of points on each line is different, this approach is not as descriptive as hoped and was therefore disabled.
        quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        lastQuantityMeasured.append(result.samples[-1].percMeasured)
        #if scanMethod == 'pointwise': quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        #elif scanMethod == 'linewise': quantityMeasured_Results.append([(np.sum(np.sum(sample.mask, axis=1)>0)/sample.mask.shape[0])*100 for sample in result.samples])
    
        #If performing a benchmark where processing is not needed and was not performed, continue to the next loop iteration
        if benchmarkNoProcessing: continue
        
        chanAvgNRMSE_Results.append(result.chanAvgNRMSEList)
        if allChanEval: allAvgNRMSE_Results.append(result.allAvgNRMSEList)
        sumImageNRMSE_Results.append(result.sumImageNRMSEList)
        if erdModel != 'GLANDS': ERD_NRMSE_Results.append(result.ERD_NRMSEList)
        chanAvgSSIM_Results.append(result.chanAvgSSIMList)
        if allChanEval: allAvgSSIM_Results.append(result.allAvgSSIMList)
        sumImageSSIM_Results.append(result.sumImageSSIMList)
        if erdModel != 'GLANDS': ERD_SSIM_Results.append(result.ERD_SSIMList)
        
        #Save individual results
        np.savetxt(result.dir_sampleResults+'NRMSE_chanAvg.csv', np.transpose([result.percsMeasured, result.chanAvgNRMSEList]), delimiter=',')
        if allChanEval: np.savetxt(result.dir_sampleResults+'NRMSE_allAvg.csv', np.transpose([result.percsMeasured, result.allAvgNRMSEList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'NRMSE_sumImage.csv', np.transpose([result.percsMeasured, result.sumImageNRMSEList]), delimiter=',')
        if erdModel != 'GLANDS': np.savetxt(result.dir_sampleResults+'NRMSE_ERD.csv', np.transpose([result.percsMeasured, result.ERD_NRMSEList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_chanAvg.csv', np.transpose([result.percsMeasured, result.chanAvgSSIMList]), delimiter=',')
        if allChanEval: np.savetxt(result.dir_sampleResults+'SSIM_allAvg.csv', np.transpose([result.percsMeasured, result.allAvgSSIMList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_sumImage.csv', np.transpose([result.percsMeasured, result.sumImageSSIMList]), delimiter=',')
        if erdModel != 'GLANDS': np.savetxt(result.dir_sampleResults+'SSIM_ERD.csv', np.transpose([result.percsMeasured, result.ERD_SSIMList]), delimiter=',')
        
        #Save individual result plots
        basicPlot(result.percsMeasured, result.chanAvgNRMSEList, result.dir_sampleResults+'NRMSE_chanAvg'+'.tiff', xLabel=xLabel, yLabel='Average NRMSE')
        if allChanEval: basicPlot(result.percsMeasured, result.allAvgNRMSEList, result.dir_sampleResults+'NRMSE_allAvg'+'.tiff', xLabel=xLabel, yLabel='Average NRMSE')
        basicPlot(result.percsMeasured, result.sumImageNRMSEList, result.dir_sampleResults+'NRMSE_sumImage'+'.tiff', xLabel=xLabel, yLabel='Average NRMSE')
        if erdModel != 'GLANDS': basicPlot(result.percsMeasured, result.ERD_NRMSEList, result.dir_sampleResults+'NRMSE_ERD'+'.tiff', xLabel=xLabel, yLabel='Average NRMSE')
        basicPlot(result.percsMeasured, result.chanAvgSSIMList, result.dir_sampleResults+'SSIM_chanAvg'+'.tiff', xLabel=xLabel, yLabel='Average SSIM')
        if allChanEval: basicPlot(result.percsMeasured, result.allAvgSSIMList, result.dir_sampleResults+'SSIM_allAvg'+'.tiff', xLabel=xLabel, yLabel='Average SSIM')
        basicPlot(result.percsMeasured, result.sumImageSSIMList, result.dir_sampleResults+'SSIM_sumImage'+'.tiff', xLabel=xLabel, yLabel='Average SSIM')
        if erdModel != 'GLANDS': basicPlot(result.percsMeasured, result.ERD_SSIMList, result.dir_sampleResults+'SSIM_ERD'+'.tiff', xLabel=xLabel, yLabel='Average SSIM')
    
    #Delete pickled result and sampleData data from disk if they aren't needed for later bypassSampling configuration
    if not keepSamplingResultBypassData: 
        for resultLocation in resultLocations: os.remove(resultLocation) 
        for sampleDataLocation in sampleDataLocations: os.remove(sampleDataLocation)  
        os.remove(dir_TestingResults + 'resultLocations.p')
        os.remove(dir_TestingResults + 'sampleDataLocations.p')
    
    #If performing a benchmark where processing is not needed and was not performed
    if not benchmarkNoProcessing:
    
        #Extract and average results at the specified precision
        quantityMeasured, chanAvgNRMSE_Results_mean = percResults(chanAvgNRMSE_Results, quantityMeasured_Results, precision)
        if allChanEval: quantityMeasured, allAvgNRMSE_Results_mean = percResults(allAvgNRMSE_Results, quantityMeasured_Results, precision)
        quantityMeasured, sumImageNRMSE_Results_mean = percResults(sumImageNRMSE_Results, quantityMeasured_Results, precision)
        if erdModel != 'GLANDS': quantityMeasured, ERD_NRMSE_Results_mean = percResults(ERD_NRMSE_Results, quantityMeasured_Results, precision)
        quantityMeasured, chanAvgSSIM_Results_mean = percResults(chanAvgSSIM_Results, quantityMeasured_Results, precision)
        if allChanEval: quantityMeasured, allAvgSSIM_Results_mean = percResults(allAvgSSIM_Results, quantityMeasured_Results, precision)
        quantityMeasured, sumImageSSIM_Results_mean = percResults(sumImageSSIM_Results, quantityMeasured_Results, precision)
        if erdModel != 'GLANDS': quantityMeasured, ERD_SSIM_Results_mean = percResults(ERD_SSIM_Results, quantityMeasured_Results, precision)
        
        #Compute area under the average NRMSE and ERD curves
        chanAvgNRMSE_AreaUnderCurve = np.trapz(chanAvgNRMSE_Results_mean, quantityMeasured)
        if allChanEval: allAvgNRMSE_AreaUnderCurve = np.trapz(allAvgNRMSE_Results_mean, quantityMeasured)
        chanAvgSSIM_AreaUnderCurve = np.trapz(chanAvgSSIM_Results_mean, quantityMeasured)
        if allChanEval: allAvgSSIM_AreaUnderCurve = np.trapz(allAvgSSIM_Results_mean, quantityMeasured)
        if erdModel != 'GLANDS': ERD_NRMSE_AreaUnderCurve = np.trapz(ERD_NRMSE_Results_mean, quantityMeasured)
        if erdModel != 'GLANDS': ERD_SSIM_AreaUnderCurve = np.trapz(ERD_SSIM_Results_mean, quantityMeasured)
        
        #Save averaged results per quantity measured metric
        np.savetxt(dir_Results+'NRMSE_chanAvg.csv', np.transpose([quantityMeasured, chanAvgNRMSE_Results_mean]), delimiter=',')
        if allChanEval: np.savetxt(dir_Results+'NRMSE_allAvg.csv', np.transpose([quantityMeasured, allAvgNRMSE_Results_mean]), delimiter=',')
        np.savetxt(dir_Results+'NRMSE_sumImage.csv', np.transpose([quantityMeasured, sumImageNRMSE_Results_mean]), delimiter=',')
        if erdModel != 'GLANDS': np.savetxt(dir_Results+'NRMSE_ERD.csv', np.transpose([quantityMeasured, ERD_NRMSE_Results_mean]), delimiter=',')
        np.savetxt(dir_Results+'SSIM_chanAvg.csv', np.transpose([quantityMeasured, chanAvgSSIM_Results_mean]), delimiter=',')
        if allChanEval: np.savetxt(dir_Results+'SSIM_allAvg.csv', np.transpose([quantityMeasured, allAvgSSIM_Results_mean]), delimiter=',')
        np.savetxt(dir_Results+'SSIM_sumImage.csv', np.transpose([quantityMeasured, sumImageSSIM_Results_mean]), delimiter=',')
        if erdModel != 'GLANDS': np.savetxt(dir_Results+'SSIM_ERD.csv', np.transpose([quantityMeasured, ERD_SSIM_Results_mean]), delimiter=',')

        #Export plots of averaged results
        basicPlot(quantityMeasured, chanAvgNRMSE_Results_mean, dir_Results+'NRMSE_chanAvg'+'.tiff', xLabel=xLabel, yLabel='Average NRMSE')
        if allChanEval: basicPlot(quantityMeasured, allAvgNRMSE_Results_mean, dir_Results+'NRMSE_allAvg'+'.tiff', xLabel=xLabel, yLabel='Average NRMSE')
        basicPlot(quantityMeasured, sumImageNRMSE_Results_mean, dir_Results+'NRMSE_sumImage'+'.tiff', xLabel=xLabel, yLabel='Average NRMSE')
        if erdModel != 'GLANDS': basicPlot(quantityMeasured, ERD_NRMSE_Results_mean, dir_Results+'NRMSE_ERD'+'.tiff', xLabel=xLabel, yLabel='Average NRMSE')
        basicPlot(quantityMeasured, chanAvgSSIM_Results_mean, dir_Results+'SSIM_chanAvg'+'.tiff', xLabel=xLabel, yLabel='Average SSIM')
        if allChanEval: basicPlot(quantityMeasured, allAvgSSIM_Results_mean, dir_Results+'SSIM_allAvg'+'.tiff', xLabel=xLabel, yLabel='Average SSIM')
        basicPlot(quantityMeasured, sumImageSSIM_Results_mean, dir_Results+'SSIM_sumImage'+'.tiff', xLabel=xLabel, yLabel='Average SSIM')
        if erdModel != 'GLANDS': basicPlot(quantityMeasured, ERD_SSIM_Results_mean, dir_Results+'SSIM_ERD'+'.tiff', xLabel=xLabel, yLabel='Average SSIM')

        #Find the final results for each image
        lastChanAvgNRMSE = [chanAvgNRMSE_Results[i][-1] for i in range(0, numJobs)]
        if allChanEval: lastAllAvgNRMSE = [allAvgNRMSE_Results[i][-1] for i in range(0, numJobs)]
        lastSumImageNRMSE = [sumImageNRMSE_Results[i][-1] for i in range(0, numJobs)]
        if erdModel != 'GLANDS': lastERD_NRMSE = [ERD_NRMSE_Results[i][-1] for i in range(0, numJobs)]
        lastChanAvgSSIM = [chanAvgSSIM_Results[i][-1] for i in range(0, numJobs)]
        if allChanEval: lastAllAvgSSIM = [chanAvgSSIM_Results[i][-1] for i in range(0, numJobs)]
        lastSumImageSSIM = [sumImageSSIM_Results[i][-1] for i in range(0, numJobs)]
        if erdModel != 'GLANDS': lastERD_SSIM = [ERD_SSIM_Results[i][-1] for i in range(0, numJobs)]
        
    #Print out final results 
    dataPrintout = [['','Average', '', 'Standard Deviation']]
    dataPrintout.append(['Final %:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    #elif scanMethod == 'linewise': dataPrintout.append(['Final % of Lines:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    
    #If not performing a benchmark where processing is not needed and was not performed
    if not benchmarkNoProcessing:
        if allChanEval: dataPrintout.append(['All Channel NRMSE:', np.mean(lastAllAvgNRMSE), '+/-', np.std(lastAllAvgNRMSE)])
        if allChanEval: dataPrintout.append(['All Channel NRMSE Area Under Curve:', allAvgNRMSE_AreaUnderCurve])
        dataPrintout.append(['Targeted Channel NRMSE:', np.mean(lastChanAvgNRMSE), '+/-', np.std(lastChanAvgNRMSE)])
        dataPrintout.append(['Targeted Channel NRMSE Area Under Curve:', chanAvgNRMSE_AreaUnderCurve])
        dataPrintout.append(['Sum Image NRMSE:', np.mean(lastSumImageNRMSE), '+/-', np.std(lastSumImageNRMSE)])
        if erdModel != 'GLANDS': dataPrintout.append(['Final ERD NRMSE:', np.mean(lastERD_NRMSE), '+/-', np.std(lastERD_NRMSE)])
        if erdModel != 'GLANDS': dataPrintout.append(['ERD NRMSE Area Under Curve:', ERD_NRMSE_AreaUnderCurve])
        dataPrintout.append([])
        if allChanEval: dataPrintout.append(['All Channel SSIM:', np.mean(lastAllAvgSSIM), '+/-', np.std(lastAllAvgSSIM)])
        if allChanEval: dataPrintout.append(['All Channel SSIM Area Under Curve:', allAvgSSIM_AreaUnderCurve])
        dataPrintout.append(['Targeted Channel SSIM:', np.mean(lastChanAvgSSIM), '+/-', np.std(lastChanAvgSSIM)])
        dataPrintout.append(['Targeted Channel SSIM Area Under Curve:', chanAvgSSIM_AreaUnderCurve])
        dataPrintout.append(['Sum Image SSIM:', np.mean(lastSumImageSSIM), '+/-', np.std(lastSumImageSSIM)])
        if erdModel != 'GLANDS': dataPrintout.append(['Final ERD SSIM:', np.mean(lastERD_SSIM), '+/-', np.std(lastERD_SSIM)])
        if erdModel != 'GLANDS': dataPrintout.append(['ERD SSIM Area Under Curve:', ERD_SSIM_AreaUnderCurve])
    dataPrintout.append([])
    dataPrintout.append(['Run Time (s):', np.mean(timeResults)])
    dataPrintout.append([])
    dataPrintout.append(['Note: The following values should not be considered wholly representative of actual code performance/efficiency.'])
    dataPrintout.append(['They are only intended for debugging and internal evaluation of relative system performance!'])
    dataPrintout.append([])
    dataPrintout.append(['File Load Time (s):', np.mean(allAvgTimesFileLoad)])
    if erdModel != 'GLANDS': dataPrintout.append(['Targeted Reconstruction Compute Time (s):', np.mean(allAvgTimesComputeRecon)])
    if erdModel != 'GLANDS': dataPrintout.append(['ERD Compute Time (s):', np.mean(allAvgTimesComputeERD)])
    dataPrintout.append(['Targeted Compute Time (s):', np.mean(allAvgTimesComputeIter)])
    pd.DataFrame(dataPrintout).to_csv(dir_Results + 'dataPrintout.csv', index=False)
    