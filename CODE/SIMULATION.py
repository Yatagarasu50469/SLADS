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
            
            #Create a model for each GPU available (provided there is a job for it)
            numJobs = len(sortedSampleFolders)
            if numGPUs > 0: 
                models = [Model_Actor.remote(erdModel, dir_TrainingResults, modelName, gpus[gpuNum]) for gpuNum in range(0, np.clip(numGPUs, 0, numJobs))]
            else: models = [Model_Actor.remote(erdModel, dir_TrainingResults, modelName)]
            numModels = len(models)
            
            #Setup model on each GPU sequentially (prevents potential file conflicts)
            for model in models: _ = ray.get(model.setup.remote())
        
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
        numJobs = len(resultLocations)
    
    #Perform completion/evaluation/visualization routines
    quantityMeasured_Results, timeResults, allAvgTimesComputeERD, allAvgTimesComputeRecon, allAvgTimesFileLoad, allAvgTimesComputeIter, lastQuantityMeasured = [], [], [], [], [], [], []
    chanAvgPSNR_Results, allAvgPSNR_Results, sumImagePSNR_Results, avgERD_PSNR_Results, avgChanERD_PSNR_Results = [], [], [], [], []
    chanAvgSSIM_Results, allAvgSSIM_Results, sumImageSSIM_Results, avgERD_SSIM_Results, avgChanERD_SSIM_Results = [], [], [], [], []
    chanAvgNRMSE_Results, allAvgNRMSE_Results, sumImageNRMSE_Results, avgERD_NRMSE_Results, avgChanERD_NRMSE_Results = [], [], [], [], []
    warningOOM = False
    
    for index in tqdm(range(0, len(resultLocations)), desc='Processing', position=0, leave=True, ascii=asciiFlag):
    
        #Load pickled result and sampleData from disk
        result = pickle.load(open(resultLocations[index], "rb" ))
        sampleData = pickle.load(open(sampleDataLocations[index], "rb" ))
    
        #Export metrics and perform final analysis for the result
        result.complete(sampleData)
        
        #Check to see if OOM occurred
        if result.warningOOM: warningOOM = True
        
        #==========================================================================================================================================================================
        #Gather global results
        #==========================================================================================================================================================================
        
        #Add individual results to rolling lists for averaging
        timeResults.append(result.finalTime)
        allAvgTimesFileLoad.append(result.avgTimeFileLoad)
        allAvgTimesComputeRecon.append(result.avgTimeComputeRecon)
        allAvgTimesComputeERD.append(result.avgTimeComputeERD)
        allAvgTimesComputeIter.append(result.avgTimeComputeIter)
        
        #Accumulate the percentages measured at each step
        quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        lastQuantityMeasured.append(result.samples[-1].percMeasured)
    
        #If performing a benchmark where processing is not needed and was not performed, continue to the next loop iteration
        if benchmarkNoProcessing: continue
        
        chanAvgPSNR_Results.append(result.chanAvgPSNRList)
        if allChanEval: allAvgPSNR_Results.append(result.allAvgPSNRList)
        sumImagePSNR_Results.append(result.sumImagePSNRList)
        if erdModel != 'GLANDS': 
            avgERD_PSNR_Results.append(result.avgERD_PSNRList)
            avgChanERD_PSNR_Results.append(result.avgChanERD_PSNRList)

        chanAvgSSIM_Results.append(result.chanAvgSSIMList)
        if allChanEval: allAvgSSIM_Results.append(result.allAvgSSIMList)
        sumImageSSIM_Results.append(result.sumImageSSIMList)
        if erdModel != 'GLANDS': 
            avgERD_SSIM_Results.append(result.avgERD_SSIMList)
            avgChanERD_SSIM_Results.append(result.avgChanERD_SSIMList)
        
        chanAvgNRMSE_Results.append(result.chanAvgNRMSEList)
        if allChanEval: allAvgNRMSE_Results.append(result.allAvgNRMSEList)
        sumImageNRMSE_Results.append(result.sumImageNRMSEList)
        if erdModel != 'GLANDS': 
            avgERD_NRMSE_Results.append(result.avgERD_NRMSEList)
            avgChanERD_NRMSE_Results.append(result.avgChanERD_NRMSEList)
        #==========================================================================================================================================================================
        
        #==========================================================================================================================================================================
        #Save individual results
        #==========================================================================================================================================================================
        np.savetxt(result.dir_sampleResults+'PSNR_chanAvg.csv', np.transpose([result.percsMeasured, result.chanAvgPSNRList]), delimiter=',')
        if allChanEval: np.savetxt(result.dir_sampleResults+'PSNR_allAvg.csv', np.transpose([result.percsMeasured, result.allAvgPSNRList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'PSNR_sumImage.csv', np.transpose([result.percsMeasured, result.sumImagePSNRList]), delimiter=',')
        if erdModel != 'GLANDS': 
            np.savetxt(result.dir_sampleResults+'PSNR_avgERD.csv', np.transpose([result.percsMeasured, result.avgERD_PSNRList]), delimiter=',')
            np.savetxt(result.dir_sampleResults+'PSNR_avgChanERD.csv', np.transpose([result.percsMeasured, result.avgChanERD_PSNRList]), delimiter=',')
        
        np.savetxt(result.dir_sampleResults+'SSIM_chanAvg.csv', np.transpose([result.percsMeasured, result.chanAvgSSIMList]), delimiter=',')
        if allChanEval: np.savetxt(result.dir_sampleResults+'SSIM_allAvg.csv', np.transpose([result.percsMeasured, result.allAvgSSIMList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_sumImage.csv', np.transpose([result.percsMeasured, result.sumImageSSIMList]), delimiter=',')
        if erdModel != 'GLANDS': 
            np.savetxt(result.dir_sampleResults+'SSIM_avgERD.csv', np.transpose([result.percsMeasured, result.avgERD_SSIMList]), delimiter=',')
            np.savetxt(result.dir_sampleResults+'SSIM_avgChanERD.csv', np.transpose([result.percsMeasured, result.avgChanERD_SSIMList]), delimiter=',')
        
        np.savetxt(result.dir_sampleResults+'NRMSE_chanAvg.csv', np.transpose([result.percsMeasured, result.chanAvgNRMSEList]), delimiter=',')
        if allChanEval: np.savetxt(result.dir_sampleResults+'NRMSE_allAvg.csv', np.transpose([result.percsMeasured, result.allAvgNRMSEList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'NRMSE_sumImage.csv', np.transpose([result.percsMeasured, result.sumImageNRMSEList]), delimiter=',')
        if erdModel != 'GLANDS': 
            np.savetxt(result.dir_sampleResults+'NRMSE_avgERD.csv', np.transpose([result.percsMeasured, result.avgERD_NRMSEList]), delimiter=',')
            np.savetxt(result.dir_sampleResults+'NRMSE_avgChanERD.csv', np.transpose([result.percsMeasured, result.avgChanERD_NRMSEList]), delimiter=',')
        #==========================================================================================================================================================================
        
        #==========================================================================================================================================================================
        #Save individual result plots
        #==========================================================================================================================================================================
        basicPlot(result.percsMeasured, result.chanAvgPSNRList, result.dir_sampleResults+'PSNR_chanAvg'+'.tiff', xLabel='% Measured', yLabel='Average PSNR')
        if allChanEval: basicPlot(result.percsMeasured, result.allAvgPSNRList, result.dir_sampleResults+'PSNR_allAvg'+'.tiff', xLabel='% Measured', yLabel='Average PSNR')
        basicPlot(result.percsMeasured, result.sumImagePSNRList, result.dir_sampleResults+'PSNR_sumImage'+'.tiff', xLabel='% Measured', yLabel='Average PSNR')
        if erdModel != 'GLANDS': 
            basicPlot(result.percsMeasured, result.avgERD_PSNRList, result.dir_sampleResults+'PSNR_avgERD'+'.tiff', xLabel='% Measured', yLabel='Average PSNR')
            basicPlot(result.percsMeasured, result.avgChanERD_PSNRList, result.dir_sampleResults+'PSNR_avgChanERD'+'.tiff', xLabel='% Measured', yLabel='Average PSNR')
        
        basicPlot(result.percsMeasured, result.chanAvgSSIMList, result.dir_sampleResults+'SSIM_chanAvg'+'.tiff', xLabel='% Measured', yLabel='Average SSIM')
        if allChanEval: basicPlot(result.percsMeasured, result.allAvgSSIMList, result.dir_sampleResults+'SSIM_allAvg'+'.tiff', xLabel='% Measured', yLabel='Average SSIM')
        basicPlot(result.percsMeasured, result.sumImageSSIMList, result.dir_sampleResults+'SSIM_sumImage'+'.tiff', xLabel='% Measured', yLabel='Average SSIM')
        if erdModel != 'GLANDS': 
            basicPlot(result.percsMeasured, result.avgERD_SSIMList, result.dir_sampleResults+'SSIM_avgERD'+'.tiff', xLabel='% Measured', yLabel='Average SSIM')
            basicPlot(result.percsMeasured, result.avgChanERD_SSIMList, result.dir_sampleResults+'SSIM_avgChanERD'+'.tiff', xLabel='% Measured', yLabel='Average SSIM')
        
        basicPlot(result.percsMeasured, result.chanAvgNRMSEList, result.dir_sampleResults+'NRMSE_chanAvg'+'.tiff', xLabel='% Measured', yLabel='Average NRMSE')
        if allChanEval: basicPlot(result.percsMeasured, result.allAvgNRMSEList, result.dir_sampleResults+'NRMSE_allAvg'+'.tiff', xLabel='% Measured', yLabel='Average NRMSE')
        basicPlot(result.percsMeasured, result.sumImageNRMSEList, result.dir_sampleResults+'NRMSE_sumImage'+'.tiff', xLabel='% Measured', yLabel='Average NRMSE')
        if erdModel != 'GLANDS': 
            basicPlot(result.percsMeasured, result.avgERD_NRMSEList, result.dir_sampleResults+'NRMSE_avgERD'+'.tiff', xLabel='% Measured', yLabel='Average NRMSE')
            basicPlot(result.percsMeasured, result.avgChanERD_NRMSEList, result.dir_sampleResults+'NRMSE_avgChanERD'+'.tiff', xLabel='% Measured', yLabel='Average NRMSE')
        #==========================================================================================================================================================================
    
    #If performing a benchmark (where processing is not needed and was not performed)
    if not benchmarkNoProcessing:
        
        #==========================================================================================================================================================================
        #Extract and average results at the specified precision
        #==========================================================================================================================================================================
        quantityMeasured, chanAvgPSNR_Results_mean = percResults(chanAvgPSNR_Results, quantityMeasured_Results, precision)
        if allChanEval: quantityMeasured, allAvgPSNR_Results_mean = percResults(allAvgPSNR_Results, quantityMeasured_Results, precision)
        quantityMeasured, sumImagePSNR_Results_mean = percResults(sumImagePSNR_Results, quantityMeasured_Results, precision)
        if erdModel != 'GLANDS': 
            quantityMeasured, avgERD_PSNR_Results_mean = percResults(avgERD_PSNR_Results, quantityMeasured_Results, precision)
            quantityMeasured, avgChanERD_PSNR_Results_mean = percResults(avgChanERD_PSNR_Results, quantityMeasured_Results, precision)
        
        quantityMeasured, chanAvgSSIM_Results_mean = percResults(chanAvgSSIM_Results, quantityMeasured_Results, precision)
        if allChanEval: quantityMeasured, allAvgSSIM_Results_mean = percResults(allAvgSSIM_Results, quantityMeasured_Results, precision)
        quantityMeasured, sumImageSSIM_Results_mean = percResults(sumImageSSIM_Results, quantityMeasured_Results, precision)
        if erdModel != 'GLANDS': 
            quantityMeasured, avgERD_SSIM_Results_mean = percResults(avgERD_SSIM_Results, quantityMeasured_Results, precision)
            quantityMeasured, avgChanERD_SSIM_Results_mean = percResults(avgChanERD_SSIM_Results, quantityMeasured_Results, precision)
        
        quantityMeasured, chanAvgNRMSE_Results_mean = percResults(chanAvgNRMSE_Results, quantityMeasured_Results, precision)
        if allChanEval: quantityMeasured, allAvgNRMSE_Results_mean = percResults(allAvgNRMSE_Results, quantityMeasured_Results, precision)
        quantityMeasured, sumImageNRMSE_Results_mean = percResults(sumImageNRMSE_Results, quantityMeasured_Results, precision)
        if erdModel != 'GLANDS': 
            quantityMeasured, avgERD_NRMSE_Results_mean = percResults(avgERD_NRMSE_Results, quantityMeasured_Results, precision)
            quantityMeasured, avgChanERD_NRMSE_Results_mean = percResults(avgChanERD_NRMSE_Results, quantityMeasured_Results, precision)
        #==========================================================================================================================================================================
        
        #==========================================================================================================================================================================
        #Compute area under the average NRMSE/SSIM/PSNR curves
        #==========================================================================================================================================================================
        chanAvgPSNR_AUC = np.trapz(chanAvgPSNR_Results_mean, quantityMeasured)
        if allChanEval: allAvgPSNR_AUC = np.trapz(allAvgPSNR_Results_mean, quantityMeasured)
        if erdModel != 'GLANDS': 
            avgERD_PSNR_AUC = np.trapz(avgERD_PSNR_Results_mean, quantityMeasured)
            avgChanERD_PSNR_AUC = np.trapz(avgChanERD_PSNR_Results_mean, quantityMeasured)
        
        chanAvgSSIM_AUC = np.trapz(chanAvgSSIM_Results_mean, quantityMeasured)
        if allChanEval: allAvgSSIM_AUC = np.trapz(allAvgSSIM_Results_mean, quantityMeasured)
        if erdModel != 'GLANDS': 
            avgERD_SSIM_AUC = np.trapz(avgERD_SSIM_Results_mean, quantityMeasured)
            avgChanERD_SSIM_AUC = np.trapz(avgChanERD_SSIM_Results_mean, quantityMeasured)
        
        chanAvgNRMSE_AUC = np.trapz(chanAvgNRMSE_Results_mean, quantityMeasured)
        if allChanEval: allAvgNRMSE_AUC = np.trapz(allAvgNRMSE_Results_mean, quantityMeasured)
        if erdModel != 'GLANDS': 
            avgERD_NRMSE_AUC = np.trapz(avgERD_NRMSE_Results_mean, quantityMeasured)
            avgChanERD_NRMSE_AUC = np.trapz(avgChanERD_NRMSE_Results_mean, quantityMeasured)
        #==========================================================================================================================================================================
        
        #==========================================================================================================================================================================
        #Save averaged results per quantity measured metric
        #==========================================================================================================================================================================
        np.savetxt(dir_Results+'PSNR_chanAvg.csv', np.transpose([quantityMeasured, chanAvgPSNR_Results_mean]), delimiter=',')
        if allChanEval: np.savetxt(dir_Results+'PSNR_allAvg.csv', np.transpose([quantityMeasured, allAvgPSNR_Results_mean]), delimiter=',')
        np.savetxt(dir_Results+'PSNR_sumImage.csv', np.transpose([quantityMeasured, sumImagePSNR_Results_mean]), delimiter=',')
        if erdModel != 'GLANDS': 
            np.savetxt(dir_Results+'PSNR_avgERD.csv', np.transpose([quantityMeasured, avgERD_PSNR_Results_mean]), delimiter=',')
            np.savetxt(dir_Results+'PSNR_avgChanERD.csv', np.transpose([quantityMeasured, avgChanERD_PSNR_Results_mean]), delimiter=',')
        
        np.savetxt(dir_Results+'SSIM_chanAvg.csv', np.transpose([quantityMeasured, chanAvgSSIM_Results_mean]), delimiter=',')
        if allChanEval: np.savetxt(dir_Results+'SSIM_allAvg.csv', np.transpose([quantityMeasured, allAvgSSIM_Results_mean]), delimiter=',')
        np.savetxt(dir_Results+'SSIM_sumImage.csv', np.transpose([quantityMeasured, sumImageSSIM_Results_mean]), delimiter=',')
        if erdModel != 'GLANDS': 
            np.savetxt(dir_Results+'SSIM_avgERD.csv', np.transpose([quantityMeasured, avgERD_SSIM_Results_mean]), delimiter=',')
            np.savetxt(dir_Results+'SSIM_avgChanERD.csv', np.transpose([quantityMeasured, avgChanERD_SSIM_Results_mean]), delimiter=',')
        
        np.savetxt(dir_Results+'NRMSE_chanAvg.csv', np.transpose([quantityMeasured, chanAvgNRMSE_Results_mean]), delimiter=',')
        if allChanEval: np.savetxt(dir_Results+'NRMSE_allAvg.csv', np.transpose([quantityMeasured, allAvgNRMSE_Results_mean]), delimiter=',')
        np.savetxt(dir_Results+'NRMSE_sumImage.csv', np.transpose([quantityMeasured, sumImageNRMSE_Results_mean]), delimiter=',')
        if erdModel != 'GLANDS': 
            np.savetxt(dir_Results+'NRMSE_avgERD.csv', np.transpose([quantityMeasured, avgERD_NRMSE_Results_mean]), delimiter=',')
            np.savetxt(dir_Results+'NRMSE_avgChanERD.csv', np.transpose([quantityMeasured, avgChanERD_NRMSE_Results_mean]), delimiter=',')
        #==========================================================================================================================================================================
        
        #==========================================================================================================================================================================
        #Export plots of averaged results
        #==========================================================================================================================================================================
        basicPlot(quantityMeasured, chanAvgPSNR_Results_mean, dir_Results+'PSNR_chanAvg'+'.tiff', xLabel='% Measured', yLabel='Average PSNR')
        if allChanEval: basicPlot(quantityMeasured, allAvgPSNR_Results_mean, dir_Results+'PSNR_allAvg'+'.tiff', xLabel='% Measured', yLabel='Average PSNR')
        basicPlot(quantityMeasured, sumImagePSNR_Results_mean, dir_Results+'PSNR_sumImage'+'.tiff', xLabel='% Measured', yLabel='Average PSNR')
        if erdModel != 'GLANDS': 
            basicPlot(quantityMeasured, avgERD_PSNR_Results_mean, dir_Results+'PSNR_avgERD'+'.tiff', xLabel='% Measured', yLabel='Average PSNR')
            basicPlot(quantityMeasured, avgChanERD_PSNR_Results_mean, dir_Results+'PSNR_avgChanERD'+'.tiff', xLabel='% Measured', yLabel='Average PSNR')
        
        basicPlot(quantityMeasured, chanAvgSSIM_Results_mean, dir_Results+'SSIM_chanAvg'+'.tiff', xLabel='% Measured', yLabel='Average SSIM')
        if allChanEval: basicPlot(quantityMeasured, allAvgSSIM_Results_mean, dir_Results+'SSIM_allAvg'+'.tiff', xLabel='% Measured', yLabel='Average SSIM')
        basicPlot(quantityMeasured, sumImageSSIM_Results_mean, dir_Results+'SSIM_sumImage'+'.tiff', xLabel='% Measured', yLabel='Average SSIM')
        if erdModel != 'GLANDS': 
            basicPlot(quantityMeasured, avgERD_SSIM_Results_mean, dir_Results+'SSIM_avgERD'+'.tiff', xLabel='% Measured', yLabel='Average SSIM')
            basicPlot(quantityMeasured, avgChanERD_SSIM_Results_mean, dir_Results+'SSIM_avgChanERD'+'.tiff', xLabel='% Measured', yLabel='Average SSIM')
        
        basicPlot(quantityMeasured, chanAvgNRMSE_Results_mean, dir_Results+'NRMSE_chanAvg'+'.tiff', xLabel='% Measured', yLabel='Average NRMSE')
        if allChanEval: basicPlot(quantityMeasured, allAvgNRMSE_Results_mean, dir_Results+'NRMSE_allAvg'+'.tiff', xLabel='% Measured', yLabel='Average NRMSE')
        basicPlot(quantityMeasured, sumImageNRMSE_Results_mean, dir_Results+'NRMSE_sumImage'+'.tiff', xLabel='% Measured', yLabel='Average NRMSE')
        if erdModel != 'GLANDS': 
            basicPlot(quantityMeasured, avgERD_NRMSE_Results_mean, dir_Results+'NRMSE_avgERD'+'.tiff', xLabel='% Measured', yLabel='Average NRMSE')
            basicPlot(quantityMeasured, avgChanERD_NRMSE_Results_mean, dir_Results+'NRMSE_avgChanERD'+'.tiff', xLabel='% Measured', yLabel='Average NRMSE')
        #==========================================================================================================================================================================
        
        #==========================================================================================================================================================================
        #Find the final results for each image
        #==========================================================================================================================================================================
        lastChanAvgPSNR = [chanAvgPSNR_Results[i][-1] for i in range(0, numJobs)]
        if allChanEval: lastAllAvgPSNR = [allAvgPSNR_Results[i][-1] for i in range(0, numJobs)]
        lastSumImagePSNR = [sumImagePSNR_Results[i][-1] for i in range(0, numJobs)]
        if erdModel != 'GLANDS': 
            lastAvgERD_PSNR = [avgERD_PSNR_Results[i][-1] for i in range(0, numJobs)]
            lastAvgChanERD_PSNR = [avgChanERD_PSNR_Results[i][-1] for i in range(0, numJobs)]
        
        lastChanAvgSSIM = [chanAvgSSIM_Results[i][-1] for i in range(0, numJobs)]
        if allChanEval: lastAllAvgSSIM = [allAvgSSIM_Results[i][-1] for i in range(0, numJobs)]
        lastSumImageSSIM = [sumImageSSIM_Results[i][-1] for i in range(0, numJobs)]
        if erdModel != 'GLANDS': 
            lastAvgERD_SSIM = [avgERD_SSIM_Results[i][-1] for i in range(0, numJobs)]
            lastAvgChanERD_SSIM = [avgChanERD_SSIM_Results[i][-1] for i in range(0, numJobs)]
        
        lastChanAvgNRMSE = [chanAvgNRMSE_Results[i][-1] for i in range(0, numJobs)]
        if allChanEval: lastAllAvgNRMSE = [allAvgNRMSE_Results[i][-1] for i in range(0, numJobs)]
        lastSumImageNRMSE = [sumImageNRMSE_Results[i][-1] for i in range(0, numJobs)]
        if erdModel != 'GLANDS': 
            lastAvgERD_NRMSE = [avgERD_NRMSE_Results[i][-1] for i in range(0, numJobs)]
            lastAvgChanERD_NRMSE = [avgChanERD_NRMSE_Results[i][-1] for i in range(0, numJobs)]
        #==========================================================================================================================================================================
        
    #Setup to print/save final results 
    dataPrintout = [['','Average', '', 'Standard Deviation']]
    dataPrintout.append(['Final %:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    
    #If not performing a benchmark where processing is not needed and was not performed
    if not benchmarkNoProcessing:
    
        #==========================================================================================================================================================================
        #Format averaged metrics
        #==========================================================================================================================================================================
        if allChanEval: dataPrintout.append(['All Channel PSNR:', np.mean(lastAllAvgPSNR), '+/-', np.std(lastAllAvgPSNR)])
        if allChanEval: dataPrintout.append(['All Channel PSNR AUC:', allAvgPSNR_AUC])
        dataPrintout.append(['Targeted Channel PSNR:', np.mean(lastChanAvgPSNR), '+/-', np.std(lastChanAvgPSNR)])
        dataPrintout.append(['Targeted Channel PSNR AUC:', chanAvgPSNR_AUC])
        dataPrintout.append(['Sum Image PSNR:', np.mean(lastSumImagePSNR), '+/-', np.std(lastSumImagePSNR)])
        if erdModel != 'GLANDS': 
            dataPrintout.append(['Final Avg ERD PSNR:', np.mean(lastAvgERD_PSNR), '+/-', np.std(lastAvgERD_PSNR)])
            dataPrintout.append(['ERD Avg PSNR AUC:', avgERD_PSNR_AUC])
            dataPrintout.append(['Final Avg Chan ERD PSNR:', np.mean(lastAvgChanERD_PSNR), '+/-', np.std(lastAvgChanERD_PSNR)])
            dataPrintout.append(['ERD Avg Chan PSNR AUC:', avgChanERD_PSNR_AUC])
        dataPrintout.append([])
        
        if allChanEval: dataPrintout.append(['All Channel SSIM:', np.mean(lastAllAvgSSIM), '+/-', np.std(lastAllAvgSSIM)])
        if allChanEval: dataPrintout.append(['All Channel SSIM AUC:', allAvgSSIM_AUC])
        dataPrintout.append(['Targeted Channel SSIM:', np.mean(lastChanAvgSSIM), '+/-', np.std(lastChanAvgSSIM)])
        dataPrintout.append(['Targeted Channel SSIM AUC:', chanAvgSSIM_AUC])
        dataPrintout.append(['Sum Image SSIM:', np.mean(lastSumImageSSIM), '+/-', np.std(lastSumImageSSIM)])
        if erdModel != 'GLANDS': 
            dataPrintout.append(['Final Avg ERD SSIM:', np.mean(lastAvgERD_SSIM), '+/-', np.std(lastAvgERD_SSIM)])
            dataPrintout.append(['ERD Avg SSIM AUC:', avgERD_SSIM_AUC])
            dataPrintout.append(['Final Avg Chan ERD SSIM:', np.mean(lastAvgChanERD_SSIM), '+/-', np.std(lastAvgChanERD_SSIM)])
            dataPrintout.append(['ERD Avg Chan SSIM AUC:', avgChanERD_SSIM_AUC])
        dataPrintout.append([])
        
        if allChanEval: dataPrintout.append(['All Channel NRMSE:', np.mean(lastAllAvgNRMSE), '+/-', np.std(lastAllAvgNRMSE)])
        if allChanEval: dataPrintout.append(['All Channel NRMSE AUC:', allAvgNRMSE_AUC])
        dataPrintout.append(['Targeted Channel NRMSE:', np.mean(lastChanAvgNRMSE), '+/-', np.std(lastChanAvgNRMSE)])
        dataPrintout.append(['Targeted Channel NRMSE AUC:', chanAvgNRMSE_AUC])
        dataPrintout.append(['Sum Image NRMSE:', np.mean(lastSumImageNRMSE), '+/-', np.std(lastSumImageNRMSE)])
        if erdModel != 'GLANDS': 
            dataPrintout.append(['Final Avg ERD NRMSE:', np.mean(lastAvgERD_NRMSE), '+/-', np.std(lastAvgERD_NRMSE)])
            dataPrintout.append(['Avg ERD NRMSE AUC:', avgERD_NRMSE_AUC])
            dataPrintout.append(['Final Avg Chan ERD NRMSE:', np.mean(lastAvgChanERD_NRMSE), '+/-', np.std(lastAvgChanERD_NRMSE)])
            dataPrintout.append(['Avg Chan ERD NRMSE AUC:', avgChanERD_NRMSE_AUC])
        dataPrintout.append([])
        #==========================================================================================================================================================================
        
        #Save metrics to disk
        pd.DataFrame(dataPrintout).to_csv(dir_Results + 'dataPrintout.csv', index=False)
        
    #Store timing metrics
    timePrintout = []
    timePrintout.append(['Run Time (s):', np.mean(timeResults)])
    timePrintout.append([])
    timePrintout.append(['Note: The following values should not be considered wholly representative of actual code performance/efficiency.'])
    timePrintout.append(['They are only intended for debugging and internal evaluation of relative system performance!'])
    timePrintout.append([])
    timePrintout.append(['File Load Time (s):', np.mean(allAvgTimesFileLoad)])
    if erdModel != 'GLANDS': 
        timePrintout.append(['Targeted Reconstruction Compute Time (s):', np.mean(allAvgTimesComputeRecon)])
        timePrintout.append(['ERD Compute Time (s):', np.mean(allAvgTimesComputeERD)])
    timePrintout.append(['Targeted Compute Time (s):', np.mean(allAvgTimesComputeIter)])
    pd.DataFrame(timePrintout).to_csv(dir_Results + 'timePrintout.csv', index=False)
    
    #If an OOM occurred, then notify the user before exiting
    if allChanEval and warningOOM: print('\nWarning - Fallback was used for allChanEval reconstructions; insufficient RAM was available for simultaneous operations on all workers.')

    #Delete pickled result and sampleData data from disk if they aren't needed for later bypassSampling configuration
    if not keepResultData: 
        for resultLocation in resultLocations: os.remove(resultLocation) 
        for sampleDataLocation in sampleDataLocations: os.remove(sampleDataLocation)  
        os.remove(dir_TestingResults + 'resultLocations.p')
        os.remove(dir_TestingResults + 'sampleDataLocations.p')
    