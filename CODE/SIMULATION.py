#==================================================================
#SIMULATION SPECIFIC METHOD AND CLASS DEFINITIONS
#==================================================================

#Given a set of sample paths, perform simulations using a trained SLADS Model
def simulateSampling(sortedSampleFolders, dir_Results, optimalC, modelName):
    
    #If consistency in the random generator is desired for comparisons, then reset seed
    if consistentSeed: 
        np.random.seed(0)
        random.seed(0)

    #Load the samples
    sampleDataset = [SampleData(sampleFolder, initialPercToScan, stopPerc, scanMethod, lineRevist, True, False, False, False, liveOutputFlag, False, False, False) for sampleFolder in tqdm(sortedSampleFolders, desc='Imports', leave=True, ascii=asciiFlag)]
    numJobs = len(sampleDataset)

    #Store the complete databases to disk; sometimes useful/needed for post-processing and results writeup
    if storeTestingSampleData: pickle.dump(sampleDataset, open(dir_TestingResults + 'testingSampleData.p', 'wb'))
    
    #Setup a model for each GPU available (provided there is a job for it), running once on for pre-compilation (otherwise affects reported timings)
    if (erdModel == 'DLADS' or erdModel == 'GLANDS') and numGPUs > 0: 
        models = [Model_Actor.remote(erdModel, dir_TrainingResults+modelName, gpuNum) for gpuNum in range(0, np.clip(numGPUs, 0, numJobs))]
        _ = [ray.get(model.generateERD.remote(np.empty((1,512,512,3), dtype=np.float32))) for model in models]
    else: 
        models = [Model_Actor.remote(erdModel, dir_TrainingResults+modelName)]
    numModels = len(models)
    
    #Run algorithm for each of the samples, in parallel if possible, timing and storing metric progression for each; (1 less CPU in pool for model server deployment)
    if parallelization and numJobs>1: 
    
        #Initialize a global progress bar
        maxProgress = round(float(np.sum([sampleData.stopPerc for sampleData in sampleDataset])), 2)
        pbar = tqdm(total=maxProgress, desc = '% Sampled', leave=True, ascii=asciiFlag)
        samplingProgress_Actor = SamplingProgress_Actor.remote()
        
        #Start parallel sampling operations
        if numGPUs > 0: futures = [(sampleDataset[sampleNum], optimalC, models[sampleNum%numModels], percToScan, percToViz, lineVisitAll, dir_Results, True, samplingProgress_Actor, 1.0) for sampleNum in range(0, numJobs)]
        else: futures = [(sampleDataset[sampleNum], optimalC, models[0], percToScan, percToViz, lineVisitAll, dir_Results, True, samplingProgress_Actor, 1.0) for sampleNum in range(0, numJobs)]
        computePool = Pool(numberCPUS-numModels)
        results = computePool.starmap_async(runSampling, futures)
        computePool.close()
        
        #While some results have yet to be returned, regularly update the global progress bar, then retrieve results, and purge/reset ray (actors and server)
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
        results = results.get()
        del samplingProgress_Actor
        resetRay(numberCPUS)
    else: results = [runSampling(sampleDataset[sampleNum], optimalC, models[0], percToScan, percToViz, lineVisitAll, dir_Results, False) for sampleNum in tqdm(range(0, numJobs), desc='Scanning', position=0, leave=True, ascii=asciiFlag)]

    #Perform completion/visualization routines
    chanAvgPSNR_Results, allAvgPSNR_Results, sumImagePSNR_Results, ERDPSNR_Results = [], [], [], []
    chanAvgSSIM_Results, allAvgSSIM_Results, sumImageSSIM_Results, ERDSSIM_Results = [], [], [], []
    quantityMeasured_Results, timeResults, allERDTimes = [], [], []
    
    #Set the quantity measured metric for labeling the x axes
    xLabel = '% Measured'
    #elif scanMethod == 'linewise': xLabel = '% Lines Measured'
    
    for result in tqdm(results, desc='Processing', position=0, leave=True, ascii=asciiFlag):
    
        #Export metrics and perform final analysis for the result
        result.complete()
        
        #Add individual results to rolling lists for averaging
        timeResults.append(result.finalTime)
        allERDTimes.append(result.computeERDTimes)
        chanAvgPSNR_Results.append(result.chanAvgPSNRList)
        if allChanEval: allAvgPSNR_Results.append(result.allAvgPSNRList)
        sumImagePSNR_Results.append(result.sumImagePSNRList)
        ERDPSNR_Results.append(result.ERDPSNRList)
        chanAvgSSIM_Results.append(result.chanAvgSSIMList)
        if allChanEval: allAvgSSIM_Results.append(result.allAvgSSIMList)
        sumImageSSIM_Results.append(result.sumImageSSIMList)
        ERDSSIM_Results.append(result.ERDSSIMList)
        
        #Save individual results
        np.savetxt(result.dir_sampleResults+'PSNR_chanAvg.csv', np.transpose([result.percsMeasured, result.chanAvgPSNRList]), delimiter=',')
        if allChanEval: np.savetxt(result.dir_sampleResults+'PSNR_allAvg.csv', np.transpose([result.percsMeasured, result.allAvgPSNRList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'PSNR_sumImage.csv', np.transpose([result.percsMeasured, result.sumImagePSNRList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'PSNR_ERD.csv', np.transpose([result.percsMeasured, result.ERDPSNRList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_chanAvg.csv', np.transpose([result.percsMeasured, result.chanAvgSSIMList]), delimiter=',')
        if allChanEval: np.savetxt(result.dir_sampleResults+'SSIM_allAvg.csv', np.transpose([result.percsMeasured, result.allAvgSSIMList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_sumImage.csv', np.transpose([result.percsMeasured, result.sumImageSSIMList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_ERD.csv', np.transpose([result.percsMeasured, result.ERDSSIMList]), delimiter=',')
        
        #Save individual result plots
        basicPlot(result.percsMeasured, result.chanAvgPSNRList, result.dir_sampleResults+'PSNR_chanAvg'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        if allChanEval: basicPlot(result.percsMeasured, result.allAvgPSNRList, result.dir_sampleResults+'PSNR_allAvg'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        basicPlot(result.percsMeasured, result.sumImagePSNRList, result.dir_sampleResults+'PSNR_sumImage'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        basicPlot(result.percsMeasured, result.ERDPSNRList, result.dir_sampleResults+'PSNR_ERD'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        basicPlot(result.percsMeasured, result.chanAvgSSIMList, result.dir_sampleResults+'SSIM_chanAvg'+'.png', xLabel=xLabel, yLabel='Average SSIM')
        if allChanEval: basicPlot(result.percsMeasured, result.allAvgSSIMList, result.dir_sampleResults+'SSIM_allAvg'+'.png', xLabel=xLabel, yLabel='Average SSIM')
        basicPlot(result.percsMeasured, result.sumImageSSIMList, result.dir_sampleResults+'SSIM_sumImage'+'.png', xLabel=xLabel, yLabel='Average SSIM')
        basicPlot(result.percsMeasured, result.ERDSSIMList, result.dir_sampleResults+'SSIM_ERD'+'.png', xLabel=xLabel, yLabel='Average SSIM')
        
        #If pointwise, then consider the percentage measured at each step; if linewise then consider the number of lines scanned
        #Since number of points on each line is different, this approach is not as descriptive as hoped and was therefore disabled.
        quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        #if scanMethod == 'pointwise': quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        #elif scanMethod == 'linewise': quantityMeasured_Results.append([(np.sum(np.sum(sample.mask, axis=1)>0)/sample.mask.shape[0])*100 for sample in result.samples])
        
    #Extract and average results at the specified precision
    quantityMeasured, chanAvgPSNR_Results_mean = percResults(chanAvgPSNR_Results, quantityMeasured_Results, precision)
    if allChanEval: quantityMeasured, allAvgPSNR_Results_mean = percResults(allAvgPSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, sumImagePSNR_Results_mean = percResults(sumImagePSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, ERDPSNR_Results_mean = percResults(ERDPSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, chanAvgSSIM_Results_mean = percResults(chanAvgSSIM_Results, quantityMeasured_Results, precision)
    if allChanEval: quantityMeasured, allAvgSSIM_Results_mean = percResults(allAvgSSIM_Results, quantityMeasured_Results, precision)
    quantityMeasured, sumImageSSIM_Results_mean = percResults(sumImageSSIM_Results, quantityMeasured_Results, precision)
    quantityMeasured, ERDSSIM_Results_mean = percResults(ERDSSIM_Results, quantityMeasured_Results, precision)
    
    #Compute area under the average PSNR and ERD curves
    chanAvgPSNR_AreaUnderCurve = np.trapz(chanAvgPSNR_Results_mean, quantityMeasured)
    if allChanEval: allAvgPSNR_AreaUnderCurve = np.trapz(allAvgPSNR_Results_mean, quantityMeasured)
    chanAvgSSIM_AreaUnderCurve = np.trapz(chanAvgSSIM_Results_mean, quantityMeasured)
    if allChanEval: allAvgSSIM_AreaUnderCurve = np.trapz(allAvgSSIM_Results_mean, quantityMeasured)
    ERDPSNR_AreaUnderCurve = np.trapz(ERDPSNR_Results_mean, quantityMeasured)
    
    #Save averaged results per quantity measured metric
    np.savetxt(dir_Results+'PSNR_chanAvg.csv', np.transpose([quantityMeasured, chanAvgPSNR_Results_mean]), delimiter=',')
    if allChanEval: np.savetxt(dir_Results+'PSNR_allAvg.csv', np.transpose([quantityMeasured, allAvgPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'PSNR_sumImage.csv', np.transpose([quantityMeasured, sumImagePSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'PSNR_ERD.csv', np.transpose([quantityMeasured, ERDPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'SSIM_chanAvg.csv', np.transpose([quantityMeasured, chanAvgSSIM_Results_mean]), delimiter=',')
    if allChanEval: np.savetxt(dir_Results+'SSIM_allAvg.csv', np.transpose([quantityMeasured, allAvgSSIM_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'SSIM_sumImage.csv', np.transpose([quantityMeasured, sumImageSSIM_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'SSIM_ERD.csv', np.transpose([quantityMeasured, ERDSSIM_Results_mean]), delimiter=',')

    #Export plots of averaged results
    basicPlot(quantityMeasured, chanAvgPSNR_Results_mean, dir_Results+'PSNR_chanAvg'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    if allChanEval: basicPlot(quantityMeasured, allAvgPSNR_Results_mean, dir_Results+'PSNR_allAvg'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    basicPlot(quantityMeasured, sumImagePSNR_Results_mean, dir_Results+'PSNR_sumImage'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    basicPlot(quantityMeasured, ERDPSNR_Results_mean, dir_Results+'PSNR_ERD'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    basicPlot(quantityMeasured, chanAvgSSIM_Results_mean, dir_Results+'SSIM_chanAvg'+'.png', xLabel=xLabel, yLabel='Average SSIM')
    if allChanEval: basicPlot(quantityMeasured, allAvgSSIM_Results_mean, dir_Results+'SSIM_allAvg'+'.png', xLabel=xLabel, yLabel='Average SSIM')
    basicPlot(quantityMeasured, sumImageSSIM_Results_mean, dir_Results+'SSIM_sumImage'+'.png', xLabel=xLabel, yLabel='Average SSIM')
    basicPlot(quantityMeasured, ERDSSIM_Results_mean, dir_Results+'SSIM_ERD'+'.png', xLabel=xLabel, yLabel='Average SSIM')

    #Find the final results for each image
    lastQuantityMeasured = [quantityMeasured_Results[i][-1] for i in range(0, numJobs)]
    lastChanAvgPSNR = [chanAvgPSNR_Results[i][-1] for i in range(0, numJobs)]
    if allChanEval: lastAllAvgPSNR = [allAvgPSNR_Results[i][-1] for i in range(0, numJobs)]
    lastSumImagePSNR = [sumImagePSNR_Results[i][-1] for i in range(0, numJobs)]
    lastERDPSNR = [ERDPSNR_Results[i][-1] for i in range(0, numJobs)]
    lastChanAvgSSIM = [chanAvgSSIM_Results[i][-1] for i in range(0, numJobs)]
    if allChanEval: lastAllAvgSSIM = [chanAvgSSIM_Results[i][-1] for i in range(0, numJobs)]
    lastSumImageSSIM = [sumImageSSIM_Results[i][-1] for i in range(0, numJobs)]
    lastERDSSIM = [ERDSSIM_Results[i][-1] for i in range(0, numJobs)]
    allERDTimes = np.concatenate(allERDTimes)
    
    #Print out final results 
    dataPrintout = [['','Average', '', 'Standard Deviation']]
    dataPrintout.append(['Final %:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    #elif scanMethod == 'linewise': dataPrintout.append(['Final % of Lines:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    if allChanEval: dataPrintout.append(['All PSNR:', np.mean(lastAllAvgPSNR), '+/-', np.std(lastAllAvgPSNR)])
    if allChanEval: dataPrintout.append(['All PSNR Area Under Curve:', allAvgPSNR_AreaUnderCurve])
    dataPrintout.append(['Channel PSNR:', np.mean(lastChanAvgPSNR), '+/-', np.std(lastChanAvgPSNR)])
    dataPrintout.append(['Channel PSNR Area Under Curve:', chanAvgPSNR_AreaUnderCurve])
    dataPrintout.append(['Sum Image PSNR:', np.mean(lastSumImagePSNR), '+/-', np.std(lastSumImagePSNR)])
    dataPrintout.append(['Final ERD PSNR:', np.mean(lastERDPSNR), '+/-', np.std(lastERDPSNR)])
    dataPrintout.append(['ERD PSNR Area Under Curve:', ERDPSNR_AreaUnderCurve])
    dataPrintout.append([])
    if allChanEval: dataPrintout.append(['All SSIM:', np.mean(lastAllAvgSSIM), '+/-', np.std(lastAllAvgSSIM)])
    if allChanEval: dataPrintout.append(['All SSIM Area Under Curve:', allAvgSSIM_AreaUnderCurve])
    dataPrintout.append(['Channel SSIM:', np.mean(lastChanAvgSSIM), '+/-', np.std(lastChanAvgSSIM)])
    dataPrintout.append(['Channel SSIM Area Under Curve:', chanAvgSSIM_AreaUnderCurve])
    dataPrintout.append(['Sum Image SSIM:', np.mean(lastSumImageSSIM), '+/-', np.std(lastSumImageSSIM)])
    dataPrintout.append(['Final ERD SSIM:', np.mean(lastERDSSIM), '+/-', np.std(lastERDSSIM)])
    dataPrintout.append([])
    dataPrintout.append(['ERD Compute Time (s):', np.mean(allERDTimes), '+/-', np.std(allERDTimes)])
    dataPrintout.append(['Run Time (s):', np.mean(timeResults), '+/-', np.std(timeResults)])
    pd.DataFrame(dataPrintout).to_csv(dir_Results + 'dataPrintout.csv')
