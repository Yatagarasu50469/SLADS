#==================================================================
#TESTING SLADS SPECIFIC
#==================================================================

#Given a set of sample paths, perform simulations using a trained SLADS Model
def simulateSLADS(sortedSampleFolders, dir_Results, optimalC, modelName):

    #If consistency in the random generator is desired for comparisons, then reset seed
    if consistentSeed: 
        tf.random.set_seed(0)
        np.random.seed(0)
        random.seed(0)
    
    #Load the samples
    sampleDataset = [SampleData(sampleFolder, initialPercToScan, stopPerc, scanMethod, lineRevist, False, True) for sampleFolder in tqdm(sortedSampleFolders, desc='Reading', leave=True, ascii=True)]

    #Start server, deploy, and get handle for model queries; run Tensorflow model once for pre-compilation (otherwise affects reported timings)
    serve.start()
    ModelServer.deploy(erdModel, dir_TrainingResults+modelName)
    model = ModelServer.get_handle()
    if erdModel == 'DLADS': _ = ray.get(model.remote(np.empty((1,64,64,3))))
    
    #Run algorithm for each of the samples, timing and storing metric progression for each; (1 less CPU in pool for server deployment)
    if parallelization and len(sampleDataset)>1: 
        futures = [(sampleDataset[sampleNum], optimalC, model, percToScan, percToViz, False, False, lineVisitAll, liveOutputFlag, dir_Results, False, False, False) for sampleNum in range(0,len(sampleDataset))]
        p = Pool(numberCPUS-1)
        results = p.starmap(runSampling, futures)
        p.close()
        p.join()
    else: results = [runSampling(sampleDataset[sampleNum], optimalC, model, percToScan, percToViz, False, False, lineVisitAll, liveOutputFlag, dir_Results, False, False, False) for sampleNum in tqdm(range(0,len(sampleDataset)), desc='Samples', position=0, leave=True, ascii=True)]
    
    #Shutdown the server to ensure resources are returned to ray
    serve.shutdown()
    
    #Perform completion/visualization routines
    chanAvgPSNR_Results, avgPSNR_Results, sumImagePSNR_Results, ERDPSNR_Results = [], [], [], []
    chanAvgSSIM_Results, avgSSIM_Results, sumImageSSIM_Results, ERDSSIM_Results = [], [], [], []
    quantityMeasured_Results, timeResults, allERDTimes = [], [], []
    
    #Set the quantity measured metric for labeling the x axes
    xLabel = '% Measured'
    #elif scanMethod == 'linewise': xLabel = '% Lines Measured'
    
    for result in tqdm(results, desc='Visualization', position=0, leave=True, ascii=True):
    
        #Export metrics and perform final analysis for the result
        result.complete()
        
        #Add individual results to rolling lists for averaging
        timeResults.append(result.finalTime)
        allERDTimes.append(result.computeERDTimes)
        chanAvgPSNR_Results.append(result.chanAvgPSNRList)
        sumImagePSNR_Results.append(result.sumImagePSNRList)
        ERDPSNR_Results.append(result.ERDPSNRList)
        chanAvgSSIM_Results.append(result.chanAvgSSIMList)
        sumImageSSIM_Results.append(result.sumImageSSIMList)
        ERDSSIM_Results.append(result.ERDSSIMList)
        
        #Save individual results
        np.savetxt(result.dir_sampleResults+'PSNR_chanAvg.csv', np.transpose([result.percsMeasured, result.chanAvgPSNRList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'PSNR_sumImage.csv', np.transpose([result.percsMeasured, result.sumImagePSNRList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'PSNR_ERD.csv', np.transpose([result.percsMeasured, result.ERDPSNRList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_chanAvg.csv', np.transpose([result.percsMeasured, result.chanAvgSSIMList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_sumImage.csv', np.transpose([result.percsMeasured, result.sumImageSSIMList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_ERD.csv', np.transpose([result.percsMeasured, result.ERDSSIMList]), delimiter=',')
        
        #Save individual result plots
        basicPlot(result.percsMeasured, result.chanAvgPSNRList, result.dir_sampleResults+'PSNR_chanAvg'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        basicPlot(result.percsMeasured, result.sumImagePSNRList, result.dir_sampleResults+'PSNR_sumImage'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        basicPlot(result.percsMeasured, result.ERDPSNRList, result.dir_sampleResults+'PSNR_ERD'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        basicPlot(result.percsMeasured, result.chanAvgSSIMList, result.dir_sampleResults+'SSIM_chanAvg'+'.png', xLabel=xLabel, yLabel='Average SSIM')
        basicPlot(result.percsMeasured, result.sumImageSSIMList, result.dir_sampleResults+'SSIM_sumImage'+'.png', xLabel=xLabel, yLabel='Average SSIM')
        basicPlot(result.percsMeasured, result.ERDSSIMList, result.dir_sampleResults+'SSIM_ERD'+'.png', xLabel=xLabel, yLabel='Average SSIM')
            
        
        #If pointwise, then consider the percentage measured at each step; if linewise then consider the number of lines scanned
        #Since number of points on each line is different, this approach is not as descriptive as hoped and was therefore disabled.
        quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        #if scanMethod == 'pointwise': quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        #elif scanMethod == 'linewise': quantityMeasured_Results.append([(np.sum(np.sum(sample.mask, axis=1)>0)/sample.mask.shape[0])*100 for sample in result.samples])
        
    #Extract and average results at the specified precision
    quantityMeasured, chanAvgPSNR_Results_mean = percResults(chanAvgPSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, sumImagePSNR_Results_mean = percResults(sumImagePSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, ERDPSNR_Results_mean = percResults(ERDPSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, chanAvgSSIM_Results_mean = percResults(chanAvgSSIM_Results, quantityMeasured_Results, precision)
    quantityMeasured, sumImageSSIM_Results_mean = percResults(sumImageSSIM_Results, quantityMeasured_Results, precision)
    quantityMeasured, ERDSSIM_Results_mean = percResults(ERDSSIM_Results, quantityMeasured_Results, precision)
    
    #Compute area under the average PSNR and ERD curves
    chanAvgPSNR_AreaUnderCurve = np.trapz(chanAvgPSNR_Results_mean, quantityMeasured)
    chanAvgSSIM_AreaUnderCurve = np.trapz(chanAvgSSIM_Results_mean, quantityMeasured)
    ERDPSNR_AreaUnderCurve = np.trapz(ERDPSNR_Results_mean, quantityMeasured)
    
    #Save averaged results per quantity measured metric
    np.savetxt(dir_Results+'PSNR_chanAvg.csv', np.transpose([quantityMeasured, chanAvgPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'PSNR_sumImage.csv', np.transpose([quantityMeasured, sumImagePSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'PSNR_ERD.csv', np.transpose([quantityMeasured, ERDPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'SSIM_chanAvg.csv', np.transpose([quantityMeasured, chanAvgSSIM_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'SSIM_sumImage.csv', np.transpose([quantityMeasured, sumImageSSIM_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'SSIM_ERD.csv', np.transpose([quantityMeasured, ERDSSIM_Results_mean]), delimiter=',')

    #Export plots of averaged results
    basicPlot(quantityMeasured, chanAvgPSNR_Results_mean, dir_Results+'PSNR_chanAvg'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    basicPlot(quantityMeasured, sumImagePSNR_Results_mean, dir_Results+'PSNR_sumImage'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    basicPlot(quantityMeasured, ERDPSNR_Results_mean, dir_Results+'PSNR_ERD'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    basicPlot(quantityMeasured, chanAvgSSIM_Results_mean, dir_Results+'SSIM_chanAvg'+'.png', xLabel=xLabel, yLabel='Average SSIM')
    basicPlot(quantityMeasured, sumImageSSIM_Results_mean, dir_Results+'SSIM_sumImage'+'.png', xLabel=xLabel, yLabel='Average SSIM')
    basicPlot(quantityMeasured, ERDSSIM_Results_mean, dir_Results+'SSIM_ERD'+'.png', xLabel=xLabel, yLabel='Average SSIM')

    #Find the final results for each image
    lastQuantityMeasured = [quantityMeasured_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastchanAvgPSNR = [chanAvgPSNR_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastSumImagePSNR = [sumImagePSNR_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastERDPSNR = [ERDPSNR_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastchanAvgSSIM = [chanAvgSSIM_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastSumImageSSIM = [sumImageSSIM_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastERDSSIM = [ERDSSIM_Results[i][-1] for i in range(0, len(sampleDataset))]
    allERDTimes = np.concatenate(allERDTimes)
    
    #Print out final results 
    dataPrintout = [['','Average', '', 'Standard Deviation']]
    dataPrintout.append(['Final %:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    #elif scanMethod == 'linewise': dataPrintout.append(['Final % of Lines:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    dataPrintout.append(['Channel PSNR:', np.mean(lastchanAvgPSNR), '+/-', np.std(lastchanAvgPSNR)])
    dataPrintout.append(['Channel PSNR Area Under Curve:', chanAvgPSNR_AreaUnderCurve])
    dataPrintout.append(['Sum Image PSNR:', np.mean(lastSumImagePSNR), '+/-', np.std(lastSumImagePSNR)])
    dataPrintout.append(['Final ERD PSNR:', np.mean(lastERDPSNR), '+/-', np.std(lastERDPSNR)])
    dataPrintout.append(['ERD PSNR Area Under Curve:', ERDPSNR_AreaUnderCurve])
    dataPrintout.append([])
    dataPrintout.append(['Channel SSIM:', np.mean(lastchanAvgSSIM), '+/-', np.std(lastchanAvgSSIM)])
    dataPrintout.append(['Channel SSIM Area Under Curve:', chanAvgSSIM_AreaUnderCurve])
    dataPrintout.append(['Sum Image SSIM:', np.mean(lastSumImageSSIM), '+/-', np.std(lastSumImageSSIM)])
    dataPrintout.append(['Final ERD SSIM:', np.mean(lastERDSSIM), '+/-', np.std(lastERDSSIM)])
    dataPrintout.append([])
    dataPrintout.append(['ERD Compute Time (s):', np.mean(allERDTimes), '+/-', np.std(allERDTimes)])
    dataPrintout.append(['Run Time (s):', np.mean(timeResults), '+/-', np.std(timeResults)])
    pd.DataFrame(dataPrintout).to_csv(dir_Results + 'dataPrintout.csv')
