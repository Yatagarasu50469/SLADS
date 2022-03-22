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
    sampleDataset = [SampleData(sampleFolder, initialPercToScan, stopPerc, scanMethod, lineRevist, False) for sampleFolder in tqdm(sortedSampleFolders, desc='Reading', leave=True, ascii=True)]

    #Start server, deploy, and get handle for model queries; run Tensorflow model once for pre-compilation (otherwise affects reported timings)
    serve.start()
    ModelServer.deploy(erdModel, dir_TrainingResults+modelName)
    model = ModelServer.get_handle()
    if erdModel == 'DLADS': _ = ray.get(model.remote(np.empty((1,64,64,3))))
    
    #Run algorithm for each of the samples, timing and storing metric progression for each; (1 less CPU in pool for server deployment)
    if parallelization: 
        futures = [(sampleDataset[sampleNum], optimalC, model, percToScan, percToViz, False, False, lineVisitAll, liveOutputFlag, dir_Results, False, False, False) for sampleNum in range(0,len(sampleDataset))]
        p = Pool(numberCPUS-1)
        results = p.starmap(runSampling, futures)
        p.close()
        p.join()
    else: results = [runSampling(sampleDataset[sampleNum], optimalC, model, percToScan, percToViz, False, False, lineVisitAll, liveOutputFlag, dir_Results, False, False, False) for sampleNum in tqdm(range(0,len(sampleDataset)), desc='Samples', position=0, leave=True, ascii=True)]
    
    #Shutdown the server to ensure resources are returned to ray
    serve.shutdown()
    
    #Perform completion/visualization routines
    mzAvgPSNR_Results, avgPSNR_Results, TICPSNR_Results, ERDPSNR_Results = [], [], [], []
    mzAvgSSIM_Results, avgSSIM_Results, TICSSIM_Results, ERDSSIM_Results = [], [], [], []
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
        mzAvgPSNR_Results.append(result.mzAvgPSNRList)
        #avgPSNR_Results.append(result.avgPSNRList)
        TICPSNR_Results.append(result.TICPSNRList)
        ERDPSNR_Results.append(result.ERDPSNRList)
        mzAvgSSIM_Results.append(result.mzAvgSSIMList)
        #avgSSIM_Results.append(result.avgSSIMList)
        TICSSIM_Results.append(result.TICSSIMList)
        ERDSSIM_Results.append(result.ERDSSIMList)
        
        #Save individual results
        np.savetxt(result.dir_sampleResults+'PSNR_mzAvg.csv', np.transpose([result.percsMeasured, result.mzAvgPSNRList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'PSNR_TIC.csv', np.transpose([result.percsMeasured, result.TICPSNRList]), delimiter=',')
        #np.savetxt(result.dir_sampleResults+'PSNR_avgmz.csv', np.transpose([result.percsMeasured, result.avgPSNRList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'PSNR_ERD.csv', np.transpose([result.percsMeasured, result.ERDPSNRList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_mzAvg.csv', np.transpose([result.percsMeasured, result.mzAvgSSIMList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_TIC.csv', np.transpose([result.percsMeasured, result.TICSSIMList]), delimiter=',')
        #np.savetxt(result.dir_sampleResults+'SSIM_avgmz.csv', np.transpose([result.percsMeasured, result.avgSSIMList]), delimiter=',')
        np.savetxt(result.dir_sampleResults+'SSIM_ERD.csv', np.transpose([result.percsMeasured, result.ERDSSIMList]), delimiter=',')
        
        #Save individual result plots
        basicPlot(result.percsMeasured, result.mzAvgPSNRList, result.dir_sampleResults+'PSNR_mzAvg'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        basicPlot(result.percsMeasured, result.TICPSNRList, result.dir_sampleResults+'PSNR_TIC'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        #basicPlot(result.percsMeasured, result.avgPSNRList, result.dir_sampleResults+'PSNR_avgmz'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        basicPlot(result.percsMeasured, result.ERDPSNRList, result.dir_sampleResults+'PSNR_ERD'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
        basicPlot(result.percsMeasured, result.mzAvgSSIMList, result.dir_sampleResults+'SSIM_mzAvg'+'.png', xLabel=xLabel, yLabel='Average SSIM')
        basicPlot(result.percsMeasured, result.TICSSIMList, result.dir_sampleResults+'SSIM_TIC'+'.png', xLabel=xLabel, yLabel='Average SSIM')
        #basicPlot(result.percsMeasured, result.avgSSIMList, result.dir_sampleResults+'SSIM_avgmz'+'.png', xLabel=xLabel, yLabel='Average SSIM')
        basicPlot(result.percsMeasured, result.ERDSSIMList, result.dir_sampleResults+'SSIM_ERD'+'.png', xLabel=xLabel, yLabel='Average SSIM')
            
        
        #If pointwise, then consider the percentage measured at each step; if linewise then consider the number of lines scanned
        #Since number of points on each line is different, this approach is not as descriptive as hoped and was therefore disabled.
        quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        #if scanMethod == 'pointwise': quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        #elif scanMethod == 'linewise': quantityMeasured_Results.append([(np.sum(np.sum(sample.mask, axis=1)>0)/sample.mask.shape[0])*100 for sample in result.samples])
        
    #Extract and average results at the specified precision
    quantityMeasured, mzAvgPSNR_Results_mean = percResults(mzAvgPSNR_Results, quantityMeasured_Results, precision)
    #quantityMeasured, avgPSNR_Results_mean = percResults(avgPSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, TICPSNR_Results_mean = percResults(TICPSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, ERDPSNR_Results_mean = percResults(ERDPSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, mzAvgSSIM_Results_mean = percResults(mzAvgSSIM_Results, quantityMeasured_Results, precision)
    #quantityMeasured, avgSSIM_Results_mean = percResults(avgSSIM_Results, quantityMeasured_Results, precision)
    quantityMeasured, TICSSIM_Results_mean = percResults(TICSSIM_Results, quantityMeasured_Results, precision)
    quantityMeasured, ERDSSIM_Results_mean = percResults(ERDSSIM_Results, quantityMeasured_Results, precision)
    
    #Compute area under the average PSNR and ERD curves
    mzAvgPSNR_AreaUnderCurve = np.trapz(mzAvgPSNR_Results_mean, quantityMeasured)
    mzAvgSSIM_AreaUnderCurve = np.trapz(mzAvgSSIM_Results_mean, quantityMeasured)
    ERDPSNR_AreaUnderCurve = np.trapz(ERDPSNR_Results_mean, quantityMeasured)
    
    #Save averaged results per quantity measured metric
    np.savetxt(dir_Results+'PSNR_mzAvg.csv', np.transpose([quantityMeasured, mzAvgPSNR_Results_mean]), delimiter=',')
    #np.savetxt(dir_Results+'PSNR_avgmz.csv', np.transpose([quantityMeasured, avgPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'PSNR_TIC.csv', np.transpose([quantityMeasured, TICPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'PSNR_ERD.csv', np.transpose([quantityMeasured, ERDPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'SSIM_mzAvg.csv', np.transpose([quantityMeasured, mzAvgSSIM_Results_mean]), delimiter=',')
    #np.savetxt(dir_Results+'SSIM_avgmz.csv', np.transpose([quantityMeasured, avgSSIM_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'SSIM_TIC.csv', np.transpose([quantityMeasured, TICSSIM_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'SSIM_ERD.csv', np.transpose([quantityMeasured, ERDSSIM_Results_mean]), delimiter=',')

    #Export plots of averaged results
    basicPlot(quantityMeasured, mzAvgPSNR_Results_mean, dir_Results+'PSNR_mzAvg'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    basicPlot(quantityMeasured, TICPSNR_Results_mean, dir_Results+'PSNR_TIC'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    #basicPlot(quantityMeasured, avgPSNR_Results_mean, dir_Results+'PSNR_avgmz'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    basicPlot(quantityMeasured, ERDPSNR_Results_mean, dir_Results+'PSNR_ERD'+'.png', xLabel=xLabel, yLabel='Average PSNR (dB)')
    basicPlot(quantityMeasured, mzAvgSSIM_Results_mean, dir_Results+'SSIM_mzAvg'+'.png', xLabel=xLabel, yLabel='Average SSIM')
    basicPlot(quantityMeasured, TICSSIM_Results_mean, dir_Results+'SSIM_TIC'+'.png', xLabel=xLabel, yLabel='Average SSIM')
    #basicPlot(quantityMeasured, avgSSIM_Results_mean, dir_Results+'SSIM_avgmz'+'.png', xLabel=xLabel, yLabel='Average SSIM')
    basicPlot(quantityMeasured, ERDSSIM_Results_mean, dir_Results+'SSIM_ERD'+'.png', xLabel=xLabel, yLabel='Average SSIM')

    #Find the final results for each image
    lastQuantityMeasured = [quantityMeasured_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastmzAvgPSNR = [mzAvgPSNR_Results[i][-1] for i in range(0, len(sampleDataset))]
    #lastAvgPSNR = [avgPSNR_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastTICPSNR = [TICPSNR_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastERDPSNR = [ERDPSNR_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastmzAvgSSIM = [mzAvgSSIM_Results[i][-1] for i in range(0, len(sampleDataset))]
    #lastAvgSSIM = [avgSSIM_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastTICSSIM = [TICSSIM_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastERDSSIM = [ERDSSIM_Results[i][-1] for i in range(0, len(sampleDataset))]
    allERDTimes = np.concatenate(allERDTimes)
    
    #Print out final results 
    dataPrintout = [['','Average', '', 'Standard Deviation']]
    dataPrintout.append(['Final %:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    #elif scanMethod == 'linewise': dataPrintout.append(['Final % of Lines:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    dataPrintout.append(['mz PSNR:', np.mean(lastmzAvgPSNR), '+/-', np.std(lastmzAvgPSNR)])
    dataPrintout.append(['mz PSNR Area Under Curve:', mzAvgPSNR_AreaUnderCurve])
    dataPrintout.append(['TIC PSNR:', np.mean(lastTICPSNR), '+/-', np.std(lastTICPSNR)])
    dataPrintout.append(['Final ERD PSNR:', np.mean(lastERDPSNR), '+/-', np.std(lastERDPSNR)])
    dataPrintout.append(['ERD PSNR Area Under Curve:', ERDPSNR_AreaUnderCurve])
    dataPrintout.append([])
    dataPrintout.append(['mz SSIM:', np.mean(lastmzAvgSSIM), '+/-', np.std(lastmzAvgSSIM)])
    dataPrintout.append(['mz SSIM Area Under Curve:', mzAvgSSIM_AreaUnderCurve])
    dataPrintout.append(['TIC SSIM:', np.mean(lastTICSSIM), '+/-', np.std(lastTICSSIM)])
    dataPrintout.append(['Final ERD SSIM:', np.mean(lastERDSSIM), '+/-', np.std(lastERDSSIM)])
    dataPrintout.append([])
    dataPrintout.append(['ERD Compute Time (s):', np.mean(allERDTimes), '+/-', np.std(allERDTimes)])
    dataPrintout.append(['Run Time (s):', np.mean(timeResults), '+/-', np.std(timeResults)])
    pd.DataFrame(dataPrintout).to_csv(dir_Results + 'dataPrintout.csv')
