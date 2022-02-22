#==================================================================
#TESTING SLADS SPECIFIC
#==================================================================

#Given a set of sample paths, perform simulations using a trained SLADS Model
def simulateSLADS(sortedSampleFolders, dir_Results, optimalC):

    #Start server, deploy, and get handle for model queries; run Tensorflow model once for pre-compilation (otherwise affects reported timings)
    serve.start()

    if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': ModelServer.deploy(erdModel, dir_TrainingResults+'model_cValue_'+str(optimalC)+'.npy')
    elif erdModel == 'DLADS': ModelServer.deploy(erdModel, dir_TrainingResults+'model_cValue_'+str(optimalC))
    model = ModelServer.get_handle()
    if erdModel == 'DLADS': _ = ray.get(model.remote(np.empty((1,64,64,3)))).copy()
    
    #If consistency in the random generator is desired for comparisons, then reset seed
    if consistentSeed: 
        tf.random.set_seed(0)
        np.random.seed(0)
        random.seed(0)
    
    sampleDataset = [SampleData(sampleFolder, initialPercToScan, stopPerc, scanMethod, True, lineRevist, True) for sampleFolder in tqdm(sortedSampleFolders, desc='Reading', leave=True, ascii=True)]

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
    mzAvgPSNR_Results, avgPSNR_Results, ERDPSNR_Results = [], [], []
    mzAvgSSIM_Results, avgSSIM_Results, ERDSSIM_Results = [], [], []
    quantityMeasured_Results, timeResults, allERDTimes = [], [], []
    for result in tqdm(results, desc='Visualization', position=0, leave=True, ascii=True):
        result.complete()
        timeResults.append(result.finalTime)
        allERDTimes.append(result.computeERDTimes)
        mzAvgPSNR_Results.append(result.mzAvgPSNRList)
        avgPSNR_Results.append(result.avgPSNRList)
        ERDPSNR_Results.append(result.ERDPSNRList)
        mzAvgSSIM_Results.append(result.mzAvgSSIMList)
        avgSSIM_Results.append(result.avgSSIMList)
        ERDSSIM_Results.append(result.ERDSSIMList)
        
        #If pointwise, then consider the percentage measured at each step; if linewise then consider the number of lines scanned
        quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        #if scanMethod == 'pointwise': quantityMeasured_Results.append([sample.percMeasured for sample in result.samples])
        #elif scanMethod == 'linewise': quantityMeasured_Results.append([(np.sum(np.sum(sample.mask, axis=1)>0)/sample.mask.shape[0])*100 for sample in result.samples])
        
    #Extract and average results at the specified precision
    quantityMeasured, mzAvgPSNR_Results_mean = percResults(mzAvgPSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, avgPSNR_Results_mean = percResults(avgPSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, ERDPSNR_Results_mean = percResults(ERDPSNR_Results, quantityMeasured_Results, precision)
    quantityMeasured, mzAvgSSIM_Results_mean = percResults(mzAvgSSIM_Results, quantityMeasured_Results, precision)
    quantityMeasured, avgSSIM_Results_mean = percResults(avgSSIM_Results, quantityMeasured_Results, precision)
    quantityMeasured, ERDSSIM_Results_mean = percResults(ERDSSIM_Results, quantityMeasured_Results, precision)
    
    #Compute area under the average PSNR and ERD curves
    mzAvgPSNR_AreaUnderCurve = np.trapz(mzAvgPSNR_Results_mean, quantityMeasured)
    ERDPSNR_AreaUnderCurve = np.trapz(ERDPSNR_Results_mean, quantityMeasured)
    
    #Save average results per percentage data
    np.savetxt(dir_Results+'mzAvgPSNR.csv', np.transpose([quantityMeasured, mzAvgPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'avgPSNR.csv', np.transpose([quantityMeasured, avgPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'ERDPSNR.csv', np.transpose([quantityMeasured, ERDPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'mzAvgSSIM.csv', np.transpose([quantityMeasured, mzAvgSSIM_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'avgPSNR.csv', np.transpose([quantityMeasured, avgSSIM_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'ERDSSIM.csv', np.transpose([quantityMeasured, ERDSSIM_Results_mean]), delimiter=',')

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(quantityMeasured, mzAvgPSNR_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    #if scanMethod == 'pointwise': ax1.set_xlabel('% Measured')
    #elif scanMethod == 'linewise': ax1.set_xlabel('% Lines Measured')
    ax1.set_ylabel('Average mz PSNR (dB)')
    plt.savefig(dir_Results + 'mzAvgPSNR_Percentage' + '.png')
    plt.close()

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(quantityMeasured, avgPSNR_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    #if scanMethod == 'pointwise': ax1.set_xlabel('% Measured')
    #elif scanMethod == 'linewise': ax1.set_xlabel('% Lines Measured')
    ax1.set_ylabel('Average PSNR (dB)')
    plt.savefig(dir_Results + 'avgPSNR_Percentage' + '.png')
    plt.close()
    
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(quantityMeasured, ERDPSNR_Results_mean, color='black')
    if scanMethod == 'pointwise': ax1.set_xlabel('% Measured')
    elif scanMethod == 'linewise': ax1.set_xlabel('% Lines Measured')
    ax1.set_ylabel('Average PSNR (dB)')
    plt.savefig(dir_Results + 'ERDPSNR_Percentage' + '.png')
    plt.close()
    
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(quantityMeasured, mzAvgSSIM_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    #if scanMethod == 'pointwise': ax1.set_xlabel('% Measured')
    #elif scanMethod == 'linewise': ax1.set_xlabel('% Lines Measured')
    ax1.set_ylabel('Average mz SSIM')
    plt.savefig(dir_Results + 'mzAvgSSIM_Percentage' + '.png')
    plt.close()

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(quantityMeasured, avgSSIM_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    #if scanMethod == 'pointwise': ax1.set_xlabel('% Measured')
    #elif scanMethod == 'linewise': ax1.set_xlabel('% Lines Measured')
    ax1.set_ylabel('Average SSIM')
    plt.savefig(dir_Results + 'avgSSIM_Percentage' + '.png')
    plt.close()
    
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(quantityMeasured, ERDSSIM_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    #if scanMethod == 'pointwise': ax1.set_xlabel('% Measured')
    #elif scanMethod == 'linewise': ax1.set_xlabel('% Lines Measured')
    ax1.set_ylabel('Average SSIM')
    plt.savefig(dir_Results + 'ERDSSIM_Percentage' + '.png')
    plt.close()
    
    #Find the final results for each image
    lastQuantityMeasured = [quantityMeasured_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastmzAvgPSNR = [mzAvgPSNR_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastAvgPSNR = [avgPSNR_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastERDPSNR = [ERDPSNR_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastmzAvgSSIM = [mzAvgSSIM_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastAvgSSIM = [avgSSIM_Results[i][-1] for i in range(0, len(sampleDataset))]
    lastERDSSIM = [ERDSSIM_Results[i][-1] for i in range(0, len(sampleDataset))]
    allERDTimes = np.concatenate(allERDTimes)
    
    #Printout final results 
    dataPrintout = [['','Average', '', 'Standard Deviation']]
    dataPrintout.append(['Final %:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    #if scanMethod == 'pointwise': dataPrintout.append(['Final %:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    #elif scanMethod == 'linewise': dataPrintout.append(['Final % of Lines:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    dataPrintout.append(['mz PSNR:', np.mean(lastmzAvgPSNR), '+/-', np.std(lastmzAvgPSNR)])
    dataPrintout.append(['mz PSNR Area Under Curve:', mzAvgPSNR_AreaUnderCurve])
    dataPrintout.append(['Final ERD PSNR:', np.mean(lastERDPSNR), '+/-', np.std(lastERDPSNR)])
    dataPrintout.append(['ERD PSNR Area Under Curve:', ERDPSNR_AreaUnderCurve])
    dataPrintout.append([''])
    dataPrintout.append(['mz SSIM:', np.mean(lastmzAvgSSIM), '+/-', np.std(lastmzAvgSSIM)])
    dataPrintout.append(['SSIM:', np.mean(lastAvgSSIM), '+/-', np.std(lastAvgSSIM)])
    dataPrintout.append(['Final ERD SSIM:', np.mean(lastERDSSIM), '+/-', np.std(lastERDSSIM)])
    dataPrintout.append([''])
    dataPrintout.append(['ERD Compute Time (s):', np.mean(allERDTimes), '+/-', np.std(allERDTimes)])
    dataPrintout.append(['Run Time (s):', np.mean(timeResults), '+/-', np.std(timeResults)])
    pd.DataFrame(dataPrintout).to_csv(dir_Results + 'dataPrintout.csv')
