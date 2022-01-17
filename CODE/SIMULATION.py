#==================================================================
#TESTING SLADS SPECIFIC
#==================================================================

#import tempfile
       
#Given a set of sample paths, perform simulations using a trained SLADS Model
def simulateSLADS(sortedSampleFolders, dir_Results, model, optimalC):

    #If consistency in the random generator is desired for comparisons, then reset seed
    if consistentSeed: 
        tf.random.set_seed(0)
        np.random.seed(0)
        random.seed(0)
    
    #If existing testing database should be used, try loading it (adjust its internal configuration variables) and if it doesn't exist then generate a new one anyway
    if loadTestDataset and os.path.exists(dir_TrainingResults + 'testingDatabase.p'): 
        sampleDataset = pickle.load(open(dir_TrainingResults + 'testingDatabase.p', "rb" ))
        for sampleData in sampleDataset: 
            sampleData.scanMethod = scanMethod
            sampleData.initialPercToScan = initialPercToScan
            sampleData.stopPerc = stopPerc
            sampleData.lineRevist = lineRevist
            sampleData.generateInitialSets(scanMethod)
            sampleData.readScanData()
    else:
        sampleDataset = [SampleData(sampleFolder, initialPercToScan, stopPerc, scanMethod, RDMethod, True, lineRevist, True) for sampleFolder in tqdm(sortedSampleFolders, desc='Reading', leave=True, ascii=True)]
        pickle.dump(sampleDataset, open(dir_TrainingResults + 'testingDatabase.p', 'wb'))

    #Run algorithm for each of the samples, timing and storing metric progression for each
    if parallelization: 
        futures = [runSLADS_parhelper.remote(sampleDataset[sampleNum], optimalC, model, percToScan, percToViz, False, False, lineVisitAll, liveOutputFlag, dir_Results, False) for sampleNum in range(0,len(sampleDataset))]
        results = ray.get(futures)
    else: results = [runSLADS(sampleDataset[sampleNum], optimalC, model, percToScan, percToViz, False, False, lineVisitAll, liveOutputFlag, dir_Results, False) for sampleNum in tqdm(range(0,len(sampleDataset)), desc='Samples', position=0, leave=True, ascii=True)]
    
    #Perform completion/visualization routines
    mzAvgPSNR_Results, avgPSNR_Results, ERDPSNR_Results = [], [], []
    mzAvgSSIM_Results, avgSSIM_Results, ERDSSIM_Results = [], [], []
    quantityMeasured_Results, time_Results = [], []
    for result in tqdm(results, desc='Visualization', position=0, leave=True, ascii=True):
        result.complete()
        time_Results.append(result.finalTime)
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
    lastTime = [time_Results[i] for i in range(0, len(sampleDataset))]
    
    #Printout final results 
    dataPrintout = []
    dataPrintout.append(['Average'])
    dataPrintout.append(['Final %:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    #if scanMethod == 'pointwise': dataPrintout.append(['Final %:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    #elif scanMethod == 'linewise': dataPrintout.append(['Final % of Lines:', np.mean(lastQuantityMeasured), '+/-', np.std(lastQuantityMeasured)])
    dataPrintout.append(['mz PSNR:', np.mean(lastmzAvgPSNR), '+/-', np.std(lastmzAvgPSNR)])
    dataPrintout.append(['mz PSNR Area Under Curve:', mzAvgPSNR_AreaUnderCurve])
    dataPrintout.append(['ERD PSNR:', np.mean(lastERDPSNR), '+/-', np.std(lastERDPSNR)])
    dataPrintout.append(['ERD Area Under Curve:', ERDPSNR_AreaUnderCurve])
    dataPrintout.append([''])
    dataPrintout.append(['PSNR:', np.mean(lastAvgPSNR), '+/-', np.std(lastAvgPSNR)])
    dataPrintout.append(['mz SSIM:', np.mean(lastmzAvgSSIM), '+/-', np.std(lastmzAvgSSIM)])
    dataPrintout.append(['SSIM:', np.mean(lastAvgSSIM), '+/-', np.std(lastAvgSSIM)])
    dataPrintout.append(['ERD SSIM:', np.mean(lastERDSSIM), '+/-', np.std(lastERDSSIM)])
    dataPrintout.append(['Time:', np.mean(lastTime), '+/-', np.std(lastTime)])
    pd.DataFrame(dataPrintout).to_csv(dir_Results + 'dataPrintout.csv')
