#==================================================================
#TESTING SLADS SPECIFIC
#==================================================================

#Given a set of sample paths, perform simulations using a trained SLADS Model
def simulateSLADS(sortedSampleFolders, dir_Results, optimalC):

    #If consistentcy in the random generator is desired for comparisons, then reset seed
    if consistentSeed: 
        tf.random.set_seed(0)
        np.random.seed(0)
        random.seed(0)

    #Do not run initial samplData generation in parallel, changes mask initialization
    sampleData = [SampleData(sampleFolder, initialPercToScan, stopPerc, scanMethod, RDMethod, mzGlobalSpec, True, lineRevist, True) for sampleFolder in tqdm(sortedSampleFolders, desc='Reading', leave=True, ascii=True)]
    
    #Run algorithm for each of the samples, timing and storing metric progression for each
    if parallelization: 
        futures = [runSLADS_parhelper.remote(sampleData[sampleNum], optimalC, True, percToScan, percToViz, False, False, lineVisitAll, False) for sampleNum in range(0,len(sampleData))]
        results = ray.get(futures)
    else: results = [runSLADS(sampleData[sampleNum], optimalC, True, percToScan, percToViz, False, False, lineVisitAll, False) for sampleNum in tqdm(range(0,len(sampleData)), desc='Samples', position=0, leave=True, ascii=True)]
    
    #Perform completion/visualization routines
    mzAvgPSNR_Results, avgPSNR_Results, ERDPSNR_Results = [], [], []
    mzAvgSSIM_Results, avgSSIM_Results, ERDSSIM_Results = [], [], []
    perc_Results, time_Results = [], []
    for result in tqdm(results, desc='Visualization', position=0, leave=True, ascii=True):
        result.complete(dir_Results)
        time_Results.append(result.finalTime)
        mzAvgPSNR_Results.append(result.mzAvgPSNRList)
        avgPSNR_Results.append(result.avgPSNRList)
        ERDPSNR_Results.append(result.ERDPSNRList)
        mzAvgSSIM_Results.append(result.mzAvgSSIMList)
        avgSSIM_Results.append(result.avgSSIMList)
        ERDSSIM_Results.append(result.ERDSSIMList)
        perc_Results.append([sample.percMeasured for sample in result.samples])
        
    #Extract percentage results at the specified precision
    percents, mzAvgPSNR_Results_mean = percResults(mzAvgPSNR_Results, perc_Results, precision)
    percents, avgPSNR_Results_mean = percResults(avgPSNR_Results, perc_Results, precision)
    percents, ERDPSNR_Results_mean = percResults(ERDPSNR_Results, perc_Results, precision)
    percents, mzAvgSSIM_Results_mean = percResults(mzAvgSSIM_Results, perc_Results, precision)
    percents, avgSSIM_Results_mean = percResults(avgSSIM_Results, perc_Results, precision)
    percents, ERDSSIM_Results_mean = percResults(ERDSSIM_Results, perc_Results, precision)
    
    #Save average results per percentage data
    np.savetxt(dir_Results+'mzAvgPSNR_Percentage.csv', np.transpose([percents, mzAvgPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'avgPSNR_Percentage.csv', np.transpose([percents, avgPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'ERDPSNR_Percentage.csv', np.transpose([percents, ERDPSNR_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'mzAvgSSIM_Percentage.csv', np.transpose([percents, mzAvgSSIM_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'avgPSNR_Percentage.csv', np.transpose([percents, avgSSIM_Results_mean]), delimiter=',')
    np.savetxt(dir_Results+'ERDSSIM_Percentage.csv', np.transpose([percents, ERDSSIM_Results_mean]), delimiter=',')

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, mzAvgPSNR_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    ax1.set_ylabel('Average mz PSNR (dB)')
    plt.savefig(dir_Results + 'mzAvgPSNR_Percentage' + '.png')
    plt.close()

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, avgPSNR_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    ax1.set_ylabel('Average PSNR (dB)')
    plt.savefig(dir_Results + 'avgPSNR_Percentage' + '.png')
    plt.close()
    
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, ERDPSNR_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    ax1.set_ylabel('Average PSNR (dB)')
    plt.savefig(dir_Results + 'ERDPSNR_Percentage' + '.png')
    plt.close()
    
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, mzAvgSSIM_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    ax1.set_ylabel('Average mz SSIM')
    plt.savefig(dir_Results + 'mzAvgSSIM_Percentage' + '.png')
    plt.close()

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, avgSSIM_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    ax1.set_ylabel('Average SSIM')
    plt.savefig(dir_Results + 'avgSSIM_Percentage' + '.png')
    plt.close()
    
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, ERDSSIM_Results_mean, color='black')
    ax1.set_xlabel('% Measured')
    ax1.set_ylabel('Average SSIM')
    plt.savefig(dir_Results + 'ERDSSIM_Percentage' + '.png')
    plt.close()
    
    #Find the final results for each image
    lastPercMeasured = [perc_Results[i][-1] for i in range(0, len(sampleData))]
    lastmzAvgPSNR = [mzAvgPSNR_Results[i][-1] for i in range(0, len(sampleData))]
    lastAvgPSNR = [avgPSNR_Results[i][-1] for i in range(0, len(sampleData))]
    lastERDPSNR = [ERDPSNR_Results[i][-1] for i in range(0, len(sampleData))]
    lastmzAvgSSIM = [mzAvgSSIM_Results[i][-1] for i in range(0, len(sampleData))]
    lastAvgSSIM = [avgSSIM_Results[i][-1] for i in range(0, len(sampleData))]
    lastERDSSIM = [ERDSSIM_Results[i][-1] for i in range(0, len(sampleData))]
    lastTime = [time_Results[i] for i in range(0, len(sampleData))]
    
    #Printout final results 
    dataPrintout = []
    dataPrintout.append(['Average Final %:', np.mean(lastPercMeasured), '+/-', np.std(lastPercMeasured)])
    dataPrintout.append(['Average mz PSNR:', np.mean(lastmzAvgPSNR), '+/-', np.std(lastmzAvgPSNR)])
    dataPrintout.append(['Average PSNR:', np.mean(lastAvgPSNR), '+/-', np.std(lastAvgPSNR)])
    dataPrintout.append(['Average ERD PSNR:', np.mean(lastERDPSNR), '+/-', np.std(lastERDPSNR)])
    dataPrintout.append(['Average mz SSIM:', np.mean(lastmzAvgSSIM), '+/-', np.std(lastmzAvgSSIM)])
    dataPrintout.append(['Average SSIM:', np.mean(lastAvgSSIM), '+/-', np.std(lastAvgSSIM)])
    dataPrintout.append(['Average ERD SSIM:', np.mean(lastERDSSIM), '+/-', np.std(lastERDSSIM)])
    dataPrintout.append(['Average Time:', np.mean(lastTime), '+/-', np.std(lastTime)])
    pd.DataFrame(dataPrintout).to_csv(dir_Results + 'dataPrintout.csv')
