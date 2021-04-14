#==================================================================
#TESTING SLADS SPECIFIC
#==================================================================

#Given a set of sample paths, perform testing using a trained SLADS Model
def testSLADS(sortedTestingSampleFolders, model, optimalC):

    #If consistentcy in the random generator is desired for comparisons, then reset seed
    if consistentSeed: np.random.seed(0)
    
    #Setup testing samples
    testingSamples = []
    for testingSampleFolder in tqdm(sortedTestingSampleFolders, desc='Reading Samples', leave=True, ascii=True):
        dataSampleName = os.path.basename(testingSampleFolder)
        
        #Read all available scan data into a sample object
        sample = Sample(testingSampleFolder, initialPercToScan, scanMethod, True, ignoreMissingLines=True)
        sample.readScanData(lineRevistMethod)

        #Indicate where resulting data should be stored
        sample.resultsPath = dir_TestingResults

        #Define information as a new Sample object
        testingSamples.append(copy.deepcopy(sample))

    #Run algorithm for each of the testing samples, timing and storing metric progression for each
    time_testingResults = []
    results =[]
    for sampleNum in tqdm(range(0,len(testingSamples)), desc='Testing Samples', position=0, leave=True, ascii=True):
        t0 = time.time()
        results.append(runSLADS(testingSamples[sampleNum], model, scanMethod, optimalC, percToScan, percToViz, stopPerc, simulationFlag=True, trainPlotFlag=False, animationFlag=animationGen, tqdmHide=False, oracleFlag=False, bestCFlag=False))
        time_testingResults.append(time.time()-t0)
    
    #Perform completion/visualization routines
    mzAvgPSNR_testingResults = []
    avgPSNR_testingResults = []
    ERDPSNR_testingResults = []
    perc_testingResults = []
    for result in tqdm(results, desc='Visualization', position=0, leave=True, ascii=True):
        result.complete(optimalC)
        mzAvgPSNR_testingResults.append(result.mzAvgPSNRList)
        avgPSNR_testingResults.append(result.avgPSNRList)
        ERDPSNR_testingResults.append(result.ERDPSNRList)
        perc_testingResults.append(result.percMeasuredList)
        
    #Extract percentage results at the specified precision
    percents, mzAvgPSNR_testingResults_mean = percResults(mzAvgPSNR_testingResults, perc_testingResults, precision)
    percents, avgPSNR_testingResults_mean = percResults(avgPSNR_testingResults, perc_testingResults, precision)
    percents, ERDPSNR_testingResults_mean = percResults(ERDPSNR_testingResults, perc_testingResults, precision)
    
    #Save average results per percentage data
    np.savetxt(dir_TestingResults+'mzAvgPSNR_Percentage.csv', np.transpose([percents, mzAvgPSNR_testingResults_mean]), delimiter=',')
    np.savetxt(dir_TestingResults+'avgPSNR_Percentage.csv', np.transpose([percents, avgPSNR_testingResults_mean]), delimiter=',')
    np.savetxt(dir_TestingResults+'ERDPSNR_Percentage.csv', np.transpose([percents, ERDPSNR_testingResults_mean]), delimiter=',')

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, mzAvgPSNR_testingResults_mean, color='black')
    #ax1.set_ylim([0, 50])
    ax1.set_xlabel('% Measured')
    ax1.set_ylabel('Average mz PSNR')
    plt.savefig(dir_TestingResults + 'mzAvgPSNR_Percentage' + '.png')
    plt.close()

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, avgPSNR_testingResults_mean, color='black')
    #ax1.set_ylim([0, 50])
    ax1.set_xlabel('% Measured')
    ax1.set_ylabel('Average PSNR')
    plt.savefig(dir_TestingResults + 'avgPSNR_Percentage' + '.png')
    plt.close()
    
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, ERDPSNR_testingResults_mean, color='black')
    #ax1.set_ylim([0, 50])    
    ax1.set_xlabel('% Measured')
    ax1.set_ylabel('Average PSNR')
    plt.savefig(dir_TestingResults + 'ERDPSNR_Percentage' + '.png')
    plt.close()
    
    #Find the final results for each image
    lastPercMeasured = [perc_testingResults[i][-1] for i in range(0, len(testingSamples))]
    lastmzAvgPSNR = [mzAvgPSNR_testingResults[i][-1] for i in range(0, len(testingSamples))]
    lastAvgPSNR = [avgPSNR_testingResults[i][-1] for i in range(0, len(testingSamples))]
    lastERDPSNR = [ERDPSNR_testingResults[i][-1] for i in range(0, len(testingSamples))]
    lastTime = [time_testingResults[i] for i in range(0, len(testingSamples))]
    
    #Printout final results 
    dataPrintout = []
    dataPrintout.append(['Average Final %:', np.mean(lastPercMeasured), '+/-', np.std(lastPercMeasured)])
    dataPrintout.append(['Average mz PSNR:', np.mean(lastmzAvgPSNR), '+/-', np.std(lastmzAvgPSNR)])
    dataPrintout.append(['Average PSNR:', np.mean(lastAvgPSNR), '+/-', np.std(lastAvgPSNR)])
    dataPrintout.append(['Average ERD PSNR:', np.mean(lastERDPSNR), '+/-', np.std(lastERDPSNR)])
    dataPrintout.append(['Average Time:', np.mean(lastTime), '+/-', np.std(lastTime)])
    pd.DataFrame(dataPrintout).to_csv(dir_TestingResults + 'dataPrintout.csv')