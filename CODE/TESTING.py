#==================================================================
#TESTING SLADS SPECIFIC
#==================================================================

#Given a set of sample paths, perform testing using a trained SLADS Model
def testSLADS(sortedTestingSampleFolders, model, optimalC):

    #If consistentcy in the random generator is desired for comparisons
    if consistentSeed: np.random.seed(0)
    
    #Setup testing samples
    testingSamples = []
    for testingSampleFolder in sortedTestingSampleFolders:
        dataSampleName = os.path.basename(testingSampleFolder)
        
        #Read all available scan data into a sample object
        sample = Sample(testingSampleFolder, ignoreMissingLines=True)
        sample.readScanData(lineRevistMethod)

        #Create a mask object
        sample.maskObject = MaskObject(sample.numColumns, sample.numLines, initialPercToScan, scanMethod)

        #Perform averaging of the multiple channels and subsequent normalization
        sample.avgGroundTruthImage = np.average(np.asarray(sample.mzImages), axis=0, weights=sample.mzWeights)
        sample.avgGroundTruthImage = MinMaxScaler().fit_transform(sample.avgGroundTruthImage.reshape(-1, 1)).reshape(sample.avgGroundTruthImage.shape)

        #Indicate where resulting data should be stored
        sample.resultsPath = dir_TestingResults

        #Define information as a new Sample object
        testingSamples.append(copy.deepcopy(sample))

    #Create holding arrays for all of the results
    MSE_testingResults = []
    SSIM_testingResults = []
    PSNR_testingResults = []
    TD_testingResults = []
    ERDPSNR_testingResults = []
    perc_testingResults = []
    time_testingResults = []
    
    #If an animation will be produced
    if animationGen:
        dir_AnimationVideos = dir_Animations + 'Videos/'
        if os.path.exists(dir_AnimationVideos): shutil.rmtree(dir_AnimationVideos)
        os.makedirs(dir_AnimationVideos)

    for sampleNum in tqdm(range(0,len(testingSamples)), desc='Testing Samples', position=0, leave=True, ascii=True):
        t0 = time.time()
        result = runSLADS([testingSamples[sampleNum]], model, scanMethod, optimalC, percToScan, stopPerc, 0, simulationFlag=True, trainPlotFlag=False, animationFlag=animationGen, tqdmHide=False, oracleFlag=False, bestCFlag=False)
        time_testingResults.append(time.time()-t0)
        result.complete(optimalC)
        MSE_testingResults.append(result.MSEList)
        SSIM_testingResults.append(result.SSIMList)
        PSNR_testingResults.append(result.PSNRList)
        TD_testingResults.append(result.TDList)
        ERDPSNR_testingResults.append(result.ERDPSNRList)
        perc_testingResults.append(result.percMeasuredList)
    
    #Extract percentage results at the specified precision
    percents, testingMSE_mean = percResults(MSE_testingResults, perc_testingResults, precision)
    percents, testingSSIM_mean = percResults(SSIM_testingResults, perc_testingResults, precision)
    percents, testingPSNR_mean = percResults(PSNR_testingResults, perc_testingResults, precision)
    percents, testingTD_mean = percResults(TD_testingResults, perc_testingResults, precision)
    percents, ERDPSNR_mean = percResults(ERDPSNR_testingResults, perc_testingResults, precision)

    #Save average results per percentage data
    np.savetxt(dir_TestingResults+'testingAverageMSE_Percentage.csv', np.transpose([percents, testingMSE_mean]), delimiter=',')
    np.savetxt(dir_TestingResults+'testingAverageSSIM_Percentage.csv', np.transpose([percents, testingSSIM_mean]), delimiter=',')
    np.savetxt(dir_TestingResults+'testingAveragePSNR_Percentage.csv', np.transpose([percents, testingPSNR_mean]), delimiter=',')
    np.savetxt(dir_TestingResults+'testingAverageTD_Percentage.csv', np.transpose([percents, testingTD_mean]), delimiter=',')
    np.savetxt(dir_TestingResults+'testingAverageERDPSNR_Percentage.csv', np.transpose([percents, ERDPSNR_mean]), delimiter=',')

    #Save average plots per percentage data in plot
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, testingMSE_mean,color='black') 
    ax1.set_xlabel('% Pixels Measured')
    ax1.set_ylabel('Average MSE')
    plt.savefig(dir_TestingResults + 'testingAverageMSE_Percentage' + '.png')
    plt.close()

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, testingSSIM_mean,color='black') 
    ax1.set_xlabel('% Pixels Measured')
    ax1.set_ylabel('Average SSIM')
    plt.savefig(dir_TestingResults + 'testingAverageSSIM_Percentage' + '.png')
    plt.close()

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, testingPSNR_mean,color='black') 
    ax1.set_xlabel('% Pixels Measured')
    ax1.set_ylabel('Average PSNR')
    plt.savefig(dir_TestingResults + 'testingAveragePSNR_Percentage' + '.png')
    plt.close()
    
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, testingTD_mean,color='black') 
    ax1.set_xlabel('% Pixels Measured')
    ax1.set_ylabel('Average Total Distortion')
    plt.savefig(dir_TestingResults + 'testingAverageTD_Percentage' + '.png')
    plt.close()

    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, ERDPSNR_mean,color='black') 
    ax1.set_xlabel('% Pixels Measured')
    ax1.set_ylabel('Average ERD PSNR')
    plt.savefig(dir_TestingResults + 'ERDPSNR_Percentage' + '.png')
    plt.close()

    #Find the final results for each image
    lastSSIMResult = []
    lastMSEResult = []
    lastPSNRResult = []
    lastTDResult = []
    lastERDPSNRResult = []
    lastTimeResult = []
    #percLinesScanned = []
    percPixelsScanned = []
    for i in range(0, len(SSIM_testingResults)):
        lastPercMeasured = perc_testingResults[i][len(perc_testingResults[i])-1]
        lastSSIMResult.append(SSIM_testingResults[i][len(SSIM_testingResults[i])-1])
        lastMSEResult.append(MSE_testingResults[i][len(MSE_testingResults[i])-1])
        lastPSNRResult.append(PSNR_testingResults[i][len(PSNR_testingResults[i])-1])
        lastTDResult.append(TD_testingResults[i][len(TD_testingResults[i])-1])
        lastERDPSNRResult.append(ERDPSNR_testingResults[i][len(ERDPSNR_testingResults[i])-1])
        lastTimeResult.append(time_testingResults[i]/lastPercMeasured)
        #percLinesScanned.append((len(SSIM_testingResults[i])/len(maskObject.linesToScan))*100)
        percPixelsScanned.append(lastPercMeasured)

    time_testingResults = time_testingResults/np.mean(percPixelsScanned)

    #Printout final results 
    dataPrintout = []
    dataPrintout.append(['Average MSE:', np.mean(lastMSEResult), '+/-', np.std(lastMSEResult)])
    dataPrintout.append(['Average SSIM:', np.mean(lastSSIMResult), '+/-', np.std(lastSSIMResult)])
    dataPrintout.append(['Average PSNR:', np.mean(lastPSNRResult), '+/-', np.std(lastPSNRResult)])
    dataPrintout.append(['Average TD:', np.mean(lastTDResult), '+/-', np.std(lastTDResult)])
    dataPrintout.append(['Average ERD PSNR:', np.mean(lastERDPSNRResult), '+/-', np.std(lastERDPSNRResult)])
    dataPrintout.append(['Average Time', np.mean(lastTimeResult), '+/-', np.std(lastTimeResult)])
    #dataPrintout.append(['Average % Lines Scanned:', np.mean(percLinesScanned),'+/-', np.std(percLinesScanned)])
    dataPrintout.append(['Average % Pixels Scanned:', np.mean(percPixelsScanned),'+/-',np.std(percPixelsScanned)])
    pd.DataFrame(dataPrintout).to_csv(dir_TestingResults + 'dataPrintout.csv')



