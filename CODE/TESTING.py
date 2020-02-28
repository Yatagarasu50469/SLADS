#==================================================================
#TESTING SLADS SPECIFIC
#==================================================================

#Given a set of sample paths, perform testing using a trained SLADS Model
def testSLADS(sortedTestingSampleFolders, bestC, bestModel):
    
    #Setup testing samples
    testingSamples = []
    for testingSampleFolder in sortedTestingSampleFolders:
        dataSampleName = os.path.basename(testingSampleFolder)
    
        #Obtain images
        images = []
        massRanges = []
        for imageFileName in natsort.natsorted(glob.glob(testingSampleFolder + '/*.' + 'csv'), reverse=False):
            images.append(np.nan_to_num(np.loadtxt(imageFileName, delimiter=',')))
            massRanges.append([os.path.basename(imageFileName)[2:10], os.path.basename(imageFileName)[11:19]])

        #Create a new maskObject
        maskObject = MaskObject(images[0].shape[1], images[0].shape[0], measurementPercs=[], numMasks=0)
    
        #Weight images equally
        mzWeights = np.ones(len(images))/len(images)

        #Define information as a new Sample object
        testingSamples.append(Sample(dataSampleName, images, massRanges, maskObject, mzWeights, dir_TestingResults))

    #Set function for the pool
    #with contextlib.redirect_stdout(None):
    #    parFunction = ray.remote(runSLADS)
    #    time.sleep(1)

    #Add constant static parameters to shared pool memory
    #info_id = ray.put(info)
    #testingSamples_id = ray.put(testingSamples)
    #testingModel_id = ray.put(bestModel)
    #stopPerc_id = ray.put(stopPerc)
    #simulationFlag_id = ray.put(True)
    #trainPlotFlag_id = ray.put(False)
    #animationFlag_id = ray.put(animationGen)
    #tqdmHide_id = ray.put(True)
    #bestCFlag_id = ray.put(False)

    #Create holding arrays for all of the results
    MSE_testingResults = []
    SSIM_testingResults = []
    TD_testingResults = []
    perc_testingResults = []

    #Perform pool function and extract variables from the results
    #idens = [parFunction.remote(info_id, testingSamples_id, testingModel_id, stopPerc_id, sampleNum, simulationFlag_id, trainPlotFlag_id, animationFlag_id, tqdmHide_id, bestCFlag_id) for sampleNum in range(0, len(testingSamples))]

    #Perform pool function and extract variables from the results
    #for result in tqdm(parIterator(idens), total=len(idens), desc='Testing Samples', position=0, leave=True, ascii=True):
    #    info, testingSamples, bestModel, stopPerc, sampleNum, True, False, animationGen, True, False
    #    MSE_testingResults.append(result.MSEList)
    #    SSIM_testingResults.append(result.SSIMList)
    #    TD_testingResults.append(result.TDList)
    #    perc_testingResults.append(result.percMeasuredList)

    for sampleNum in tqdm(range(0,len(testingSamples)), desc='Testing Samples', position=0, leave=True, ascii=True):
        result = runSLADS(info, testingSamples, bestModel, stopPerc, sampleNum, True, False, animationGen, True, False)
        MSE_testingResults.append(result.MSEList)
        SSIM_testingResults.append(result.SSIMList)
        TD_testingResults.append(result.TDList)
        perc_testingResults.append(result.percMeasuredList)

    #Define precision of the percentage averaging (as percentage is inconsistent between acquistion steps)
    precision = 0.01

    #Extract percentage results at the specified precision
    percents, testingSSIM_mean = percResults(SSIM_testingResults, perc_testingResults, precision)

    #Save average SSIM per percentage data
    np.savetxt(dir_TestingResults+'testingAverageSSIM_Percentage.csv', np.transpose([percents, testingSSIM_mean]), delimiter=',')

    #Save average SSIM per percentage data in plot
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(percents, testingSSIM_mean,color='black') 
    ax1.set_xlabel('% Pixels Measured')
    ax1.set_ylabel('Average SSIM')
    plt.savefig(dir_TestingResults + 'testingAverageSSIM_Percentage' + '.png')
    plt.close()

    #Find the final results for each image
    lastSSIMResult = []
    lastMSEResult = []
    lastTDResult = []
    percLinesScanned = []
    percPixelsScanned = []
    for i in range(0, len(SSIM_testingResults)):
        lastSSIMResult.append(SSIM_testingResults[i][len(SSIM_testingResults[i])-1])
        lastMSEResult.append(MSE_testingResults[i][len(MSE_testingResults[i])-1])
        lastTDResult.append(TD_testingResults[i][len(TD_testingResults[i])-1])
        #percLinesScanned.append((len(SSIM_testingResults[i])/len(maskObject.linesToScan))*100)
        percPixelsScanned.append(perc_testingResults[i][len(perc_testingResults[i])-1])

    #Printout final results 
    dataPrintout = []
    dataPrintout.append(['Average SSIM:', np.mean(lastSSIMResult), '+/-', np.std(lastSSIMResult)])
    dataPrintout.append(['Average MSE:', np.mean(lastMSEResult), '+/-', np.std(lastMSEResult)])
    dataPrintout.append(['Average TD:', np.mean(lastTDResult), '+/-', np.std(lastTDResult)])
    dataPrintout.append(['Average % Lines Scanned:', np.mean(percLinesScanned),'+/-', np.std(percLinesScanned)])
    dataPrintout.append(['Average % Pixels Scanned:', np.mean(percPixelsScanned),'+/-',np.std(percPixelsScanned)])
    pd.DataFrame(dataPrintout).to_csv(dir_TestingResults + 'dataPrintout.csv')



