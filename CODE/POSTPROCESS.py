#==================================================================
#POST-PROCESSING SPECIFIC METHOD AND CLASS DEFINITIONS
#==================================================================

#Perform SLADS with external equipment
def postprocess(sortedSampleFolders, optimalC, modelName):

    #Setup a model only on a single GPU (if available), running once on for pre-compilation (otherwise affects reported timings)
    if (erdModel == 'DLADS' or erdModel == 'GLANDS') and numGPUs > 0: 
        model = Model_Actor.remote(erdModel, dir_TrainingResults+modelName, 0)
        _ = ray.get(model.generateERD.remote(np.empty((1,512,512,len(inputChannels)), dtype=np.float32)))
    else: 
        model = Model_Actor.remote(erdModel, dir_TrainingResults+modelName)
    
    #Load in data, creating corresponding sample and result objects
    sampleDataset = [SampleData(sampleFolder, 0, stopPerc, scanMethod, lineRevist, True, False, False, False, liveOutputFlag, False, True, False) for sampleFolder in tqdm(sortedSampleFolders, desc='Samples', leave=True, ascii=asciiFlag)]
    results = [Result(sampleData, dir_PostResults, optimalC) for sampleData in sampleDataset]
    
    #Perform computations for the measurement state and update corresponding result objects
    for sampleNum in range(0, len(sortedSampleFolders)):
        tempScanData = TempScanData()
        sample = Sample(sampleDataset[sampleNum], tempScanData)
        sample.performMeasurements(sampleDataset[sampleNum], tempScanData, results[sampleNum], [], model, optimalC, False)
        results[sampleNum].update(sample, True)

    #Call completion/printout function for each result
    _ = [result.complete() for result in tqdm(results, desc='Processing', position=0, leave=True, ascii=asciiFlag)]