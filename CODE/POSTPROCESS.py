#==================================================================
#POST-PROCESSING SPECIFIC METHOD AND CLASS DEFINITIONS
#==================================================================

#Perform SLADS with external equipment
def postprocess(sortedSampleFolders, optimalC, modelName):

    #Setup a model only on a single GPU (if available), running once on for pre-compilation (otherwise affects reported timings)
    if (erdModel == 'DLADS' or erdModel == 'GLANDS') and numGPUs > 0: 
        model = Model_Actor.remote(erdModel, dir_TrainingResults+modelName, 0)
        _ = ray.get(model.generateERD.remote(np.empty((1,512,512,3), dtype=np.float32)))
    else: 
        model = Model_Actor.remote(erdModel, dir_TrainingResults+modelName)
    
    #Load in data, creating corresponding sample and result objects
    sampleDataset = [SampleData(sampleFolder, 0, stopPerc, scanMethod, lineRevist, True, True, False) for sampleFolder in tqdm(sortedSampleFolders, desc='Samples', leave=True, ascii=asciiFlag)]
    samples = [Sample(sampleData) for sampleData in sampleDataset]
    results = [Result(sampleData, liveOutputFlag, dir_PostResults, False, False, optimalC, False) for sampleData in sampleDataset]
    
    #Perform computations for the measurement state and update corresponding result objects
    for sampleNum in range(0, len(sortedSampleFolders)):
        samples[sampleNum].performMeasurements(sampleDataset[sampleNum], results[sampleNum], [], model, optimalC, False, False, False, False)
        results[sampleNum].update(samples[sampleNum])

    #Call completion/printout function for each result
    _ = [result.complete() for result in tqdm(results, desc='Visualization', position=0, leave=True, ascii=True)]