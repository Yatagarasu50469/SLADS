#==================================================================
#POST-PROCESS
#==================================================================

#Perform SLADS with external equipment
def postprocess(sortedSampleFolders, optimalC, modelName):
    
    #Setup a model only on a single GPU (if available)
    if numGPUs > 0: model = Model_Actor.remote(erdModel, dir_TrainingResults, modelName, gpus[0])
    else: model = Model_Actor.remote(erdModel, dir_TrainingResults, modelName)
    _ = ray.get(model.setup.remote())
    
    for sampleFolder in tqdm(sortedSampleFolders, desc='Samples', leave=True, ascii=asciiFlag):
        
        #Load in data, creating corresponding sample and result objects
        sampleData = SampleData(sampleFolder, 0, stopPerc, scanMethod, lineRevist, True, False, False, False, liveOutputFlag, False, True, False)
        sample = Sample(sampleData, tempScanData)
        result = Result(None, sampleData, dir_PostResults, optimalC)
        
        #Perform computations for the measurement state and update corresponding result objects
        tempScanData = TempScanData()
        sample.performMeasurements(sampleDataset[sampleNum], tempScanData, results[sampleNum], [], model, optimalC, False)
        result.update(sample, sampleData, True)
        
        #Call completion/printout
        _ = result.complete(sampleData)
    