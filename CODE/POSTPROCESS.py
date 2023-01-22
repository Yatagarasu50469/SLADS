#==================================================================
#POST-PROCESSING SPECIFIC METHOD AND CLASS DEFINITIONS
#==================================================================

#Perform SLADS with external equipment
def postprocess(sortedSampleFolders, optimalC, modelName):

    #Start server, deploy, and get handle for model queries; run Tensorflow model once for pre-compilation (otherwise affects reported timings)
    serve.start()
    ModelServer.deploy(erdModel, dir_TrainingResults+modelName)
    model = ModelServer.get_handle()
    if erdModel == 'DLADS': _ = ray.get(model.remote(np.empty((1,64,64,3), dtype=np.float32)))
    
    #Load in data, creating corresponding sample and result objects
    sampleDataset = [SampleData(sampleFolder, 0, stopPerc, scanMethod, lineRevist, True, True, False) for sampleFolder in tqdm(sortedSampleFolders, desc='Reading', leave=True, ascii=True)]
    samples = [Sample(sampleData) for sampleData in sampleDataset]
    results = [Result(sampleData, liveOutputFlag, dir_PostResults, False, False, optimalC, False) for sampleData in sampleDataset]
    
    #Perform computations for the measurement state and update corresponding result objects
    for sampleNum in range(0, len(sortedSampleFolders)):
        samples[sampleNum].performMeasurements(sampleDataset[sampleNum], results[sampleNum], [], model, optimalC, False, False, False, False)
        results[sampleNum].update(samples[sampleNum])

    #Shutdown the server to ensure resources are returned to ray
    serve.shutdown()

    #Call completion/printout function for each result
    _ = [result.complete() for result in tqdm(results, desc='Visualization', position=0, leave=True, ascii=True)]