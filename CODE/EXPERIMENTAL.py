#==================================================================
#IMPLEMENTATION/EXPERIMENTAL
#==================================================================

#Signal external equipment and wait for LOCK file to exist
def equipWait():
    
    #Remove LOCK file and wait until it appears again, then remove UNLOCK file
    if os.path.isfile(dir_ImpDataFinal + 'LOCK'): os.remove(dir_ImpDataFinal + 'LOCK')
    print('Waiting for LOCK')
    while True:
        if not os.path.isfile(dir_ImpDataFinal + 'LOCK'): time.sleep(0.1)
        else: break
    print('Received LOCK')
    if os.path.isfile(dir_ImpDataFinal + 'UNLOCK'): os.remove(dir_ImpDataFinal + 'UNLOCK')

#Perform SLADS with external equipment
def performImplementation(optimalC, modelName):
    
    #Wait for equipment to initialize scan
    equipWait()
    
    #Create a sample object and read the first sets of information
    sampleData = SampleData(dir_ImpDataFinal, initialPercToScan, stopPerc, scanMethod, lineRevist, False, False, False, False, liveOutputFlag, True, False, False, impSampleName)

    #Setup a model
    if numGPUs > 0: model = Model_Actor.remote(erdModel, dir_TrainingResults, modelName, gpus[0])
    else: model = Model_Actor.remote(erdModel, dir_TrainingResults, modelName)
    _ = ray.get(model.setup.remote())

    #Run sampling
    result = runSampling(None, sampleData, optimalC, model, percToScan, percToViz, lineVisitAll, dir_ImpResults, False)
    
    #Indicate to equipment that the sample scan has concluded
    print('Writing DONE')
    with open(dir_ImpDataFinal + 'DONE', 'w') as filehandle: filehandle.writelines('')
    if os.path.isfile(dir_ImpDataFinal + 'LOCK'): os.remove(dir_ImpDataFinal + 'LOCK')
    
    #Call for completion/printout
    result.complete(sampleData)

    #Print out final results
    dataPrintout = [[]]
    dataPrintout.append(['Final %:', result.samples[-1].percMeasured])
    dataPrintout.append(['Run Time (s):', result.finalTime])
    dataPrintout.append([])
    dataPrintout.append(['Note: The following values should not be considered wholly representative of actual code performance/efficiency.'])
    dataPrintout.append(['They are only intended for debugging and internal evaluation of relative system performance!'])
    dataPrintout.append([])
    dataPrintout.append(['File Load Time (s):', result.avgTimeFileLoad])
    dataPrintout.append(['Targeted Reconstruction Compute Time (s):', result.avgTimeComputeRecon])
    dataPrintout.append(['ERD Compute Time (s):', result.avgTimeComputeERD])
    
    pd.DataFrame(dataPrintout).to_csv(dir_ImpDataFinal + 'dataPrintout.csv')
    pd.DataFrame(dataPrintout).to_csv(result.dir_sampleResults + 'dataPrintout.csv')