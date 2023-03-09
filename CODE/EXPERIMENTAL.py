#==================================================================
#IMPLEMENTATION/EXPERIMENTAL SPECIFIC METHOD AND CLASS DEFINITIONS
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

    #Setup a model only on a single GPU (if available), running once on for pre-compilation (otherwise affects reported timings)
    if (erdModel == 'DLADS' or erdModel == 'GLANDS') and numGPUs > 0: 
        model = Model_Actor.remote(erdModel, dir_TrainingResults+modelName, 0)
        _ = ray.get(model.generateERD.remote(np.empty((1,512,512,len(inputChannels)), dtype=np.float32)))
    else: 
        model = Model_Actor.remote(erdModel, dir_TrainingResults+modelName)
    
    #Wait for equipment to initialize scan
    equipWait()
    
    #Create a sample object and read the first sets of information
    sampleData = SampleData(dir_ImpDataFinal, initialPercToScan, stopPerc, scanMethod, lineRevist, False, False, False, False, liveOutputFlag, True, False, False, impSampleName)

    #Run sampling
    result = runSampling(sampleData, optimalC, model, percToScan, percToViz, lineVisitAll, dir_ImpResults, False)
    
    #Indicate to equipment that the sample scan has concluded
    print('Writing DONE')
    with open(dir_ImpDataFinal + 'DONE', 'w') as filehandle: filehandle.writelines('')
    if os.path.isfile(dir_ImpDataFinal + 'LOCK'): os.remove(dir_ImpDataFinal + 'LOCK')
    
    #Remove loaded model from memory, returning any allocated resources to Ray
    del model
    
    #Call for completion/printout
    result.complete()

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