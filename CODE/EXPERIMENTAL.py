#==================================================================
#IMPLEMENTATION SLADS SPECIFIC
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

    #Start server, deploy, and get handle for model queries; run Tensorflow model once for pre-compilation (otherwise affects reported timings)
    serve.start()
    ModelServer.deploy(erdModel, dir_TrainingResults+modelName)
    model = ModelServer.get_handle()
    if erdModel == 'DLADS': _ = ray.get(model.remote(np.empty((1,64,64,3)))).copy()

    #Wait for equipment to initialize scan
    equipWait()
    
    #Create a sample object and read the first sets of information
    sampleData = SampleData(dir_ImpDataFinal, initialPercToScan, stopPerc, scanMethod, lineRevist, False, False)

    #Run sampling
    result = runSampling(sampleData, optimalC, model, percToScan, percToViz, False, False, lineVisitAll, liveOutputFlag, dir_ImpResults, False, True, False)
    
    #Indicate to equipment that the sample scan has concluded
    print('Writing DONE')
    with open(dir_ImpDataFinal + 'DONE', 'w') as filehandle: filehandle.writelines('')
    if os.path.isfile(dir_ImpDataFinal + 'LOCK'): os.remove(dir_ImpDataFinal + 'LOCK')

    #Shutdown the server to ensure resources are returned to ray
    serve.shutdown()

    #Call completion/printout function
    print('Generating Visualizations')
    result.complete()

