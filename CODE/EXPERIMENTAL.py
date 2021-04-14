#==================================================================
#IMPLEMENTATION SLADS SPECIFIC
#==================================================================

#Signal external equipment and wait for LOCK file to exist
def equipWait():
    
    #Signal the equipment by removing the LOCK file if it exists
    if os.path.isfile(dir_ImpDataFinal + 'LOCK'): os.remove(dir_ImpDataFinal + 'LOCK')
    print('Waiting for LOCK')
    while True:
        if not os.path.isfile(dir_ImpDataFinal + 'LOCK'):
            time.sleep(0.1)
        else:
            break
    print('Received LOCK')

#Perform SLADS with external equipment
def performImplementation(model, optimalC):

    #Wait for equipment to initialize scan
    equipWait()
    
    #Create a sample object and read the first sets of information
    sample = Sample(dir_ImpDataFinal, initialPercToScan, scanMethod)
    
    #Indicate where resulting data should be stored
    sample.resultsPath = dir_ImpResults
    
    #For each of the initial sets that must be obtained, print out the infomration to UNLOCK and wait for equipment
    for setNum in range(0, len(sample.initialSets)):
        print('Writing UNLOCK')
        with open(dir_ImpDataFinal + 'UNLOCK', 'w') as filehandle: _ = [filehandle.writelines(str(tuple([pos[0]+1, pos[1]]))+'\n') for pos in sample.initialSets[setNum]]
        equipWait()

    #Update internal sample data with the acquired informations
    sample.readScanData(lineRevistMethod)

    #Run SLADS
    result = runSLADS(sample, model, scanMethod, optimalC, percToScan, percToViz, stopPerc, simulationFlag=False, trainPlotFlag=False, animationFlag=animationGen, tqdmHide=False, oracleFlag=False, bestCFlag=False)
    
    #Indicate to equipment that the sample scan has concluded
    print('Writing DONE')
    with open(dir_ImpDataFinal + 'DONE', 'w') as filehandle: filehandle.writelines('')

    #Call completion/printout function
    print('Generating Visualizations')
    result.complete(None)
