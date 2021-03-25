#==================================================================
#IMPLEMENTATION SLADS SPECIFIC
#==================================================================

#Signal external equipment and wait for LOCK file to exist
def equipWait():
    
    #Signal the equipment by removing the LOCK file if it exists
    if os.path.isfile(dir_ImpResults + 'LOCK'): os.remove(dir_ImpResults + 'LOCK')
    while True:
        if not os.path.isfile(dir_ImpResults + 'LOCK'):
            time.sleep(0.1)
        else:
            break

#Perform SLADS with external equipment
def performImplementation(model, optimalC):

    #Clean up files from previous runs if they exist
    if os.path.isfile(dir_ImpResults + 'DONE'): os.remove(dir_ImpResults + 'DONE')
    if os.path.isfile(dir_ImpResults + 'UNLOCK'): os.remove(dir_ImpResults + 'UNLOCK')
    
    #Wait for equipment to initialize scan
    equipWait()
    
    sample = Sample(dir_ImpResults, initialPercToScan, scanMethod)
    sample.readScanData(lineRevistMethod)
    
    #For each of the initial sets that must be obtained
    for setNum in range(0, len(sample.initialSets)):

        #Export the set to a file UNLOCK
        with open(dir_ImpResults + 'UNLOCK', 'w') as filehandle: filehandle.writelines(str(sample.initialSets[setNum][0]) + ', ' + str(sample.initialSets[setNum][1]))
        
        #Wait for equipment to finish scan
        equipWait()

    #Update internal sample data with the acquired informationssss
    sample.readScanData(lineRevistMethod)

    #Run SLADS
    result = runSLADS(sample, bestModel, scanMethod, optimalC, percToScan, percToViz, stopPerc, simulationFlag=False, trainPlotFlag=False, animationFlag=animationGen, tqdmHide=False, oracleFlag=False, bestCFlag=False)
    
    #Call completion/printout function
    result.complete(None)
    
    #Indicate to equipment that the sample scan has concluded
    with open(dir_ImpResults + 'DONE', 'w') as filehandle: filehandle.writelines('')

    #Move all of the files to finalized directory
    for fileName in glob.glob(dir_ImpResults + '*'): shutil.move(fileName, dir_ImpDataFinal)
