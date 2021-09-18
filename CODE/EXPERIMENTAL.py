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
def performImplementation(optimalC):
    
    #Storage location for matching sequentially generated indexes with physical line numbers
    if unorderedNames: physicalLineNums = {}

    #Wait for equipment to initialize scan
    equipWait()
    
    #Create a sample object and read the first sets of information
    sampleData = SampleData(dir_ImpDataFinal, initialPercToScan, stopPerc, scanMethod, RDMethod, mzGlobalSpec, False, lineRevist, False)

    #Run SLADS
    result = runSLADS(sampleData, optimalC, True, percToScan, percToViz, False, False, lineVisitAll, False)
    
    #Indicate to equipment that the sample scan has concluded
    print('Writing DONE')
    with open(dir_ImpDataFinal + 'DONE', 'w') as filehandle: filehandle.writelines('')
    if os.path.isfile(dir_ImpDataFinal + 'LOCK'): os.remove(dir_ImpDataFinal + 'LOCK')

    #Call completion/printout function
    print('Generating Visualizations')
    result.complete(dir_ImpResults)
