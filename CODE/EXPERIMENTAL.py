#==================================================================
#IMPLEMENTATION SLADS SPECIFIC
#==================================================================

#Signal external equipment and wait for LOCK file to exist
def equipWait():
    
    #Signal the equipment by removing the LOCK file if it exists
    if os.path.isfile('./INPUT/IMP/LOCK'): os.remove('./INPUT/IMP/LOCK')
    while True:
        if not os.path.isfile('./INPUT/IMP/LOCK'):
            time.sleep(0.1)
        else:
            break

#Perform SLADS with external equipment
def performImplementation(model, bestC):

    #Clean up files from previous runs if they exist
    if os.path.isfile('./INPUT/IMP/DONE'): os.remove('./INPUT/IMP/DONE')
    if os.path.isfile('./INPUT/IMP/UNLOCK'): os.remove('./INPUT/IMP/UNLOCK')
    
    #Wait for equipment to initialize scan
    equipWait()
    
    #Read in the image data for size information
    images, massRanges, imageHeight, imageWidth = readScanData('./INPUT/IMP/')
    
    #Create a mask object
    maskObject = MaskObject(imageWidth, imageHeight, [], 0, scanMethod)
    
    #For each of the initial sets that must be obtained
    for setNum in range(0, len(maskObject.initialSets)):

        #Export the set to a file UNLOCK
        with open('./INPUT/IMP/UNLOCK', 'w') as filehandle: filehandle.writelines(str(maskObject.initialSets[setNum][0]) + ', ' + str(maskObject.initialSets[setNum][1]))
        
        #Wait for equipment to finish scan
        equipWait()

    #Update internal sample data with the acquired information
    images, massRanges, imageHeight, imageWidth = readScanData('./INPUT/IMP/')

    #Weight images equally
    mzWeights = np.ones(len(images))/len(images)

    #Define information as a new Sample object
    impSample = Sample(impSampleName, images, massRanges, maskObject, mzWeights, dir_ImpResults)
    
    #Run SLADS
    result = runSLADS(impSample, bestModel, bestC, percToScan, stopPerc, 0, simulationFlag=False, trainPlotFlag=False, animationFlag=animationGen, tqdmHide=False, oracleFlag=False, bestCFlag=False)
    
    #Call completion/printout function
    result.complete(0)
    
    #Indicate to equipment that the sample scan has concluded
    with open('./INPUT/IMP/DONE', 'w') as filehandle: filehandle.writelines('')

    #Move all of the csv files to finalized directory
    for fileName in glob.glob('./INPUT/IMP/*'): shutil.move(fileName, dir_ImpDataFinal)
