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

def readScanData():
    images = []
    massRanges = []
    for imageFileName in natsort.natsorted(glob.glob('./INPUT/IMP/*.' + 'csv'), reverse=False):
        images.append(np.nan_to_num(np.loadtxt(imageFileName, delimiter=',')))
        massRanges.append([os.path.basename(imageFileName)[2:10], os.path.basename(imageFileName)[11:19]])
    return images, massRanges

#Perform SLADS with external equipment
def performImplementation(bestC, bestModel):

    #Clean up files from previous runs if they exist
    if os.path.isfile('./INPUT/IMP/DONE'): os.remove('./INPUT/IMP/DONE')
    if os.path.isfile('./INPUT/IMP/UNLOCK'): os.remove('./INPUT/IMP/UNLOCK')
    
    #Wait for equipment to initialize scan
    equipWait()
    
    #Read in the image data (blank) for size information
    images, massRanges = readScanData()
    
    #Create a new maskObject
    maskObject = MaskObject(images[0].shape[1], images[0].shape[0], [], numMasks)

    #For each of the initial sets that must be obtained
    for setNum in range(0, len(maskObject.initialSets)):

        #Export the set to a file UNLOCK
        with open('./INPUT/IMP/UNLOCK', 'w') as filehandle: filehandle.writelines(str(maskObject.initialSets[setNum][0]) + ', ' + str(maskObject.initialSets[setNum][1]))
        
        #Wait for equipment to finish scan
        equipWait()

    #Update internal sample data with the acquired information
    images, massRanges = readScanData()

    #Weight images equally
    mzWeights = np.ones(len(images))/len(images)

    #Define information as a new Sample object
    impSample = Sample(impSampleName, images, massRanges, maskObject, mzWeights, dir_ImpResults)
    
    #Run SLADS
    result = runSLADS(info, impSample, bestModel, stopPerc, 0, simulationFlag=False, trainPlotFlag=False, animationFlag=animationGen, tqdmHide=False)
    
    #Indicate to equipment that the sample scan has concluded
    with open('./INPUT/IMP/DONE', 'w') as filehandle: filehandle.writelines('')

    #Move all of the csv files to finalized directory
    for fileName in glob.glob('./INPUT/IMP/*.' + 'csv'): shutil.move(fileName, dir_ImpDataFinal)