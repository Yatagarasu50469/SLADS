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
    
    #Set a default aspect ratio
    aspect = [1,1]
    
    if physResize:
        #Import the sample's pixel aspect ratio (width, height)
        aspect = np.loadtxt('./INPUT/IMP/aspect.txt', delimiter=',')
    
    images = []
    originalImages = []
    massRanges = []
    #Import each of the images according to their mz range order
    for imageFileName in natsort.natsorted('./INPUT/IMP/*.' + 'csv'), reverse=False):
        image = np.nan_to_num(np.loadtxt(imageFileName, delimiter=','))
        imageHeight, imageWidth = image.shape
        if physResize:
            originalImages.append(image)
            if imageWidth > imageHeight:
                image = cv2.resize((image), (int(round((aspect[0]/aspect[1])*imageHeight)), imageHeight), interpolation = cv2.INTER_LINEAR)
            elif imageHeight > imageWidth:
                image = cv2.resize((image), (imageWidth, int(round((aspect[0]/aspect[1])*imageWidth))), interpolation = cv2.INTER_LINEAR)
        if not physResize:
            originalImages.append(image)
        images.append(image)
        massRanges.append([os.path.basename(imageFileName)[2:10], os.path.basename(imageFileName)[11:19]])

    #Create a new maskObject
    maskObject = MaskObject(imageWidth, imageHeight, image.shape[1], image.shape[0], [], 0)

    return images, originalImages, massRanges, maskObject

#Perform SLADS with external equipment
def performImplementation(bestC, bestModel):

    #Clean up files from previous runs if they exist
    if os.path.isfile('./INPUT/IMP/DONE'): os.remove('./INPUT/IMP/DONE')
    if os.path.isfile('./INPUT/IMP/UNLOCK'): os.remove('./INPUT/IMP/UNLOCK')
    
    #Wait for equipment to initialize scan
    equipWait()
    
    #Read in the image data (blank) for size information
    images, massRanges, maskObject = readScanData()
    
    #For each of the initial sets that must be obtained
    for setNum in range(0, len(maskObject.initialSets)):

        #Export the set to a file UNLOCK
        with open('./INPUT/IMP/UNLOCK', 'w') as filehandle: filehandle.writelines(str(maskObject.initialSets[setNum][0]) + ', ' + str(maskObject.initialSets[setNum][1]))
        
        #Wait for equipment to finish scan
        equipWait()

    #Update internal sample data with the acquired information
    images, originalImages, massRanges = readScanData()

    #Weight images equally
    mzWeights = np.ones(len(images))/len(images)

    #Define information as a new Sample object
    impSample = Sample(impSampleName, images, originalImages, massRanges, maskObject, mzWeights, dir_ImpResults)
    
    #Run SLADS
    result = runSLADS(info, impSample, bestModel, stopPerc, 0, simulationFlag=False, trainPlotFlag=False, animationFlag=animationGen, tqdmHide=False, bestCFlag=False)
    
    #Indicate to equipment that the sample scan has concluded
    with open('./INPUT/IMP/DONE', 'w') as filehandle: filehandle.writelines('')

    #Move all of the csv files to finalized directory
    for fileName in glob.glob('./INPUT/IMP/*'): shutil.move(fileName, dir_ImpDataFinal)
