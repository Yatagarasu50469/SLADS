#==================================================================
#SLADS DEFINITIONS GENERAL
#==================================================================

#General information regarding samples, used for testing and best C value determination
class Sample:
    def __init__(self, name, images, massRanges, maskObject, mzWeights, resultsPath):
        self.name = name
        self.images = images
        self.massRanges = massRanges
        self.maskObject = maskObject
        self.mzWeights = mzWeights
        self.measuredImages = []
        for rangeNum in range(0,len(massRanges)): self.measuredImages.append(np.zeros([maskObject.width, maskObject.height]))
        self.resultsPath = resultsPath

#Trained SLADS Model object
class SLADSModel:
    def __init__(self, massRange, theta, cValue):
        self.massRange = massRange
        self.theta = theta
        self.cValue = cValue

#Singular result generated through runSLADS
class Result():
    def __init__(self, info, sample, avgImage, simulationFlag, animationFlag):
        self.info = info
        self.sample = sample
        self.avgImage = avgImage
        self.simulationFlag = simulationFlag
        self.animationFlag = animationFlag
        self.reconImages = []
        self.masks = []
        self.ERDValueNPs = []
        self.TDList = []
        self.MSEList = []
        self.SSIMList = []
        self.percMeasuredList = []

    def update(self, percMeasured, reconImage, sample, maskObject, ERDValuesNP, iterNum, completedRunFlag):

        #Save the model development
        self.reconImages.append(reconImage)
        self.masks.append(maskObject.mask.copy())
        self.ERDValueNPs.append(ERDValuesNP.copy())
        self.sample = sample

        if self.simulationFlag:

            #Find statistics of interest
            difference = np.sum(computeDifference(self.avgImage, reconImage, self.info.imageType))
            TD = difference/maskObject.area
            MSE = (np.sum((reconImage.astype("float") - self.avgImage.astype("float")) ** 2))/(float(maskObject.area))
            SSIM = structural_similarity(reconImage.astype("float"), self.avgImage.astype("float"))

            #Save them for each timestep
            self.TDList.append(TD)
            self.MSEList.append(MSE)
            self.SSIMList.append(SSIM)
            self.percMeasuredList.append(percMeasured)

        #If an animation will be produced and the run has completed
        if self.animationFlag and completedRunFlag:

            #Setup directory addresses
            dir_mzResults = self.sample.resultsPath + 'mzResults/'
            dir_mzSampleResults = dir_mzResults + self.sample.name + '/'

            dir_Animations = self.sample.resultsPath+ 'Animations/'
            dir_AnimationVideos = dir_Animations + 'Videos/'
            dir_AnimationFrames = dir_Animations + self.sample.name + '/'

            #Clean directories
            if os.path.exists(dir_AnimationFrames): shutil.rmtree(dir_AnimationFrames)
            os.makedirs(dir_AnimationFrames)

            if os.path.exists(dir_mzSampleResults): shutil.rmtree(dir_mzSampleResults)
            os.makedirs(dir_mzSampleResults)

        #If an animation should be produced and the run has completed for a simulation
        if self.animationFlag and completedRunFlag:

            #Normalize values
            self.ERDValueNPs = (self.ERDValueNPs-np.min(self.ERDValueNPs))*((255.0-0.0)/(np.max(self.ERDValueNPs)-np.min(self.ERDValueNPs)))+0.0

            #If this was a simulation
            if self.simulationFlag:

                #Save each of the individual mass range reconstructions
                percSampled = "{:.2f}".format(self.percMeasuredList[len(self.percMeasuredList)-1])
                for massNum in range(0, len(self.sample.massRanges)):
                    subReconImage = sample.measuredImages[massNum].astype("float")
                    mzImage = self.sample.images[massNum].astype("float")
                    SSIM = "{:.2f}".format(structural_similarity(subReconImage, mzImage))
                    massRange = str(self.sample.massRanges[massNum][0]) + '-' + str(self.sample.massRanges[massNum][1])

                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(15,5))
                    f.subplots_adjust(top = 0.80)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle('Percent Sampled: ' + percSampled + '  Measurement mz: ' + massRange + '  SSIM: ' + SSIM, fontsize=20, fontweight='bold', y = 0.95)

                    sub = f.add_subplot(1,3,1)
                    sub.imshow(self.masks[len(self.masks)-1 ], cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')

                    sub = f.add_subplot(1,3,2)
                    sub.imshow(mzImage * 255.0/mzImage.max(), cmap='hot', aspect='auto')
                    sub.set_title('Ground-Truth')

                    sub = f.add_subplot(1,3,3)
                    sub.imshow(subReconImage * 255.0/subReconImage.max(), cmap='hot', aspect='auto')
                    sub.set_title('Reconstruction')

                    saveLocation = dir_mzSampleResults + massRange +'.png'

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()


                #Generate each of the frames
                for i in range(0, len(self.masks)):

                    saveLocation = dir_AnimationFrames + 'stretched_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'

                    #1x3 without ERD
                    #=====================
                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(15,5))

                    f.subplots_adjust(top = 0.7)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle("Percent Sampled: %.2f, Measurement Iteration: %.0f\nSSIM: %.2f" % (self.percMeasuredList[i], i+1, self.SSIMList[i]), fontsize=20, fontweight='bold', y = 0.95)

                    sub = f.add_subplot(1,3,1)
                    sub.imshow(self.avgImage * 255.0/self.avgImage.max(), cmap='hot', aspect='auto')
                    sub.set_title('Ground-Truth')

                    sub = f.add_subplot(1,3,2)
                    sub.imshow(self.reconImages[i] * 255.0/self.reconImages[i].max(), cmap='hot', aspect='auto')
                    sub.set_title('Reconstructed Image')

                    sub = f.add_subplot(1,3,3)
                    sub.imshow(self.masks[i], cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()
                    #=====================

                    #2x2 with ERD printout
                    #=====================
#                    font = {'size' : 18}
#                    plt.rc('font', **font)
#                    f = plt.figure(figsize=(15,15))
#            
#                    f.subplots_adjust(top = 0.85)
#                    f.subplots_adjust(wspace=0.15, hspace=0.2)
#                    plt.suptitle("Percent Sampled: %.2f, Measurement Iteration: %.0f\nSSIM: %.2f" % (self.percMeasuredList[i], i+1, self.SSIMList[i]), fontsize=20, fontweight='bold', y = 0.95)
#                    
#                    sub = f.add_subplot(2,2,1)
#                    sub.imshow(self.avgImage * 255.0/self.avgImage.max(), cmap='hot', aspect='auto')
#                    sub.set_title('Ground-Truth')
#                    
#                    sub = f.add_subplot(2,2,2)
#                    sub.imshow(self.reconImages[i] * 255.0/self.reconImages[i].max(), cmap='hot', aspect='auto')
#                    sub.set_title('Reconstructed Image')
#                    
#                    sub = f.add_subplot(2,2,3)
#                    sub.imshow(self.masks[i], cmap='gray', aspect='auto')
#                    sub.set_title('Sampled Mask')
#                    
#                    sub = f.add_subplot(2,2,4)
#                    #im = sub.imshow(self.ERDValueNPs[i]>0, cmap='gray', aspect='auto')
#                    #sub.set_title('ERD Values > 0')
#                    #im = sub.imshow(self.ERDValueNPs[i], cmap='viridis', vmin=0, vmax=255, aspect='auto')
#                    im = sub.imshow(self.ERDValueNPs[i], cmap='viridis', aspect='auto')
#                    sub.set_title('ERD Values')
#                    cbar = f.colorbar(im, ax=sub, orientation='vertical', pad=0.01)
#                    
#                    plt.savefig(saveLocation, bbox_inches='tight')
#                    plt.close()

            #If this wasn't a simulation
            else:
                #Save each of the individual mass range reconstructions
                percSampled = "{:.2f}".format(self.percMeasuredList[len(self.percMeasuredList)-1])
                for massNum in range(0, len(self.sample.massRanges)):
                    subReconImage = sample.measuredImages[massNum].astype("float")
                    mzImage = self.sample.images[massNum].astype("float")
                    massRange = str(self.sample.massRanges[massNum][0]) + '-' + str(self.sample.massRanges[massNum][1])

                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(10,5))
                    f.subplots_adjust(top = 0.80)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle('Percent Sampled: ' + percSampled + '  Measurement mz: ' + massRange + '  SSIM: ' + SSIM, fontsize=20, fontweight='bold', y = 0.95)

                    sub = f.add_subplot(1,2,1)
                    sub.imshow(self.masks[len(self.masks)-1 ], cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')

                    sub = f.add_subplot(1,2,2)
                    sub.imshow(subReconImage * 255.0/subReconImage.max(), cmap='hot', aspect='auto')
                    sub.set_title('Reconstruction')

                    saveLocation = dir_mzSampleResults + massRange +'.png'

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()

                #Generate each of the frames
                for i in range(0, len(self.masks)):

                    saveLocation = dir_AnimationFrames + 'stretched_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'

                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(15,15))

                    f.subplots_adjust(top = 0.85)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle("Percent Sampled: %.2f, Measurement Iteration: %.0f" % (self.percMeasuredList[i], i+1), fontsize=20, fontweight='bold', y = 0.95)

                    sub = f.add_subplot(1,3,1)
                    sub.imshow(self.reconImages[i] * 255.0/self.reconImages[i].max(), cmap='hot', aspect='auto')
                    sub.set_title('Reconstructed Image')

                    sub = f.add_subplot(1,3,2)
                    sub.imshow(self.masks[i], cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')

                    sub = f.add_subplot(1,3,3)
                    im = sub.imshow(self.ERDValueNPs[i]>0, cmap='gray', aspect='auto')
                    #im = sub.imshow(self.ERDValueNPs[i], cmap='viridis', vmin=0, vmax=255, aspect='auto')
                    sub.set_title('ERD Values > 0')
                    #cbar = f.colorbar(im, ax=sub, orientation='vertical', pad=0.01)

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()

            dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'stretched_*.png'))
            height, width, layers = cv2.imread(dataFileNames[0]).shape
            animation = cv2.VideoWriter(dir_AnimationVideos + 'stretched_' + self.sample.name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
            for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
            animation.release()
            animation = None

class Info:
    def __init__(self, reconMethod, featReconMethod, neighborWeightsPower, numNeighbors, filterType, featDistCutoff, resolution, imageType):
        self.reconMethod = reconMethod
        self.featReconMethod = featReconMethod
        self.neighborWeightsPower = neighborWeightsPower
        self.numNeighbors = numNeighbors
        self.filterType = filterType
        self.featDistCutoff = featDistCutoff
        self.resolution = resolution
        self.imageType = imageType

#Storage location for the stopping parameters
class StopCondParams:
    def __init__(self, area, threshold, JforGradient, minPercentage, maxPercentage):
        if area<512**2+1:
            self.beta = 0.001*(((18-math.log(area,2))/2)+1)
        else:
            self.beta = 0.001/(((math.log(area,2)-18)/2)+1)
        self.threshold = threshold
        self.JforGradient = JforGradient
        self.minPercentage = minPercentage
        self.maxPercentage = maxPercentage

#Each sample needs a mask object
class MaskObject():
    def __init__(self, width, height, measurementPercs):
        self.width = width
        self.height = height
        self.area = width*height
        self.initialMaskPts = []
        self.percMasks = []
        self.measuredIdxs = []
        self.unMeasuredIdxs = []
        self.initialMeasuredIdxs = []
        self.initialUnMeasuredIdxs = []
        self.unMeasuredIdxsList = []
        self.measuredIdxsList = []

        #Generate a list of arrays contianing the x,y points that need to be scanned
        self.linesToScan = []
        for rowNum in np.arange(0,height,1):
            line = []
            for columnNum in np.arange(0, width, 1):
                line.append(tuple([rowNum, columnNum]))
            self.linesToScan.append(line)

        #Generate the initial set of linesToScan
        self.originalLinesToScan = copy.copy(self.linesToScan)
        self.initialMask = np.zeros([height, width])

        #Set which lines should be acquired in initial scan
        lineIndexes = [
            int((height-1)*0.1),
            int((height-1)*0.50),
            int((height-1)*0.9)
        ]

        #Set which lines should be acquired in initial scan
        for lineIndexNum in range(0, len(lineIndexes)):
            #Subtract the number of lines deleted so far
            lineIndex = lineIndexes[lineIndexNum]
            for pt in self.linesToScan[lineIndex]:
                self.initialMask[tuple(pt)] = 1
                self.initialMaskPts.append(pt)

        #Now delete the lines specified
        for lineIndexNum in range(0, len(lineIndexes)): self.delLine(lineIndexes[lineIndexNum]-lineIndexNum)

        self.initialLinesToScan = copy.copy(self.linesToScan)
        self.initialMeasuredIdxs = np.transpose(np.where(self.initialMask == 1))
        self.initialUnMeasuredIdxs = np.transpose(np.where(self.initialMask == 0))

        #Create random initial percentage masks using point measurements instead of full lines
        for measurementPerc in measurementPercs:
            self.mask = np.zeros([height, width])
            self.mask = np.random.rand(height, width) < (measurementPerc/100)
            self.percMasks.append(self.mask)
            self.measuredIdxsList.append(np.transpose(np.where(self.mask == 1)))
            self.unMeasuredIdxsList.append(np.transpose(np.where(self.mask == 0)))

        #Create random line masks, using full lines
#        for measurementPerc in measurementPercs:
#            self.mask = copy.copy(self.initialMask)
#            self.linesToScan = copy.copy(self.initialLinesToScan)
#            self.measuredIdxs = copy.copy(self.initialMeasuredIdxs)
#            self.unMeasuredIdxs = copy.copy(self.initialUnMeasuredIdxs)
#            while (np.sum(self.mask)/self.area)*100 < measurementPerc:
#                lineIndex = int((np.random.rand(1)[0]*len(self.linesToScan)))
#                for pt in self.linesToScan[lineIndex]: self.mask[tuple(pt)] = 1
#                self.delLine(lineIndex)
#            self.percMasks.append(self.mask)
#            self.measuredIdxsList.append(np.transpose(np.where(self.mask == 1)))
#            self.unMeasuredIdxsList.append(np.transpose(np.where(self.mask == 0)))

    #Update the mask given a set of new measurement locations
    def update(self, newIdxs):
        for pt in newIdxs: self.mask[tuple(pt)] = 1
        self.measuredIdxs = np.transpose(np.where(self.mask == 1))
        self.unMeasuredIdxs = np.transpose(np.where(self.mask == 0))

    #Reset the training sample's mask and linesToScan to nothing having been scanned
    def reset(self):
        self.mask = np.zeros([self.height, self.width])
        self.linesToScan = copy.copy(self.originalLinesToScan)
        self.measuredIdxs = np.transpose(np.where(self.mask == 1))
        self.unMeasuredIdxs = np.transpose(np.where(self.mask == 0))

    def delLine(self, index):
        self.linesToScan = np.delete(self.linesToScan, index, 0)

    def delPoints(self, pts):
        for i in range(0,len(self.linesToScan)):
            indexes = []
            for pt in pts:
                indexes.append([i for i, j in enumerate(self.linesToScan[i]) if j == pt])
            indexes = [x for x in np.asarray(indexes).flatten().tolist() if x != []]
            if len(indexes) > 0:
                self.linesToScan[i] = np.delete(self.linesToScan[i], indexes,0).tolist()

def runSLADS(info, samples, models, stopPerc, cNum, sampleNum, simulationFlag, trainPlotFlag, animationFlag, tqdmHide):
    
    sample = samples[sampleNum]
    maskObject = sample.maskObject
    theta = models[cNum]

    #Reinitialize the mask state to starting state
    maskObject.reset()

    #Has the stopping condition been met yet
    completedRunFlag = False

    #Current iteration
    iterNum = 1

    #Assume variable Classify=='N' (Artifact of pointwise SLADS)

    #Perform weighted averaging for the ground-truth image
    npImages = []
    for image in sample.images: npImages.append(np.asarray(image))
    avgImage = np.average(np.asarray(npImages), axis=0, weights=sample.mzWeights)

    #Initialize stopping condition object
    stopCondParams = StopCondParams(maskObject.area, 0, 50, 2, stopPerc)

    #Determine stoppingCondition function value
    stopCondFuncVal = np.zeros((int((maskObject.area)*(stopCondParams.maxPercentage)/100)+10,2))

    #Perform the initial measurements
    sample, maskObject = performMeasurements(sample, maskObject, maskObject.initialMeasuredIdxs, simulationFlag)

    #Perform initial reconstruction and ERD calculation
    reconImage, reconValues, ERDValues, ERDValuesNP = avgReconAndERD(sample, info, iterNum, maskObject, theta, reconImage=None, reconValues=None, ERDValues=None, ERDValuesNP=None, newIdxs=None, maxIdxsVect=None)

    #Determine percentage pixels measured initially
    percMeasured = (np.sum(maskObject.mask)/maskObject.area)*100

    #Retrieve the averaged image's measured values
    measuredValues = np.asarray(avgImage)[maskObject.mask == 1]

    #Check for completion state here just in case, prior to loop!
    completedRunFlag = checkStopCondFuncThreshold(stopCondParams, stopCondFuncVal, maskObject, measuredValues, iterNum)

    #Additional stopping condition for if there are no more linesToScan
    if len(maskObject.linesToScan) == 0: completedRunFlag = True

    #Initialize a result object
    result = Result(info, sample, avgImage, simulationFlag, animationFlag)
    result.update(percMeasured, reconImage, sample, maskObject, ERDValuesNP, iterNum, completedRunFlag)

    #Until the stopping criteria has been met
    with tqdm(total = float(100), desc = '% Sampled', leave = False, ascii=True, disable=tqdmHide) as pbar:

        #Initialize progress bar state according to % measured
        pbar.n = round(percMeasured,2)
        pbar.refresh()

        #Until the program has completed
        while not completedRunFlag:
            timesList = []
            
            #Step the iteration counter
            iterNum += 1

            t0 = time.time()
            #Make a duplicate of the ReconValues for stop condition gradient test
            oldReconValues = reconValues.copy()
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Find next measurement locations
            maskObject, newIdxs, maxIdxsVect = findNewMeasurementIdxs(info, maskObject, measuredValues, theta, reconValues, reconImage, ERDValues, ERDValuesNP)
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Perform measurements
            sample, maskObject = performMeasurements(sample, maskObject, newIdxs, simulationFlag)
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Perform reconstruction and ERD calculation
            reconImage, reconValues, ERDValues, ERDValuesNP = avgReconAndERD(sample, info, iterNum, maskObject, theta, reconImage, reconValues, ERDValues, ERDValuesNP, newIdxs, maxIdxsVect)
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Update the percentage of pixels that have beene measured
            percMeasured = (np.sum(maskObject.mask)/maskObject.area)*100
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Evaluate the stop condition value
            stopCondFuncVal = computeStopCondFuncVal(oldReconValues, measuredValues, stopCondParams, info, stopCondFuncVal, maxIdxsVect, iterNum, maskObject)
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Retrieve the averaged image's measured values
            measuredValues = np.asarray(avgImage)[maskObject.mask == 1]
            timesList.append(round(time.time()-t0,10))

            t0 = time.time()
            #Check the stopping condition
            completedRunFlag = checkStopCondFuncThreshold(stopCondParams, stopCondFuncVal, maskObject, measuredValues, iterNum)
            timesList.append(round(time.time()-t0,10))

            t0 = time.time()
            #Additional stopping condition for if there are no more linesToScan
            if len(maskObject.linesToScan) == 0: completedRunFlag = True
            timesList.append(round(time.time()-t0,10))

            t0 = time.time()
            #Store information to the resultsObject
            result.update(percMeasured, reconImage, sample, maskObject, ERDValuesNP, iterNum, completedRunFlag)
            timesList.append(round(time.time()-t0,10))

            #Update the progress bar
            pbar.n = round(percMeasured,2)
            pbar.refresh()

            #print(timesList)

    return result

def findNewMeasurementIdxs(info, maskObject, measuredValues, theta, reconValues, reconImage, ERDValues, ERDValuesNP):
    newIdxs = []

    #Assuming manual angle selection; sum ERD for all lines
    lineERDSums = []
    for line in maskObject.linesToScan: lineERDSums.append(np.nansum(ERDValuesNP[tuple([x[0] for x in line]), tuple([y[1] for y in line])]))

    #Choose the line with maximum ERD and extract the actual indices
    lineToScanIdx = np.nanargmax(lineERDSums)
    lineToScanIdxs = maskObject.linesToScan[lineToScanIdx]

    #Set threshold for what ERD Values are worth scanning the locations of
    threshold = 0
    #threshold = np.mean(ERDValuesNP[np.where((ERDValuesNP > 0))])

    #If there is ERD > 0 in the determined line
    if lineERDSums[lineToScanIdx] > 0:

        #Grab the ERD values in that line
        idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in lineToScanIdxs]))
        lineERDValues = ERDValuesNP[np.asarray(idxsX), np.asarray(idxsY)]

        #Find all locations above the threshold
        idxs = np.argwhere(lineERDValues > threshold)
        if len(idxs) > 0:
            newIdxs = lineToScanIdxs[idxs[0][0]:idxs[len(idxs)-1][0]+1]
        else:
            #No location was found over the threshold, just scan the full line
            newIdxs = lineToScanIdxs
    else:
        #Just choose a line to scan...
        newIdxs = lineToScanIdxs

    #Remove the line selected from further consideration
    maskObject.delLine(lineToScanIdx)

    #Extract the positions of those coordinates within the unmeasured listing
    maxIdxsVect = []
    unMeasuredIdxsList = maskObject.unMeasuredIdxs.tolist()
    ptArray = np.asarray(newIdxs).tolist()
    for pt in ptArray:
        if pt in unMeasuredIdxsList: maxIdxsVect.append(unMeasuredIdxsList.index(pt))

    return maskObject, np.asarray(newIdxs), np.asarray(maxIdxsVect)

def avgReconAndERD(sample, info, iterNum, maskObject, theta, reconImage, reconValues, ERDValues, ERDValuesNP, newIdxs, maxIdxsVect):

    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(info, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)
    
    #Perform weighted averaging of the measured images
    image = np.average(np.asarray(sample.measuredImages), axis=0, weights=sample.mzWeights)
    
    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
    measuredValues = image[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Find neighborhood values
    neighborValues = findNeighborValues(measuredValues, neighborIndices)
    
    #Compute reconstructions
    reconValues, reconImage = computeRecons(info, maskObject, maskObject.unMeasuredIdxs, maskObject.measuredIdxs, neighborValues, neighborWeights, measuredValues)
    
    #Compute full ERD Values
    ERDValues = computeFullERD(info, maskObject, measuredValues, reconValues, reconImage, theta, neighborValues, neighborWeights, neighborDistances)
    
#    
#    for image in sample.measuredImages:
#
#        #Retrieve the measured values
#        measuredValues = np.asarray(image)[maskObject.measuredIdxs]
#
#        #Find neighborhood values
#        neighborValues = findNeighborValues(measuredValues, neighborIndices)
#
#        #Compute reconstructions
#        reconValues, reconImage = computeRecons(info, maskObject, maskObject.unMeasuredIdxs, maskObject.measuredIdxs, neighborValues, neighborWeights, measuredValues)
#
#        #Compute full ERD Values
#        ERDValues = computeFullERD(info, maskObject, measuredValues, reconValues, reconImage, theta, neighborValues, neighborWeights, neighborDistances)
#
#        #Store results
#        reconImageList.append(reconImage)
#        reconValuesList.append(reconValues)
#        ERDValueList.append(ERDValues)
#
#    #Perform weighted averaging
#    reconImage = np.average(np.asarray(reconImageList), axis=0, weights=sample.mzWeights)
#    reconValues = np.average(np.asarray(reconValuesList), axis=0, weights=sample.mzWeights)
#    ERDValues = np.average(np.asarray(ERDValueList), axis=0, weights=sample.mzWeights)

    #Convert ERDValues from 1D to 2D
    ERDValuesNP = makeERDArray(ERDValues, maskObject, reconImage)

    return reconImage, reconValues, ERDValues, ERDValuesNP

def computeFullERD(info, maskObject, measuredValues, reconValues, reconImage, theta, neighborValues, neighborWeights, neighborDistances):

    # Compute features
    polyFeatures = computeFeatures(maskObject.unMeasuredIdxs, maskObject.area, neighborValues, neighborWeights, neighborDistances, info, reconValues, reconImage)

    # Compute ERD
    ERDValues = polyFeatures.dot(theta)

    return ERDValues

def makeERDArray(ERDValues, maskObject, reconImage):
    #Rearrange ERD values into array; those that have already been measured have 0 ERD
    ERDValuesNP = np.zeros([maskObject.height, maskObject.width])

    #Copy over ERD values for unmeasured points
    for i in range(0, len(maskObject.unMeasuredIdxs)): ERDValuesNP[maskObject.unMeasuredIdxs[i][0], maskObject.unMeasuredIdxs[i][1]] = ERDValues[i]

    #Remove values that are less than those already scanned (0 ERD)
    ERDValuesNP[np.where((ERDValuesNP < 0))] = 0

    #Normalize values
    #ERDValuesNP = (ERDValuesNP-np.min(ERDValuesNP))*((255.0-0.0)/(np.max(ERDValuesNP)-np.min(ERDValuesNP)))+0.0

    #Change the values based on the reconImage directly
    #ERDValuesNP[np.where((reconImage == 0)))] = 0

    return ERDValuesNP

def checkStopCondFuncThreshold(stopCondParams, StopCondFuncVal, maskObject, measuredValues, iterNum):

    if stopCondParams.threshold == 0:
        if np.shape(measuredValues)[0] >= round(maskObject.area*stopCondParams.maxPercentage/100):
            return True
        else:
            return False
    else:
        if np.shape(measuredValues)[0] >= round(maskObject.area*stopCondParams.maxPercentage/100):
            return True
        else:
            if np.logical_and(((maskObject.area)*stopCondParams.minPercentage/100)<np.shape(measuredValues)[0], stopCondFuncVal[iterNum,0]<stopCondParams.threshold):
                gradStopCondFunc = np.mean(stopCondFuncVal[iterNum,0]-stopCondFuncVal[iterNum-stopCondParams.JforGradient:iterNum-1,0])
                if gradStopCondFunc < 0:
                    return True
            else:
                return False

def computeStopCondFuncVal(reconValues, measuredValues, stopCondParams, info, stopCondFuncVal, maxIdxsVect, iterNum, maskObject):

    #Calculate the difference in values between the previous reconstruction values against the measured values
    diff = 0
    for i in range(0, len(maxIdxsVect)): diff += computeDifference(reconValues[maxIdxsVect[i]], measuredValues[len(measuredValues)-len(maxIdxsVect)+i], info.imageType)
    diff = diff/len(measuredValues)

    if iterNum == 1:
        stopCondFuncVal[iterNum,0] = stopCondParams.beta*diff
    else:
        stopCondFuncVal[iterNum,0] = ((1-stopCondParams.beta)*stopCondFuncVal[iterNum-1,0] + stopCondParams.beta*diff)

    stopCondFuncVal[iterNum,1] = np.shape(measuredValues)[0]

    return stopCondFuncVal

def findNeighbors(info, measuredIdxs, unMeasuredIdxs):

    neigh = NearestNeighbors(n_neighbors=info.numNeighbors)
    neigh.fit(measuredIdxs)
    neighborDistances, neighborIndices = neigh.kneighbors(unMeasuredIdxs)
    neighborDistances = neighborDistances*info.resolution
    unNormNeighborWeights = 1/np.power(neighborDistances, info.neighborWeightsPower)
    sumOverRow = (np.sum(unNormNeighborWeights, axis=1))
    neighborWeights = unNormNeighborWeights/sumOverRow[:, np.newaxis]

    return neighborIndices, neighborWeights, neighborDistances

def findNeighborValues(measuredValues, neighborIndices):
    return measuredValues[neighborIndices]

def computeRecons(info, maskObject, unMeasuredIdxs, measuredIdxs, neighborValues, neighborWeights, measuredValues):
    reconValues = computeWeightedMRecons(neighborValues, neighborWeights, info)
    reconImage = np.zeros((maskObject.height, maskObject.width))
    reconImage[unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]] = reconValues
    reconImage[measuredIdxs[:,0], measuredIdxs[:,1]] = measuredValues
    return(reconValues, reconImage)

def computeWeightedMRecons(neighborValues, neighborWeights, info):
    if info.featReconMethod=='DWM':
        classLabels = np.unique(neighborValues)
        classWeightSums = np.zeros((np.shape(neighborWeights)[0], np.shape(classLabels)[0]))
        for i in range(0,np.shape(classLabels)[0]):
            tempFeats = np.zeros((np.shape(neighborWeights)[0], np.shape(neighborWeights)[1]))
            np.copyto(tempFeats, neighborWeights)
            tempFeats[neighborValues != classLabels[i]]=0
            classWeightSums[:,i] = np.sum(tempFeats, axis=1)
        reconValues = classLabels[np.argmax(classWeightSums, axis=1)]
    elif info.featReconMethod=='CWM':
        reconValues = np.sum(neighborValues*neighborWeights, axis=1)
    return reconValues

def computeFeatures(unMeasuredIdxs, area, neighborValues, neighborWeights, neighborDistances, info, reconValues, reconImage):
    feature = np.zeros((np.shape(unMeasuredIdxs)[0],7))

    # Compute std div features
    diffVect = computeDifference(neighborValues, np.transpose(np.matlib.repmat(reconValues, np.shape(neighborValues)[1],1)), info.imageType)
    feature[:,0] = np.sum(neighborWeights*diffVect, axis=1)
    feature[:,1] = np.sqrt((1/info.numNeighbors)*np.sum(np.power(diffVect,2),axis=1))

    # Compute distance/density features
    cutoffDist = np.ceil(np.sqrt((info.featDistCutoff/100)*(area/np.pi)))
    feature[:,2] = neighborDistances[:,0]
    neighborsInCircle = np.sum(neighborDistances <= cutoffDist, axis=1)
    feature[:,3] = (1+(np.pi*(np.power(cutoffDist, 2))))/(1+np.sum(neighborDistances <= cutoffDist, axis=1))

    # Compute gradient features
    gradientImageX, gradientImageY = np.gradient(reconImage)

    #Assume continuous features
    gradientImageX = abs(gradientImageX)
    gradientImageY = abs(gradientImageY)
    feature[:,4] = gradientImageY[unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]]
    feature[:,5] = gradientImageX[unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]]

    #NEW Addition; add the actual reconstruction values for the unmeasured positions
    #feature[:,6] = reconImage[unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]]

    #Compute polyfeatures
    polyFeatures = np.hstack([np.ones((np.shape(feature)[0],1)), feature])
    for i in range(0, np.shape(feature)[1]):
        for j in range(i, np.shape(feature)[1]):
            temp = feature[:,i]*feature[:,j]
            polyFeatures = np.column_stack([polyFeatures, feature[:,i]*feature[:,j]])
    return polyFeatures

def computeDifference(array1, array2, imageType):
    if imageType == 'C':
        return abs(array1-array2)
    elif imageType == 'D':
        difference = array1 != array2
        return difference.astype(float)
    else:
        sys.exit('Error! - Unexpected imageType declared')
        return 0

def generateGaussianKernel(sigma, windowSize):
    return np.outer(signal.gaussian(windowSize[0], std=sigma), signal.gaussian(windowSize[1], std=sigma)).flatten()

def percResults(results, perc_testingResults, precision):
    percents = np.arange(min(np.hstack(perc_testingResults)),max(np.hstack(perc_testingResults))+precision, precision)
    averages = []
    finalPercents = []
    for percent in percents:
        values = []
        for i in range(0,len(results)):
            percList = np.array(perc_testingResults[i])
            idx = np.argmin(np.abs(np.asarray(percList)-percent))
            values.append(results[i][idx])
        averageValue = np.mean(values)
        if len(averages) == 0 or averageValue != averages[len(averages)-1]:
            averages.append(np.mean(values))
            finalPercents.append(percent)
    return finalPercents, averages


def performMeasurements(sample, maskObject, newIdxs, simulationFlag):

    #Update the maskObject according to the newIdxs
    maskObject.update(newIdxs)

    #Obtain values from the stored image information
    for imageNum in range(0,len(sample.measuredImages)):
        temp = np.asarray(sample.images[imageNum]).copy()
        temp[maskObject.mask == 0] = 0
        sample.measuredImages[imageNum] = temp.copy()
    return sample, maskObject

def sectionTitle(title):
    print('\n' + ('#' * int(consoleColumns)))
    print(title)
    print(('#' * int(consoleColumns)) + '\n')

#Watch iterator for completion of workers in parallization pool
def parIterator(idens):
    while idens:
        done, idens = ray.wait(idens)
        yield ray.get(done[0])

#==================================================================

