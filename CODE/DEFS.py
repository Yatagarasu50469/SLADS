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
        self.measuredLines = []
        self.avgImage = None

#Trained SLADS Model object
class SLADSModel:
    def __init__(self, massRange, model, cValue):
        self.massRange = massRange
        self.model = model
        self.cValue = cValue

#Singular result generated through runSLADS
class Result():
    def __init__(self, info, sample, avgGroundTruthImage, simulationFlag, animationFlag):
        self.info = info
        self.sample = sample
        self.avgGroundTruthImage = avgGroundTruthImage
        self.simulationFlag = simulationFlag
        self.animationFlag = animationFlag
        self.reconImages = []
        self.masks = []
        self.ERDValueNPs = []
        self.TDList = []
        self.MSEList = []
        self.SSIMList = []
        self.percMeasuredList = []
        self.thresholdList = []
        self.reconMasks = []

    def update(self, threshold, percMeasured, reconImage, reconMask, sample, maskObject, ERDValuesNP, iterNum, completedRunFlag):

        #Save the model development
        self.reconImages.append(reconImage)
        self.masks.append(maskObject.mask.copy())
        self.ERDValueNPs.append(ERDValuesNP.copy())
        self.sample = sample
        self.percMeasuredList.append(percMeasured)
        self.thresholdList.append(threshold)
        self.reconMasks.append(reconMask)

        if self.simulationFlag:

            #Find statistics of interest
            difference = np.sum(computeDifference(self.avgGroundTruthImage, reconImage, self.info.imageType))
            TD = difference/maskObject.area
            MSE = (np.sum((reconImage - self.avgGroundTruthImage) ** 2))/(maskObject.area)
            SSIM = structural_similarity(self.avgGroundTruthImage, reconImage, data_range=reconImage.max() - reconImage.min())

            #Save them for each timestep
            self.TDList.append(TD)
            self.MSEList.append(MSE)
            self.SSIMList.append(SSIM)

        #If an animation will be produced and the run has completed
        if self.animationFlag and completedRunFlag:

            #Setup directory addresses
            dir_mzResults = self.sample.resultsPath + 'mzResults/'
            dir_mzSampleResults = dir_mzResults + self.sample.name + '/'

            dir_Animations = self.sample.resultsPath+ 'Animations/'
            dir_AnimationVideos = dir_Animations + 'Videos/'
            dir_AnimationFrames = dir_Animations + self.sample.name + '/'

            #Clean sub-directories
            if os.path.exists(dir_AnimationFrames): shutil.rmtree(dir_AnimationFrames)
            os.makedirs(dir_AnimationFrames)

            if os.path.exists(dir_mzSampleResults): shutil.rmtree(dir_mzSampleResults)
            os.makedirs(dir_mzSampleResults)

        #If an animation should be produced and the run has completed for a simulation
        if self.animationFlag and completedRunFlag:

            #Normalize values
            self.ERDValueNPs = (self.ERDValueNPs-np.min(self.ERDValueNPs))*((255.0-0.0)/(np.max(self.ERDValueNPs)-np.min(self.ERDValueNPs)))+0.0
            self.thresholdList = np.asarray(self.thresholdList)
            self.thresholdList = (self.thresholdList-np.min(self.thresholdList))*((255.0-0.0)/(np.max(self.thresholdList)-np.min(self.thresholdList)))+0.0

            #If this was a simulation
            if self.simulationFlag:

                #Save each of the individual mass range reconstructions
                percSampled = "{:.2f}".format(self.percMeasuredList[len(self.percMeasuredList)-1])
                    
                #Find the neighbor information for the last mask
                neighborIndices, neighborWeights, neighborDistances = findNeighbors(info, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)
                
                #Retrieve the measured indexes
                idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))

                for massNum in range(0, len(self.sample.massRanges)):
                    
                    #mz image with only the actual measurements made
                    subMeasuredImage = sample.measuredImages[massNum].astype("float")

                    #Determine the measured values for the mz image
                    measuredValues = subMeasuredImage[np.asarray(idxsX), np.asarray(idxsY)]
                    
                    #Compute reconstruction for the specific mz image
                    _, subReconImage = computeRecons(info, maskObject, maskObject.unMeasuredIdxs, maskObject.measuredIdxs, findNeighborValues(measuredValues, neighborIndices), neighborWeights, measuredValues)

                    mzImage = self.sample.images[massNum].astype("float")
                    
                    #SSIM relative to the measured
                    meassureSSIM = "{:.2f}".format(structural_similarity(mzImage, subMeasuredImage, data_range=subMeasuredImage.max() - subMeasuredImage.min()))
                    
                    #SSIM relative to the reconstruction
                    reconSSIM = "{:.2f}".format(structural_similarity(mzImage, subReconImage, data_range=subReconImage.max() - subReconImage.min()))
                    
                    #Mass range string
                    massRange = str(self.sample.massRanges[massNum][0]) + '-' + str(self.sample.massRanges[massNum][1])

                    #Measured mz image
                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(15,5))
                    f.subplots_adjust(top = 0.80)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle('Percent Sampled: ' + percSampled + '  Measurement mz: ' + massRange + '  SSIM: ' + meassureSSIM, fontsize=20, fontweight='bold', y = 0.95)

                    sub = f.add_subplot(1,3,1)
                    sub.imshow(self.masks[len(self.masks)-1], cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')

                    sub = f.add_subplot(1,3,2)
                    sub.imshow(mzImage * 255.0/mzImage.max(), cmap='hot', aspect='auto')
                    sub.set_title('Ground-Truth')

                    sub = f.add_subplot(1,3,3)
                    sub.imshow(subMeasuredImage * 255.0/mzImage.max(), cmap='hot', aspect='auto')
                    sub.set_title('Measured')

                    saveLocation = dir_mzSampleResults + 'measured_' + massRange +'.png'

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()
                    
                    #Reconstructed mz image
                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(15,5))
                    f.subplots_adjust(top = 0.80)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle('Percent Sampled: ' + percSampled + '  Measurement mz: ' + massRange + '  SSIM: ' + reconSSIM, fontsize=20, fontweight='bold', y = 0.95)
                    
                    sub = f.add_subplot(1,3,1)
                    sub.imshow(self.masks[len(self.masks)-1 ], cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')
                    
                    sub = f.add_subplot(1,3,2)
                    sub.imshow(mzImage * 255.0/mzImage.max(), cmap='hot', aspect='auto')
                    sub.set_title('Ground-Truth')
                    
                    sub = f.add_subplot(1,3,3)
                    sub.imshow(subReconImage * 255.0/mzImage.max(), cmap='hot', aspect='auto')
                    sub.set_title('Reconstructed')
                    
                    saveLocation = dir_mzSampleResults + 'recon_' + massRange +'.png'
                    
                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()


                #Generate each of the frames
                for i in range(0, len(self.masks)):

                    thresholdStr = "{:.2f}".format(self.thresholdList[i])
                    
                    #1x3 without ERD
                    #=====================
                    saveLocation = dir_AnimationFrames + 'stretched_1x3_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(15,5))
                    
                    f.subplots_adjust(top = 0.7)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle("Percent Sampled: %.2f, Measurement Iteration: %.0f\nSSIM: %.2f" % (self.percMeasuredList[i], i+1, self.SSIMList[i]), fontsize=20, fontweight='bold', y = 0.95)

                    sub = f.add_subplot(1,3,1)
                    sub.imshow(self.avgGroundTruthImage * 255.0/self.avgGroundTruthImage.max(), cmap='hot', aspect='auto')
                    sub.set_title('Ground-Truth')
            
                    sub = f.add_subplot(1,3,2)
                    sub.imshow(self.reconImages[i] * 255.0/self.avgGroundTruthImage.max(), cmap='hot', aspect='auto')
                    sub.set_title('Reconstructed Image')
                    
                    sub = f.add_subplot(1,3,3)
                    sub.imshow(self.masks[i], cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()
                    #=====================

                    #2x3 with ERD printout
                    #=====================
                    saveLocation = dir_AnimationFrames + 'stretched_2x3_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'

                    f = plt.figure(figsize=(20,10))
                    f.subplots_adjust(top = 0.85)
                    f.subplots_adjust(wspace=0.4, hspace=0.2)
                    plt.suptitle("Percent Sampled: %.2f, Measurement Iteration: %.0f\nSSIM: %.2f" % (self.percMeasuredList[i], i+1, self.SSIMList[i]), fontsize=20, fontweight='bold', y = 0.95)

                    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
                    ax1.imshow(self.avgGroundTruthImage * 255.0/self.avgGroundTruthImage.max(), cmap='hot', aspect='auto')
                    ax1.set_title('Ground-Truth')

                    ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
                    ax2.imshow(self.reconImages[i] * 255.0/self.avgGroundTruthImage.max(), cmap='hot', aspect='auto')
                    ax2.set_title('Reconstructed Image')

                    ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
                    ax3.imshow(self.masks[i], cmap='gray', aspect='auto')
                    ax3.set_title('Sampled Mask')

                    ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
                    im = ax4.imshow(self.ERDValueNPs[i], cmap='viridis', vmin=0, vmax=255, aspect='auto')
                    ax4.set_title('ERD Values')
                    cbar = f.colorbar(im, ax=ax4, orientation='vertical', pad=0.01)

                    ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
                    ax5.imshow(self.reconMasks[i], cmap='gray', aspect='auto')
                    ax5.set_title('Potential Sampling Area')

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()
                    #=====================
                    
                dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'stretched_2x3_*.png'))
                height, width, layers = cv2.imread(dataFileNames[0]).shape
                animation = cv2.VideoWriter(dir_AnimationVideos + 'stretched_2x3_' + self.sample.name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
                for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                animation.release()
                animation = None
        
                dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'stretched_1x3_*.png'))
                height, width, layers = cv2.imread(dataFileNames[0]).shape
                animation = cv2.VideoWriter(dir_AnimationVideos + 'stretched_1x3_' + self.sample.name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
                for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                animation.release()
                animation = None
                    
            else: #Not a simulation
                sys.exit('ERROR! - Non simulation plots not yet fixed for color issue and modified selection')
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
                    plt.suptitle('Percent Sampled: ' + percSampled + '  Measurement mz: ' + massRange, fontsize=20, fontweight='bold', y = 0.95)

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

                    saveLocation = dir_AnimationFrames + 'stretched_1x3_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'

                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(15,5))

                    f.subplots_adjust(top = 0.80)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle("Percent Sampled: %.2f, Measurement Iteration: %.0f" % (self.percMeasuredList[i], i+1), fontsize=20, fontweight='bold', y = 0.95)

                    sub = f.add_subplot(1,3,1)
                    sub.imshow(self.reconImages[i] * 255.0/self.reconImages[i].max(), cmap='hot', aspect='auto')
                    sub.set_title('Reconstructed Image')

                    sub = f.add_subplot(1,3,2)
                    sub.imshow(self.masks[i], cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')

                    sub = f.add_subplot(1,3,3)
                    #im = sub.imshow(self.ERDValueNPs[i]>0, cmap='gray', aspect='auto')
                    #sub.set_title('ERD Values > 0')
                    im = sub.imshow(self.ERDValueNPs[i], cmap='viridis', vmin=0, vmax=255, aspect='auto')
                    sub.set_title('ERD Values')
                    cbar = f.colorbar(im, ax=sub, orientation='vertical', pad=0.01)

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()

            dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'stretched_1x3_*.png'))
            height, width, layers = cv2.imread(dataFileNames[0]).shape
            animation = cv2.VideoWriter(dir_AnimationVideos + 'stretched_1x3_' + self.sample.name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
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
    def __init__(self, width, height, measurementPercs, numMasks):
        self.width = width
        self.height = height
        self.area = width*height
        self.percMasks = []
        self.measuredIdxs = []
        self.unMeasuredIdxs = []
        self.initialMeasuredIdxs = []
        self.initialUnMeasuredIdxs = []
        self.unMeasuredIdxsList = []
        self.measuredIdxsList = []
        self.initialSets = []

        #Create a blank initial mask
        self.initialMask = np.zeros([height, width])
        
        #If scanning is to be performed line by line
        if scanMethod == 'linewise':
        
            #Generate a list of arrays (lines) contianing the x,y points that need to be scanned
            self.linesToScan = []
            for rowNum in np.arange(0,height,1):
                line = []
                for columnNum in np.arange(0, width, 1):
                    line.append(tuple([rowNum, columnNum]))
                self.linesToScan.append(line)

            #Fix internel format, so that it is consistent throughout program operation
            self.linesToScan = np.asarray(self.linesToScan).tolist()

            #Copy the initial set of linesToScan
            self.originalLinesToScan = copy.copy(self.linesToScan)
        
            #Set which lines should be acquired in initial scan
            lineIndexes = [
                int((height-1)*0.10),
                int((height-1)*0.50),
                int((height-1)*0.90)
            ]
        
            #Obtain the points in the specified lines and add them to the initial scan list
            for lineIndexNum in range(0, len(lineIndexes)):
                lineIndex = lineIndexes[lineIndexNum]
                
                #Setup the initial mask based on the points specified in the chosen initial lines
                for pt in self.linesToScan[lineIndex]: self.initialMask[tuple(pt)] = 1
                
                #Add start and end point information for equipment implementation
                self.initialSets.append([self.linesToScan[lineIndex][0], self.linesToScan[lineIndex][len(self.linesToScan[lineIndex])-1]])

            #Now delete the lines/points specified from the remaining potentials; in case of shared points between 'lines' (diagonals, spirals, etc.)
            for lineIndexNum in range(0, len(lineIndexes)): self.delLine(lineIndexes[lineIndexNum]-lineIndexNum)
            
            #Make a duplicate for fresh initializations
            self.initialLinesToScan = copy.copy(self.linesToScan)
        
        #If scanning is to be performed on a point by point basis
        elif scanMethod == 'pointwise':
            if impModel: sys.exit('Error! - pointwise scanning method has not been setup for physical model implementation!')
            
            #Generate a random initial mask that scans 1% of sample pixels 
            self.initialMask = np.random.rand(height, width) <= (1/100)
        
        #Store the initially measured and unmeasured mask locations for the first measurement step
        self.initialMeasuredIdxs = np.transpose(np.where(self.initialMask == 1))
        self.initialUnMeasuredIdxs = np.transpose(np.where(self.initialMask == 0))
        
        #CURRENTLY BEING USED FOR BOTH LINE AND POINTWISE SCANNING METHODS - SHOULD BE ALTERED FOR RANDOM PARTIAL LINES FOR LINEWISE
        #Create random initial percentage masks using point measurements
        for percNum in range(0, len(measurementPercs)):
            self.percMasks.append([])
            self.measuredIdxsList.append([])
            self.unMeasuredIdxsList.append([])
            for maskNum in range(0, numMasks):
                self.mask = np.zeros([height, width])
                self.mask = np.random.rand(height, width) < (measurementPercs[percNum]/100)
                self.percMasks[percNum].append(self.mask)
                self.measuredIdxsList[percNum].append(np.transpose(np.where(self.mask == 1)))
                self.unMeasuredIdxsList[percNum].append(np.transpose(np.where(self.mask == 0)))


    #Update the mask given a set of new measurement locations
    def update(self, newIdxs):
        for pt in newIdxs: self.mask[tuple(pt)] = 1
        self.measuredIdxs = np.transpose(np.where(self.mask == 1))
        self.unMeasuredIdxs = np.transpose(np.where(self.mask == 0))

    #Reset the training sample's mask and linesToScan
    def reset(self, simulationFlag):
        #If this is a simulation, then reset to blank mask (no initial scan)
        if simulationFlag:
            self.mask = np.zeros([self.height, self.width])
            if scanMethod == 'linewise': self.linesToScan = copy.copy(self.originalLinesToScan)
            self.measuredIdxs = np.transpose(np.where(self.mask == 1))
            self.unMeasuredIdxs = np.transpose(np.where(self.mask == 0))
        else: #Reset to state after initial measurements have been made
            self.mask = self.initialMask
            self.measuredIdxs = np.transpose(np.where(self.mask == 1))
            self.unMeasuredIdxs = np.transpose(np.where(self.mask == 0))

    def delLine(self, index):
        if scanMethod == 'linewise':
            self.linesToScan = np.delete(self.linesToScan, index, 0).tolist()
        else:
            sys.exit('Error! - Attempted to delete a line with non-linewise scanning method')

    def delPoints(self, pts):
        if scanMethod == 'linewise':
            for lineNum in range(0, len(self.linesToScan)): self.linesToScan[lineNum] = [pt for ix, pt in enumerate(self.linesToScan[lineNum]) if pt not in pts]
            self.linesToScan = [x for x in self.linesToScan if x]
        else:
            sys.exit('Error! - Attempted to delete points from lines with non-linewise scanning method')



def runSLADS(info, samples, model, stopPerc, sampleNum, simulationFlag, trainPlotFlag, animationFlag, tqdmHide, bestCFlag):

    if simulationFlag: #Here sample.images contains the full ground-truth images
        sample = samples[sampleNum]
        avgGroundTruthImage = np.average(sample.images, axis=0, weights=sample.mzWeights)
    else: #Here sample.images contains only the initially measured ground-truth images
        sample = samples
        sample.measuredImages = sample.images
        avgGroundTruthImage = []

    maskObject = sample.maskObject

    #Reinitialize the mask state to starting state
    maskObject.reset(simulationFlag)

    #Has the stopping condition been met yet
    completedRunFlag = False

    #Current iteration
    iterNum = 1

    #Assume variable Classify=='N' (Artifact of pointwise SLADS)

    #Initialize stopping condition object
    stopCondParams = StopCondParams(maskObject.area, 0, 50, 2, stopPerc)

    #Determine stoppingCondition function value
    stopCondFuncVal = np.zeros((int((maskObject.area)*(stopCondParams.maxPercentage)/100)+10,2))

    #Perform the initial measurements
    if simulationFlag: sample, maskObject = performMeasurements(iterNum, sample, maskObject, maskObject.initialMeasuredIdxs, simulationFlag)

    #Perform initial reconstruction and ERD calculation
    sample, measuredValues, reconImage, reconMask, reconValues, ERDValuesNP = avgReconAndERD(sample, info, iterNum, maskObject, model, reconImage=None, reconValues=None, ERDValuesNP=None, newIdxs=None)

    #Determine percentage pixels measured initially
    percMeasured = (np.sum(maskObject.mask)/maskObject.area)*100

    #Check for completion state here just in case, prior to loop!
    completedRunFlag = checkStopCondFuncThreshold(stopCondParams, stopCondFuncVal, maskObject, measuredValues, iterNum)

    #Additional stopping condition for if there are no more linesToScan
    if scanMethod == 'linewise':
        if len(maskObject.linesToScan) == 0: completedRunFlag = True

    #Initialize a result object
    result = Result(info, sample, avgGroundTruthImage, simulationFlag, animationFlag)
    result.update(0, percMeasured, reconImage, reconMask, sample, maskObject, ERDValuesNP, iterNum, completedRunFlag)

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
            oldReconImage = reconImage.copy()
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Find next measurement locations
            maskObject, newIdxs, threshold = findNewMeasurementIdxs(info, maskObject, sample, model, reconValues, reconImage, reconMask, ERDValuesNP)
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Perform measurements
            sample, maskObject = performMeasurements(iterNum, sample, maskObject, newIdxs, simulationFlag)
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Perform reconstruction and ERD calculation
            sample, measuredValues, reconImage, reconMask, reconValues, ERDValuesNP = avgReconAndERD(sample, info, iterNum, maskObject, model, reconImage, reconValues, ERDValuesNP, newIdxs)
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Update the percentage of pixels that have beene measured
            percMeasured = (np.sum(maskObject.mask)/maskObject.area)*100
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Evaluate the stop condition value
            stopCondFuncVal = computeStopCondFuncVal(oldReconImage, reconImage, stopCondParams, info, stopCondFuncVal, newIdxs, iterNum, maskObject)
            timesList.append(round(time.time()-t0,10))

            t0 = time.time()
            #Check the stopping condition
            completedRunFlag = checkStopCondFuncThreshold(stopCondParams, stopCondFuncVal, maskObject, measuredValues, iterNum)
            timesList.append(round(time.time()-t0,10))

            t0 = time.time()
            #Additional stopping condition for if there are no more linesToScan
            if scanMethod == 'linewise':
                if len(maskObject.linesToScan) == 0: completedRunFlag = True
            timesList.append(round(time.time()-t0,10))

            t0 = time.time()
            #Store information to the resultsObject
            result.update(threshold, percMeasured, reconImage, reconMask, sample, maskObject, ERDValuesNP, iterNum, completedRunFlag)
            timesList.append(round(time.time()-t0,10))

            #Update the progress bar
            pbar.n = round(percMeasured,2)
            pbar.refresh()

            #print(timesList)
    
    if bestCFlag: return np.trapz(result.TDList, result.percMeasuredList)

    return result

def findNewMeasurementIdxs(info, maskObject, sample, model, reconValues, reconImage, reconMask, ERDValuesNP):

    #Make sure ERDValuesNP is in np array
    ERDValuesNP = np.asarray(ERDValuesNP)

    #==========================================
    #CANNY CONVEX HULL PART 1
    #==========================================
    #Mask ERD values by the reconstruction mask
    if ERDMaskingFlag: ERDValuesNP*=reconMask
    #==========================================

    if scanMethod == 'pointwise':
        ERDValueList = [ERDValuesNP[tuple(pt)] for pt in maskObject.unMeasuredIdxs]
        newIdxs = maskObject.unMeasuredIdxs[np.argsort(ERDValueList)][::-1][:int((1/100)*maskObject.area)]

    elif scanMethod == 'linewise':
        #==========================================
        #OPTIMAL LINE DETERMINATION
        #==========================================

        #Sum ERD for all lines
        lineERDSums = [np.nansum(ERDValuesNP[tuple([x[0] for x in line]), tuple([y[1] for y in line])]) for line in maskObject.linesToScan]

        #Choose the line with maximum ERD and extract the actual indices
        lineToScanIdx = np.nanargmax(lineERDSums)
        lineToScanIdxs = maskObject.linesToScan[lineToScanIdx]

        #Set the default new Idxs to be all of the indexes identified
        newIdxs = lineToScanIdxs.copy()
        #==========================================

        #==========================================
        #CANNY CONVEX HULL PART 2
        #==========================================
        #Limit new idxs according to which are inside of the reconMask
        if ERDMaskingFlag: newIdxs = [pos for pos in newIdxs if (reconMask[tuple(pos)] >= 1)]
        #==========================================
        
        #==========================================
        #THRESHOLD DETERMINATION
        #==========================================
        #Set threshold for what ERD Values; mean of the chosen points' ERD values
        if meanThresholdFlag:
            threshold = lineERDSums[lineToScanIdx]/len(newIdxs)
        else:
            threshold = 0
        #==========================================

        #==========================================
        #PARTIAL LINE BY THRESHOLD
        #==========================================
        #If partial lines, narrow the possible points for selection by the given threshold
        if partialLineFlag and (lineERDSums[lineToScanIdx] > 0): newIdxs = [pos for pos in newIdxs if ERDValuesNP[tuple(pos)] >= threshold]
        #==========================================

        #==========================================
        #START/END POINTS BY THRESHOLD
        #==========================================
        #If not partial lines, select all points between the first and last position with an ERD above threshold
        if (not partialLineFlag) and (len(newIdxs) > 1): newIdxs = lineToScanIdxs[lineToScanIdxs.index(newIdxs[0]):lineToScanIdxs.index(newIdxs[len(newIdxs)-1])]
        #==========================================

        #==========================================
        #SELECTION SAFEGUARD
        #==========================================
        #If there are not enough locations selected, just scan the whole remainder of the line with the greatest ERD; ensures model will reach termination
        if len(newIdxs) <= (0.01*maskObject.width): newIdxs = np.asarray(lineToScanIdxs).tolist()

        #==========================================
        #POINT CONSIDERATION UPDATE
        #==========================================
        #Remove the selected points from further consideration, allows revisting lines
        if lineRevistFlag:
            maskObject.delPoints(newIdxs)
        else:
            #Remove the line selected from further consideration, does not allow revisiting
            maskObject.delLine(lineToScanIdx)
        #==========================================

    return maskObject, newIdxs, threshold

def avgReconAndERD(sample, info, iterNum, maskObject, model, reconImage, reconValues, ERDValuesNP, newIdxs):

    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(info, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)
    
    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
    
    #Perform weighted averaging of the multiple channels
    sample.avgImage = np.average(np.asarray(sample.measuredImages), axis=0, weights=sample.mzWeights)
    
    measuredValues = sample.avgImage[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Find neighborhood values
    neighborValues = findNeighborValues(measuredValues, neighborIndices)
    
    #Compute reconstructions
    reconValues, reconImage = computeRecons(info, maskObject, maskObject.unMeasuredIdxs, maskObject.measuredIdxs, neighborValues, neighborWeights, measuredValues)
    
    #Compute full ERD Values
    ERDValuesNP = computeERD(info, maskObject, measuredValues, reconValues, reconImage, model, neighborValues, neighborWeights, neighborDistances)

    #Make a copy of the current recon image
    reconMask = reconImage.copy()
    
    #Remove values from consideration outside of 3 sigma +/- mean
    reconMask*=(reconMask >= np.mean(reconMask)-3*np.std(reconMask)) & (reconMask <= np.mean(reconMask)+3*np.std(reconMask))

    #Transition to uint8 format for Canny
    reconMask = reconMask * 255.0/reconMask.max()
    reconMask = reconMask.astype('uint8')

    #Perform canny on the reconstruction
    idxsX, idxsY = np.where(cv2.Canny(reconMask, 128, 255, 3) > 0)

    #Create a convex hull from the points
    hull = cv2.convexHull(np.asarray([[i] for i in list(map(list, zip(idxsY, idxsX)))])).astype('int32')

    #Create final mask overlay
    reconMask = cv2.drawContours(np.zeros((reconMask.shape[0], reconMask.shape[1]), dtype=np.uint8), [hull], -1, (True), thickness=-1)

    return sample, measuredValues, reconImage, reconMask, reconValues, ERDValuesNP

def computeERD(info, maskObject, measuredValues, reconValues, reconImage, model, neighborValues, neighborWeights, neighborDistances):

    # Compute features
    polyFeatures = computeFeatures(maskObject.unMeasuredIdxs, maskObject.area, neighborValues, neighborWeights, neighborDistances, info, reconValues, reconImage)

    # Compute ERD
    ERDValues = model.predict(polyFeatures)
    
    #Rearrange ERD values into array; those that have already been measured have 0 ERD
    ERDValuesNP = np.zeros([maskObject.height, maskObject.width])
    
    #Copy over ERD values for unmeasured points
    for i in range(0, len(maskObject.unMeasuredIdxs)): ERDValuesNP[maskObject.unMeasuredIdxs[i][0], maskObject.unMeasuredIdxs[i][1]] = ERDValues[i]
    
    #Remove values that are less than those already scanned (0 ERD)
    ERDValuesNP[np.where((ERDValuesNP < 0))] = 0
    
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

def computeStopCondFuncVal(oldReconImage, reconImage, stopCondParams, info, stopCondFuncVal, newIdxs, iterNum, maskObject):
    
    #Calculate the average difference in values between the previous reconValues against their actual measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in newIdxs]))
    diff = np.mean(computeDifference(reconImage[np.asarray(idxsX), np.asarray(idxsY)], oldReconImage[np.asarray(idxsX), np.asarray(idxsY)], info.imageType))
    
    if iterNum == 1:
        stopCondFuncVal[iterNum] = stopCondParams.beta*diff
    else:
        stopCondFuncVal[iterNum] = ((1-stopCondParams.beta)*stopCondFuncVal[iterNum-1] + stopCondParams.beta*diff)
    
    return stopCondFuncVal

def findNeighbors(info, measuredIdxs, unMeasuredIdxs):

    neigh = NearestNeighbors()
    neigh.fit(measuredIdxs)
    neighborDistances, neighborIndices = neigh.kneighbors(unMeasuredIdxs, n_neighbors=info.numNeighbors)
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
    feature = np.zeros((np.shape(unMeasuredIdxs)[0],6))

    # Compute std div features
    diffVect = computeDifference(neighborValues, np.transpose(np.matlib.repmat(reconValues, np.shape(neighborValues)[1],1)), info.imageType)
    feature[:,0] = np.sum(neighborWeights*diffVect, axis=1)
    feature[:,1] = np.sqrt(np.sum(np.power(diffVect,2),axis=1))

    # Compute distance/density features
    cutoffDist = np.ceil(np.sqrt((info.featDistCutoff/100)*(area/np.pi)))
    feature[:,2] = neighborDistances[:,0]
    feature[:,3] = (1+(np.pi*(np.power(cutoffDist, 2))))/(1+np.sum(neighborDistances <= cutoffDist))

    # Compute gradient features
    gradientImageX, gradientImageY = np.gradient(reconImage)

    #Assume continuous features
    gradientImageX = abs(gradientImageX)
    gradientImageY = abs(gradientImageY)
    feature[:,4] = gradientImageY[unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]]
    feature[:,5] = gradientImageX[unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]]

    #Fit features to polynomial of degree 2
    polyFeatures = PolynomialFeatures(degree=2).fit_transform(feature)

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

def performMeasurements(iterNum, sample, maskObject, newIdxs, simulationFlag):

    #Update the maskObject according to the newIdxs
    maskObject.update(newIdxs)

    #If this is not a simulation then inform equipment what points to scan, wait, read in new data, update images in sample
    if not simulationFlag:
        with open('./INPUT/IMP/UNLOCK', 'w') as filehandle: filehandle.writelines(str(tuple(newIdxs[0])) + ', ' + str(tuple(newIdxs[len(newIdxs)-1])))
        equipWait()
        images, massRanges = readScanData()
        sample.images = images
        sample.measuredImages = images
    else:
        #Obtain masked values from the stored image information
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

#Convert bytes into human readable format
def sizeFunc(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0: return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

#==================================================================

