#==================================================================
#SLADS DEFINITIONS GENERAL
#==================================================================

#General information regarding samples, used for testing and best C value determination
class Sample:
    def __init__(self, name, images, originalImages, massRanges, maskObject, mzWeights, resultsPath):
        self.name = name
        self.images = images
        self.originalImages = originalImages
        self.massRanges = massRanges
        self.maskObject = maskObject
        self.mzWeights = mzWeights
        self.measuredImages = [np.zeros([maskObject.physHeight, maskObject.physWidth]) for rangeNum in range(0,len(massRanges))]
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
    def __init__(self, info, sample, maskObject, avgGroundTruthImage, simulationFlag, animationFlag):
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

    def update(self, threshold, percMeasured, reconImage, sample, maskObject, ERDValuesNP, iterNum, completedRunFlag):

        #Save the model development
        if physResize:
            reconImage = cv2.resize(reconImage, (maskObject.imageWidth, maskObject.imageHeight), interpolation = cv2.INTER_LINEAR)
            self.reconImages.append(reconImage)
            self.masks.append(cv2.resize(maskObject.mask.copy(), (maskObject.imageWidth, maskObject.imageHeight), interpolation = cv2.INTER_LINEAR))
            self.ERDValueNPs.append(cv2.resize(ERDValuesNP.copy(), (maskObject.imageWidth, maskObject.imageHeight), interpolation = cv2.INTER_LINEAR))
            self.sample = sample
            self.percMeasuredList.append(percMeasured)
            self.thresholdList.append(threshold)
        else:
            self.reconImages.append(reconImage)
            self.masks.append(maskObject.mask.copy())
            self.ERDValueNPs.append(ERDValuesNP.copy())
            self.sample = sample
            self.percMeasuredList.append(percMeasured)
            self.thresholdList.append(threshold)

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

            #Normalize ERD values
            #self.ERDValueNPs = (self.ERDValueNPs-np.min(self.ERDValueNPs))*((255.0-0.0)/(np.max(self.ERDValueNPs)-np.min(self.ERDValueNPs)))+0.0
            
            self.thresholdList = np.asarray(self.thresholdList)
            
            #Normalize threshold values to match with normalized ERD values
            #self.thresholdList = (self.thresholdList-np.min(self.thresholdList))*((255.0-0.0)/(np.max(self.thresholdList)-np.min(self.thresholdList)))+0.0

            #If this was a simulation
            if self.simulationFlag:

                #Save each of the individual mass range reconstructions
                percSampled = "{:.2f}".format(self.percMeasuredList[len(self.percMeasuredList)-1])
                
                #Get reconstructions for each mz image
                mzRecons = computeRecons(info, sample, maskObject, False)

                for massNum in range(0, len(self.sample.massRanges)):
                    
                    #mz image with only the actual measurements made
                    subMeasuredImage = sample.measuredImages[massNum].astype("float")
                    
                    #Retreive reconstruction for the specific mz image
                    subReconImage = mzRecons[massNum]

                    #Retreive ground truth for the specific mz image
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

                    #2x2 with ERD printout
                    #=====================
                    saveLocation = dir_AnimationFrames + 'stretched_2x2_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'

                    f = plt.figure(figsize=(15,10))
                    f.subplots_adjust(top = 0.85)
                    f.subplots_adjust(wspace=0.2, hspace=0.2)
                    plt.suptitle("Percent Sampled: %.2f, Measurement Iteration: %.0f\nSSIM: %.2f" % (self.percMeasuredList[i], i+1, self.SSIMList[i]), fontsize=20, fontweight='bold', y = 0.95)

                    ax1 = plt.subplot2grid(shape=(2,2), loc=(0,0))
                    ax1.imshow(self.avgGroundTruthImage * 255.0/self.avgGroundTruthImage.max(), cmap='hot', aspect='auto')
                    ax1.set_title('Ground-Truth')

                    ax2 = plt.subplot2grid((2,2), (0,1))
                    ax2.imshow(self.reconImages[i] * 255.0/self.avgGroundTruthImage.max(), cmap='hot', aspect='auto')
                    ax2.set_title('Reconstructed Image')

                    ax3 = plt.subplot2grid((2,2), (1,0))
                    ax3.imshow(self.masks[i], cmap='gray', aspect='auto')
                    ax3.set_title('Sampled Mask')

                    ax4 = plt.subplot2grid((2,2), (1,1))
                    #vmin=0, vmax=255,
                    im = ax4.imshow(self.ERDValueNPs[i], cmap='viridis', vmin=np.min(self.ERDValueNPs), vmax=np.max(self.ERDValueNPs), aspect='auto')
                    ax4.set_title('ERD Values')
                    cbar = f.colorbar(im, ax=ax4, orientation='vertical', pad=0.01)

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()
                    #=====================
                    
                dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'stretched_2x2_*.png'))
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
                
                #Save the averaged ground truth, no borders
                saveLocation = dir_AnimationFrames + 'final_groundTruth_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.avgGroundTruthImage * 255.0/self.avgGroundTruthImage.max(), cmap='hot', aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                
                #Save the averaged final reconstruction, no borders
                saveLocation = dir_AnimationFrames + 'final_reconstruction_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.reconImages[i] * 255.0/self.avgGroundTruthImage.max(), cmap='hot', aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                
                #Save the final mask, no borders
                saveLocation = dir_AnimationFrames + 'final_mask_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.masks[i], cmap='gray', aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                
            
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
    def __init__(self, imageWidth, imageHeight, physWidth, physHeight, measurementPercs, numMasks):
        
        #If physical resize is disabled, then the imageWidth/Height will match with physWidth/Height
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.physWidth = physWidth
        self.physHeight = physHeight
        self.physAspect = physWidth/physHeight if physWidth>physHeight else physHeight/physWidth
        self.area = physWidth*physHeight
        
        self.percMasks = []
        self.measuredIdxs = []
        self.unMeasuredIdxs = []
        self.initialMeasuredIdxs = []
        self.initialUnMeasuredIdxs = []
        self.unMeasuredIdxsList = []
        self.measuredIdxsList = []
        self.initialSets = []

        #Create a blank initial mask
        self.initialMask = np.zeros([physHeight, physWidth])
        
        #If scanning is to be performed line by line
        if scanMethod == 'linewise':
        
            #Generate a list of arrays (lines) contianing the x,y points that need to be scanned
            self.linesToScan = []
            for rowNum in np.arange(0,physHeight,1):
                line = []
                for columnNum in np.arange(0, physWidth, 1):
                    line.append(tuple([rowNum, columnNum]))
                self.linesToScan.append(line)

            #Fix internel format, so that it is consistent throughout program operation
            self.linesToScan = np.asarray(self.linesToScan).tolist()

            #Copy the initial set of linesToScan
            self.originalLinesToScan = copy.copy(self.linesToScan)
        
            #Set which lines should be acquired in initial scan
            lineIndexes = [
                int((physHeight-1)*0.10),
                int((physHeight-1)*0.50),
                int((physHeight-1)*0.90)
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
            self.initialMask = np.random.rand(physHeight, physWidth) <= (1/100)
        
        #Store the initially measured and unmeasured mask locations for the first measurement step
        self.initialMeasuredIdxs = np.transpose(np.where(self.initialMask == 1))
        self.initialUnMeasuredIdxs = np.transpose(np.where(self.initialMask == 0))
        
        #CURRENTLY BEING USED FOR BOTH LINE AND POINTWISE SCANNING METHODS - SHOULD POSSIBLY BE ALTERED FOR RANDOM PARTIAL LINES FOR LINEWISE
        #Create random initial percentage masks using point measurements
        for percNum in range(0, len(measurementPercs)):
            self.percMasks.append([])
            self.measuredIdxsList.append([])
            self.unMeasuredIdxsList.append([])
            for maskNum in range(0, numMasks):
                self.mask = np.zeros([physHeight, physWidth])
                self.mask = np.random.rand(physHeight, physWidth) <= (measurementPercs[percNum]/100)
                self.percMasks[percNum].append(self.mask)
                self.measuredIdxsList[percNum].append(np.transpose(np.where(self.mask == 1)))
                self.unMeasuredIdxsList[percNum].append(np.transpose(np.where(self.mask == 0)))


    #Update the mask given a set of new measurement locations
    def update(self, newIdxs):
        for pt in newIdxs: self.mask[tuple(pt)] = 1
        
        #COMPUTATIONAL FIX
        #Should change this such that we add new Idxs to the list, rather than going through the whole list again
        self.measuredIdxs = np.transpose(np.where(self.mask == 1))
        self.unMeasuredIdxs = np.transpose(np.where(self.mask == 0))

    #Reset the training sample's mask and linesToScan
    def reset(self, simulationFlag):
        #If this is a simulation, then reset to blank mask (no initial scan)
        if simulationFlag:
            self.mask = np.zeros([self.physHeight, self.physWidth])
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
        avgGroundTruthImage = np.average(sample.originalImages, axis=0, weights=sample.mzWeights)
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
    sample, reconImage, ERDValuesNP = avgReconAndERD(sample, info, iterNum, maskObject, model, newIdxs=None)

    #Determine percentage pixels measured initially
    percMeasured = (np.sum(maskObject.mask)/maskObject.area)*100

    #Check for completion state here just in case, prior to loop!
    completedRunFlag = checkStopCondFuncThreshold(stopCondParams, stopCondFuncVal, maskObject, sample, iterNum)

    #Additional stopping condition for if there are no more linesToScan
    if scanMethod == 'linewise':
        if len(maskObject.linesToScan) == 0: completedRunFlag = True

    #Initialize a result object
    result = Result(info, sample, maskObject, avgGroundTruthImage, simulationFlag, animationFlag)
    result.update(0, percMeasured, reconImage, sample, maskObject, ERDValuesNP, iterNum, completedRunFlag)

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
            #Make a duplicate of the ReconImage for stop condition gradient test
            oldReconImage = reconImage.copy()
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Find next measurement locations
            maskObject, newIdxs, threshold = findNewMeasurementIdxs(info, maskObject, sample, model, reconImage, ERDValuesNP)
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Perform measurements
            sample, maskObject = performMeasurements(iterNum, sample, maskObject, newIdxs, simulationFlag)
            timesList.append(round(time.time()-t0,10))
            
            t0 = time.time()
            #Perform reconstruction and ERD calculation
            sample, reconImage, ERDValuesNP = avgReconAndERD(sample, info, iterNum, maskObject, model, newIdxs)
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
            completedRunFlag = checkStopCondFuncThreshold(stopCondParams, stopCondFuncVal, maskObject, sample, iterNum)
            timesList.append(round(time.time()-t0,10))

            t0 = time.time()
            #Additional stopping condition for if there are no more linesToScan
            if scanMethod == 'linewise':
                if len(maskObject.linesToScan) == 0: completedRunFlag = True
            timesList.append(round(time.time()-t0,10))
    
            t0 = time.time()
            #Store information to the resultsObject
            result.update(threshold, percMeasured, reconImage, sample, maskObject, ERDValuesNP, iterNum, completedRunFlag)
            timesList.append(round(time.time()-t0,10))

            #Update the progress bar
            pbar.n = round(percMeasured,2)
            pbar.refresh()

            #print(timesList)
    
    if bestCFlag: return np.trapz(result.TDList, result.percMeasuredList)

    return result

def findNewMeasurementIdxs(info, maskObject, sample, model, reconImage, ERDValuesNP):

    #Make sure ERDValuesNP is in np array
    ERDValuesNP = np.asarray(ERDValuesNP)
    
    #Set a default threshold
    threshold = 0

    if scanMethod == 'pointwise':
        
        #Obtain a list of all ERD values for unmeasured locations
        ERDValueList = [ERDValuesNP[tuple(pt)] for pt in maskObject.unMeasuredIdxs]
        
        #Sort the values in reverse order and choose the top 1% of values
        newIdxs = maskObject.unMeasuredIdxs[np.argsort(ERDValueList)][::-1][:int(0.01*maskObject.area)]

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
        #THRESHOLD DETERMINATION
        #==========================================
        if lineMethod == 'meanThreshold':
            #Change threshold for ERD Values; mean of the chosen line's ERD values
            threshold = lineERDSums[lineToScanIdx]/len(newIdxs)

        #==========================================

        #==========================================
        #PERCENT AND THRESHOLD
        #==========================================
        if lineMethod == 'percLine':
            #Obtain a list of the ERD values in the chosen line
            lineERDValues = [ERDValuesNP[tuple(pt)] for pt in newIdxs]
            
            #Choose the top 50% of points on that line to scan
            newIdxs = np.asarray(newIdxs)[np.argsort(lineERDValues)][::-1][:int(0.5*len(newIdxs))]
            newIdxs = newIdxs.tolist()

            #Remove any points that are equal to the minimum ERD in the image
            #newIdxs = [pos for pos in newIdxs if ERDValuesNP[tuple(pos)] <= np.min(ERDValuesNP)]

        #==========================================
        #PARTIAL LINE BY THRESHOLD
        #==========================================
        #Narrow the possible points for selection by the given threshold
        if lineERDSums[lineToScanIdx] > 0: newIdxs = [pos for pos in newIdxs if ERDValuesNP[tuple(pos)] >= threshold]
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
        if len(newIdxs) <= (0.01*maskObject.physWidth): newIdxs = np.asarray(lineToScanIdxs).tolist()

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

def avgReconAndERD(sample, info, iterNum, maskObject, model, newIdxs):
    
    #Compute reconstructions
    reconImage = computeRecons(info, sample, maskObject, True)

    #Compute full ERD Values
    ERDValuesNP = computeERD(info, sample, maskObject, reconImage, model)
    
    return sample, reconImage, ERDValuesNP

def computeERD(info, sample, maskObject, reconImage, model):

    # Compute features for each of the mz images
    polyFeatures = computeFeatures(maskObject, sample, info, reconImage)

    # Compute ERD
    ERDValues = model.predict(polyFeatures)
    
    #Rearrange ERD values into array; those that have already been measured have 0 ERD
    ERDValuesNP = np.zeros([maskObject.physHeight, maskObject.physWidth])
    for i in range(0, len(maskObject.unMeasuredIdxs)): ERDValuesNP[maskObject.unMeasuredIdxs[i][0], maskObject.unMeasuredIdxs[i][1]] = ERDValues[i]
    
    #Remove values that are less than those already scanned (0 ERD)
    ERDValuesNP[np.where((ERDValuesNP < 0))] = 0
    
    return ERDValuesNP

def checkStopCondFuncThreshold(stopCondParams, StopCondFuncVal, maskObject, sample, iterNum):

    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
    measuredValues = sample.avgImage[np.asarray(idxsX), np.asarray(idxsY)]

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

def findNeighbors(info, maskObject, measuredIdxs, unMeasuredIdxs):
    neigh = NearestNeighbors(n_neighbors=info.numNeighbors, algorithm=algorithmNN, metric='asym', aspect=maskObject.physAspect)
    neigh.fit(measuredIdxs)
    neighborDistances, neighborIndices = neigh.kneighbors(unMeasuredIdxs)
    neighborDistances = neighborDistances*info.resolution
    unNormNeighborWeights = 1/np.power(neighborDistances, info.neighborWeightsPower)
    sumOverRow = (np.sum(unNormNeighborWeights, axis=1))
    neighborWeights = unNormNeighborWeights/sumOverRow[:, np.newaxis]

    return neighborIndices, neighborWeights, neighborDistances

def performRecon(inputImage, maskObject, neighborIndices, neighborWeights, neighborDistances):

    #Create a blank image for the reconstruction
    reconImage = np.zeros((maskObject.physHeight, maskObject.physWidth))
    
    #Retreive measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
    measuredValues = inputImage[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Find neighbor values
    neighborValues = measuredValues[neighborIndices]

    #Compute and save reconstruction values
    reconImage[maskObject.unMeasuredIdxs[:,0], maskObject.unMeasuredIdxs[:,1]] = np.sum(neighborValues*neighborWeights, axis=1)

    #Combine with measured values to form full reconstruction
    reconImage[maskObject.measuredIdxs[:,0], maskObject.measuredIdxs[:,1]] = measuredValues

    return reconImage

def computeRecons(info, sample, maskObject, avgRecon):

    #Find neighbor information for the resized point locations
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(info, maskObject, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)

    #If average reconstruction
    if avgRecon:
        
        #Perform weighted averaging of the multiple channels at original dimensionality, save to global object (used in computeFeatures)
        sample.avgImage = np.average(np.asarray(sample.measuredImages), axis=0, weights=sample.mzWeights)
        reconImage = performRecon(sample.avgImage, maskObject, neighborIndices, neighborWeights, neighborDistances)

        #Return the reconstruction
        return reconImage

    else:
        #Create list to hold individual mz reconstructions
        mzImages = []
        
        #For each mz range
        for mzImage in sample.measuredImages:
            
            #Perform reconstruction and append to list
            mzImages.append(performRecon(mzImage, maskObject, neighborIndices, neighborWeights, neighborDistances))
        
        #Return the list of mz reconstructions
        return mzImages

def normalize(feature):
    return (feature-np.min(feature))*((1.0-0.0)/(np.max(feature)-np.min(feature)))+0.0

def computeFeatures(maskObject, sample, info, inputImage):
    
    #Retreive recon values
    inputValues = inputImage[maskObject.unMeasuredIdxs[:,0], maskObject.unMeasuredIdxs[:,1]]
    
    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
    measuredValues = inputImage[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(info, maskObject, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)
    neighborValues = measuredValues[neighborIndices]

    #Create array to hold features
    feature = np.zeros((np.shape(maskObject.unMeasuredIdxs)[0],4))

    #Compute std div features
    diffVect = computeDifference(neighborValues, np.transpose(np.matlib.repmat(inputValues, np.shape(neighborValues)[1],1)), info.imageType)
    feature[:,0] = np.sum(neighborWeights*diffVect, axis=1)
    feature[:,0] = normalize(feature[:,0])
    feature[:,1] = np.sqrt(np.sum(np.power(diffVect,2),axis=1))
    feature[:,1] = normalize(feature[:,1])

    #Compute gradient features; assume continuous features
    gradientImageX, gradientImageY = np.gradient(inputImage)
    feature[:,2] = abs(gradientImageY)[maskObject.unMeasuredIdxs[:,0], maskObject.unMeasuredIdxs[:,1]]
    feature[:,2] = normalize(feature[:,2])
    feature[:,3] = abs(gradientImageX)[maskObject.unMeasuredIdxs[:,0], maskObject.unMeasuredIdxs[:,1]]
    feature[:,3] = normalize(feature[:,3])

    #Convert any nan values (feature 3, when all values are identical) to 0
    feature = np.nan_to_num(feature)

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

