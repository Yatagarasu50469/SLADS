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
        self.measuredImages = [np.zeros([maskObject.imageHeight, maskObject.imageWidth]) for rangeNum in range(0,len(massRanges))]
        self.resultsPath = resultsPath
        self.measuredLines = []
        self.avgImage = None

def iou(groundTruth, prediction):
    return np.sum(np.logical_and(groundTruth, prediction)) / np.sum(np.logical_or(groundTruth, prediction))

def cGaussian_parhelper(cNum, sigma, windowSize, orderForRD, imgAsBlocksOnlyUnmeasured):
    
    #For each of the selected unmeasured points calculate the captured "area"
    temp = np.zeros((windowSize[0]*windowSize[1], len(orderForRD)))
    for index in range(0,len(orderForRD)): temp[:,index] = imgAsBlocksOnlyUnmeasured[orderForRD[index],:]*generateGaussianKernel(sigma[orderForRD[index]], windowSize)
    
    return cNum, np.sum(temp, axis=0).reshape(1,-1).flatten()

#Singular result generated through runSLADS
class Result():
    def __init__(self, info, sample, maskObject, avgGroundTruthImage, simulationFlag, animationFlag, bestCFlag):
        self.info = info
        self.sample = sample
        self.avgGroundTruthImage = avgGroundTruthImage
        self.simulationFlag = simulationFlag
        self.animationFlag = animationFlag
        self.bestCFlag = bestCFlag
        self.reconImages = []
        self.RDImages = []
        self.threshReconImages = []
        self.maskObjects = []
        self.ERDValueNPs = []
        self.TDList = []
        self.MSEList = []
        self.SSIMList = []
        self.PSNRList = []
        self.iouList = []
        self.percMeasuredList = []
        self.thresholdList = []
        
        self.threshMSEList = []
        self.threshSSIMList = []
        self.threshPSNRList = []
        self.threshTDList = []      
        self.ERDPSNRList = []        
        
        #Threshold out the foreground
        self.threshMask = avgGroundTruthImage > threshold_triangle(avgGroundTruthImage)
        self.thresheldGroundTruth = self.threshMask*avgGroundTruthImage
        
        self.maximumValue = np.max(avgGroundTruthImage)
        

    def update(self, threshold, percMeasured, reconImage, sample, maskObject, ERDValuesNP, iterNum):

        #Save the model development
        self.maskObjects.append(copy.deepcopy(maskObject))
        self.reconImages.append(copy.deepcopy(reconImage))
        self.ERDValueNPs.append(copy.deepcopy(ERDValuesNP))
        self.sample = sample
        self.percMeasuredList.append(percMeasured)
        self.thresholdList.append(threshold)
    
    def complete(self, bestC, windowSize): 
        if self.simulationFlag:

            #Determine maximum value for color bar visualization
            self.maximumValue = np.max(self.reconImages)

            for maskObject in self.maskObjects:
                self.iouList.append(iou(self.threshMask, maskObject.mask))

            for reconImage in self.reconImages:

                #Find statistics of interest
                difference = np.sum(computeDifference(self.avgGroundTruthImage, reconImage, self.info.imageType))
                TD = difference/self.maskObjects[-1].area
                MSE = (np.sum((reconImage - self.avgGroundTruthImage) ** 2))/(self.maskObjects[-1].area)
                SSIM = structural_similarity(self.avgGroundTruthImage, reconImage, data_range=reconImage.max() - reconImage.min())
                PSNR = compare_psnr(self.avgGroundTruthImage, reconImage, data_range=reconImage.max() - reconImage.min())

                #Save them for each timestep
                self.TDList.append(TD)
                self.MSEList.append(MSE)
                self.SSIMList.append(SSIM)
                self.PSNRList.append(PSNR)
                
                #Find statistics of interest for thesheld image
                threshReconImage = self.threshMask*reconImage
                self.threshReconImages.append(threshReconImage)
                
                difference = np.sum(computeDifference(self.thresheldGroundTruth, threshReconImage, self.info.imageType))
                TD = difference/self.maskObjects[-1].area
                MSE = (np.sum((threshReconImage - self.thresheldGroundTruth) ** 2))/(self.maskObjects[-1].area)
                SSIM = structural_similarity(self.thresheldGroundTruth, threshReconImage, data_range=threshReconImage.max() - threshReconImage.min())
                PSNR = compare_psnr(self.thresheldGroundTruth, threshReconImage, data_range=threshReconImage.max() - threshReconImage.min())

                #Save them for each timestep
                self.threshTDList.append(TD)
                self.threshMSEList.append(MSE)
                self.threshSSIMList.append(SSIM)
                self.threshPSNRList.append(PSNR)
            
            #If determining the best c, return the area under the TD curve
            if self.bestCFlag:
                return np.trapz(self.TDList, self.percMeasuredList)
            
            for index in tqdm(range(0, len(self.maskObjects)), desc='RD Calc', leave = False, ascii=True):

                #Retrieve relevant variables
                maskObject = self.maskObjects[index]
                reconImage = self.reconImages[index]
            
                #Find neighbor information
                neighborIndices, neighborWeights, neighborDistances = findNeighbors(self.info, maskObject, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)
            
                #Calculate the sigma value for chosen c value
                sigmaValues = neighborDistances[:,0]/bestC
            
                #Flatten 2D mask array to 1D
                maskVect = np.ravel(maskObject.mask)
            
                #Use all of the unmeasured points for RD approximation
                orderForRD = np.arange(0,len(maskObject.unMeasuredIdxs)).tolist()
            
                #Compute the difference between the original and reconstructed images
                RDPP = computeDifference(self.avgGroundTruthImage, reconImage, self.info.imageType)

                #Convert differences to int
                RDPP.astype(int)

                #Pad with zeros
                RDPPWithZeros = np.pad(RDPP, [(int(np.floor(windowSize[0]/2)), ), (int(np.floor(windowSize[1]/2)), )], mode='constant')
                
                #Convert into a series of blocks and isolate unmeasured points in those blocks, save directly to shared memory
                imgAsBlocksOnlyUnmeasured = viewW(RDPPWithZeros, (windowSize[0],windowSize[1])).reshape(-1,windowSize[0]*windowSize[1])[:,::1][np.logical_not(maskVect),:]
        
                #Perform function and extract variables from the results
                result = cGaussian_parhelper(0, sigmaValues, windowSize, orderForRD, imgAsBlocksOnlyUnmeasured)
                RDImage = np.zeros((maskObject.imageHeight, maskObject.imageWidth))
                RDImage[maskObject.unMeasuredIdxs[:,0], maskObject.unMeasuredIdxs[:,1]] = result[1]
                self.RDImages.append(RDImage)
                
                #Determine PSNR of ERD
                ERDPSNR = compare_psnr(RDImage, self.ERDValueNPs[index], data_range=self.ERDValueNPs[index].max() - self.ERDValueNPs[index].min())
                self.ERDPSNRList.append(ERDPSNR)
            
        #If an animation will be produced and the run has completed
        if self.animationFlag:

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
                mzRecons = computeRecons(info, self.sample, self.maskObjects[-1], False)

                for massNum in range(0, len(self.sample.massRanges)):
                    
                    #mz image with only the actual measurements made
                    subMeasuredImage = self.sample.measuredImages[massNum]
                    
                    #Retreive reconstruction for the specific mz image
                    subReconImage = mzRecons[massNum]

                    #Retreive ground truth for the specific mz image
                    mzImage = self.sample.images[massNum]

                    #SSIM relative to the measured
                    measureSSIM = "{:.2f}".format(structural_similarity(mzImage, subMeasuredImage, data_range=subMeasuredImage.max() - subMeasuredImage.min()))
                    
                    #SSIM relative to the reconstruction
                    reconSSIM = "{:.2f}".format(structural_similarity(mzImage, subReconImage, data_range=subReconImage.max() - subReconImage.min()))
                    
                    #PSNR relative to the measured
                    measurePSNR = "{:.2f}".format(compare_psnr(mzImage, subMeasuredImage, data_range=subMeasuredImage.max() - subMeasuredImage.min()))
                    
                    #PSNR relative to the reconstruction
                    reconPSNR ="{:.2f}".format(compare_psnr(mzImage, subReconImage, data_range=subReconImage.max() - subReconImage.min()))

                    #Mass range string
                    massRange = str(self.sample.massRanges[massNum][0]) + '-' + str(self.sample.massRanges[massNum][1])

                    mzMaxValue = np.max([np.max(mzImage), np.max(subReconImage)])

                    #Measured mz image
                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(15,5))
                    f.subplots_adjust(top = 0.80)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle('Percent Sampled: ' + percSampled + '  Measurement mz: ' + massRange + '  SSIM: ' + measureSSIM + '  PSNR: ' + measurePSNR, fontsize=20, fontweight='bold', y = 0.95)

                    sub = f.add_subplot(1,3,1)
                    sub.imshow(self.maskObjects[-1].mask, cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')

                    sub = f.add_subplot(1,3,2)
                    sub.imshow(mzImage, cmap='hot', aspect='auto', vmin=0, vmax=mzMaxValue)
                    sub.set_title('Ground-Truth')

                    sub = f.add_subplot(1,3,3)
                    sub.imshow(subMeasuredImage, cmap='hot', aspect='auto', vmin=0, vmax=mzMaxValue)
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
                    plt.suptitle('Percent Sampled: ' + percSampled + '  Measurement mz: ' + massRange + '  SSIM: ' + reconSSIM + '  PSNR: ' + reconPSNR, fontsize=20, fontweight='bold', y = 0.95)
                    
                    sub = f.add_subplot(1,3,1)
                    sub.imshow(self.maskObjects[-1].mask, cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')
                    
                    sub = f.add_subplot(1,3,2)
                    sub.imshow(mzImage, cmap='hot', aspect='auto', vmin=0, vmax=mzMaxValue)
                    sub.set_title('Ground-Truth')
                    
                    sub = f.add_subplot(1,3,3)
                    sub.imshow(subReconImage, cmap='hot', aspect='auto', vmin=0, vmax=mzMaxValue)
                    sub.set_title('Reconstruction')
                    
                    saveLocation = dir_mzSampleResults + 'recon_' + massRange +'.png'
                    
                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()
                    
                    
                #Generate each of the frames
                for i in tqdm(range(0, len(self.maskObjects)), desc='Final Recon', leave = False, ascii=True):
                    
                    #2x2 with ERD printout
                    #=====================
                    saveLocation = dir_AnimationFrames + 'stretched_2x2_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'

                    f = plt.figure(figsize=(25,15))
                    f.subplots_adjust(top = 0.85)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle("Percent Sampled: %.2f,  Iteration: %.0f\nRecon PSNR: %.2f,  Foreground PSNR: %.2f\nRecon IoU: %.2f,  ERD PSNR: %.2f" % (self.percMeasuredList[i], i+1, self.PSNRList[i], self.threshPSNRList[i], self.iouList[i], self.ERDPSNRList[i]), fontsize=20, fontweight='bold', y = 0.95)

                    ax1 = plt.subplot2grid(shape=(2,3), loc=(0,0))
                    im = ax1.imshow(self.avgGroundTruthImage, cmap='hot', aspect='auto', vmin=0, vmax=self.maximumValue)
                    ax1.set_title('Ground-Truth')
                    cbar = f.colorbar(im, ax=ax1, orientation='vertical', pad=0.01)

                    ax2 = plt.subplot2grid((2,3), (0,1))
                    im = ax2.imshow(self.reconImages[i], cmap='hot', aspect='auto', vmin=0, vmax=self.maximumValue)
                    ax2.set_title('Reconstruction')
                    cbar = f.colorbar(im, ax=ax2, orientation='vertical', pad=0.01)

                    ax3 = plt.subplot2grid((2,3), (0,2))
                    im = ax3.imshow(abs(self.avgGroundTruthImage-self.reconImages[i]), cmap='hot', aspect='auto', vmin=0, vmax=self.maximumValue)
                    ax3.set_title('Absolute Difference')
                    cbar = f.colorbar(im, ax=ax3, orientation='vertical', pad=0.01)

                    ax4 = plt.subplot2grid((2,3), (1,0))
                    ax4.imshow(self.maskObjects[i].mask, cmap='gray', aspect='auto')
                    ax4.set_title('Measurement Mask')

                    ax5 = plt.subplot2grid((2,3), (1,1))
                    im = ax5.imshow(self.ERDValueNPs[i], cmap='viridis', vmin=np.min(self.ERDValueNPs), vmax=np.max(self.ERDValueNPs), aspect='auto')
                    ax5.set_title('ERD')
                    cbar = f.colorbar(im, ax=ax5, orientation='vertical', pad=0.01)

                    ax6 = plt.subplot2grid((2,3), (1,2))
                    im = ax6.imshow(self.RDImages[i], cmap='viridis', vmin=np.min(self.RDImages), vmax=np.max(self.RDImages), aspect='auto')
                    ax6.set_title('RD')
                    cbar = f.colorbar(im, ax=ax6, orientation='vertical', pad=0.01)

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()
                    #=====================

                    #No border saves
                    #=====================
                    #Save the reconstruction, no borders
                    saveLocation = dir_AnimationFrames + 'reconstruction_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.reconImages[i], cmap='hot', aspect='auto', vmin=0, vmax=self.maximumValue)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()
                    
                    #Save the thresheld reconstruction, no borders
                    saveLocation = dir_AnimationFrames + 'threshold_reconstruction_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.threshReconImages[i], cmap='hot', aspect='auto', vmin=0, vmax=self.maximumValue)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()
                    
                    #Save the ERD, no borders
                    saveLocation = dir_AnimationFrames + 'erd_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.ERDValueNPs[i], aspect='auto', vmin=0, vmax=np.max(self.ERDValueNPs))
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()
                    
                    #Save the RD, no borders
                    saveLocation = dir_AnimationFrames + 'rd_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.RDImages[i], aspect='auto', vmin=0, vmax=np.max(self.RDImages))
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()

                    #Save the mask, no borders
                    saveLocation = dir_AnimationFrames + 'mask_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.maskObjects[i].mask, cmap='gray', aspect='auto')
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()
                    #=====================
                    
                dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'stretched_2x2_*.png'))
                height, width, layers = cv2.imread(dataFileNames[0]).shape
                animation = cv2.VideoWriter(dir_AnimationVideos + 'stretched_2x2_' + self.sample.name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
                for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                animation.release()
                animation = None
                
                # dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'thresh_stretched_2x2_*.png'))
                # height, width, layers = cv2.imread(dataFileNames[0]).shape
                # animation = cv2.VideoWriter(dir_AnimationVideos + 'thresh_stretched_2x2_' + self.sample.name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
                # for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                # animation.release()
                # animation = None
        
                # dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'stretched_1x3_*.png'))
                # height, width, layers = cv2.imread(dataFileNames[0]).shape
                # animation = cv2.VideoWriter(dir_AnimationVideos + 'stretched_1x3_' + self.sample.name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
                # for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                # animation.release()
                # animation = None
                
                # dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'thresh_stretched_1x3_*.png'))
                # height, width, layers = cv2.imread(dataFileNames[0]).shape
                # animation = cv2.VideoWriter(dir_AnimationVideos + 'thresh_stretched_1x3_' + self.sample.name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
                # for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                # animation.release()
                # animation = None
                
                #Save the averaged ground truth, no borders
                saveLocation = dir_AnimationFrames + 'final_groundTruth_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.avgGroundTruthImage, cmap='hot', aspect='auto', vmin=0, vmax=self.maximumValue)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)

                #Save the averaged thresheld ground truth, no borders
                saveLocation = dir_AnimationFrames + 'final_thresh_groundTruth_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.thresheldGroundTruth, cmap='hot', aspect='auto', vmin=0, vmax=self.maximumValue)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)

                #Save the averaged final reconstruction, no borders
                saveLocation = dir_AnimationFrames + 'final_reconstruction_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.reconImages[-1], cmap='hot', aspect='auto', vmin=0, vmax=self.maximumValue)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)

                #Save the final mask, no borders
                saveLocation = dir_AnimationFrames + 'final_mask_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.maskObjects[-1].mask, cmap='gray', aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                
                #Save the averaged threshold mask, no borders
                saveLocation = dir_AnimationFrames + 'final_thresh_mask_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.threshMask, cmap='gray', aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                
            else: #Not a simulation
                sys.exit('ERROR! - Non simulation plots not yet fixed for color issue and modified selection')


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
    def __init__(self, imageWidth, imageHeight, measurementPercs, numMasks):
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.aspectRatio = imageWidth/imageHeight if imageWidth>imageHeight else imageHeight/imageWidth
        self.area = imageWidth*imageHeight
        self.percMasks = []
        self.measuredIdxs = []
        self.unMeasuredIdxs = []
        self.initialMeasuredIdxs = []
        self.initialUnMeasuredIdxs = []
        self.unMeasuredIdxsList = []
        self.measuredIdxsList = []
        self.initialSets = []

        #Create a blank initial mask
        self.initialMask = np.zeros([imageHeight, imageWidth])
        
        #If scanning is to be performed line by line
        if scanMethod == 'linewise':
        
            #Generate a list of arrays (lines) contianing the x,y points that need to be scanned
            self.linesToScan = []
            for rowNum in np.arange(0,imageHeight,1):
                line = []
                for columnNum in np.arange(0, imageWidth, 1):
                    line.append(tuple([rowNum, columnNum]))
                self.linesToScan.append(line)

            #Fix internel format, so that it is consistent throughout program operation
            self.linesToScan = np.asarray(self.linesToScan).tolist()

            #Copy the initial set of linesToScan
            self.originalLinesToScan = copy.copy(self.linesToScan)
        
            #Set which lines should be acquired in initial scan
            lineIndexes = [
                int((imageHeight-1)*0.50)
            ]
        
            #Obtain the points in the specified lines and add them to the initial scan list
            for lineIndexNum in range(0, len(lineIndexes)):
                lineIndex = lineIndexes[lineIndexNum]
                
                if partialLineFlag:
                    #Randomize and choose which points (termination %) should be scanned on the initial lines
                    newIdxs = [pt for pt in self.linesToScan[lineIndex]]
                    np.random.shuffle(newIdxs)
                    newIdxs = newIdxs[:int((stopPerc/100)*imageWidth)]
                else:
                    #Select all of the points on the chosen lines
                    for pt in self.linesToScan[lineIndex]: self.initialMask[tuple(pt)] = 1
                    newIdxs = [pt for pt in self.linesToScan[lineIndex]]
                
                #Add the points to the initial mask
                for pt in newIdxs: self.initialMask[tuple(pt)] = 1
                
                #Add point information for equipment implementation
                self.initialSets.append(newIdxs)
            
            
            
            #Now delete the lines/points specified from the remaining potentials; in case of shared points between 'lines' (diagonals, spirals, etc.)
            for lineIndexNum in range(0, len(lineIndexes)): self.delLine(lineIndexes[lineIndexNum]-lineIndexNum)
            
            #Make a duplicate for fresh initializations
            self.initialLinesToScan = copy.copy(self.linesToScan)
        
        #If scanning is to be performed on a point by point basis
        elif scanMethod == 'pointwise':
            if impModel: sys.exit('Error! - pointwise scanning method has not been setup for physical model implementation!')
            
            #Generate a random initial mask that scans 1% of sample pixels 
            self.initialMask = np.random.rand(imageHeight, imageWidth) < (1/100)
        
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
                self.mask = np.zeros([imageHeight, imageWidth])
                self.mask = np.random.rand(imageHeight, imageWidth) < (measurementPercs[percNum]/100)
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
            self.mask = np.zeros([self.imageHeight, self.imageWidth])
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

def runSLADS(info, samples, model, scanMethod, stopPerc, sampleNum, simulationFlag, trainPlotFlag, animationFlag, tqdmHide, bestCFlag):

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
    sample, reconImage, ERDValuesNP = avgReconAndERD(sample, info, iterNum, maskObject, None, None, model, newIdxs=None)

    #Determine percentage pixels measured initially
    percMeasured = (np.sum(maskObject.mask)/maskObject.area)*100

    #Check for completion state here just in case, prior to loop!
    completedRunFlag = checkStopCondFuncThreshold(stopCondParams, stopCondFuncVal, percMeasured, maskObject, sample, iterNum)

    #Additional stopping condition for if there are no more linesToScan
    if scanMethod == 'linewise' and len(maskObject.linesToScan) == 0: completedRunFlag = True

    #Initialize a result object
    result = Result(info, sample, maskObject, avgGroundTruthImage, simulationFlag, animationFlag, bestCFlag)
    result.update(0, percMeasured, reconImage, sample, maskObject, ERDValuesNP, iterNum)

    #Until the stopping criteria has been met
    with tqdm(total = float(100), desc = '% Sampled', leave = False, ascii=True, disable=tqdmHide) as pbar:

        #Initialize progress bar state according to % measured
        pbar.n = round(percMeasured,2)
        pbar.refresh()

        #Until the program has completed
        while not completedRunFlag:
            
            #Step the iteration counter
            iterNum += 1
            
            #Make a duplicate of the ReconImage for stop condition gradient test
            oldReconImage = reconImage.copy()
            
            #Find next measurement locations
            maskObject, newIdxs, threshold = findNewMeasurementIdxs(info, maskObject, sample, model, reconImage, ERDValuesNP, scanMethod)
            
            #Perform measurements
            sample, maskObject = performMeasurements(iterNum, sample, maskObject, newIdxs, simulationFlag)
            
            #Perform reconstruction and ERD calculation
            sample, reconImage, ERDValuesNP = avgReconAndERD(sample, info, iterNum, maskObject, reconImage, ERDValuesNP, model, newIdxs)
            
            #Update the percentage of pixels that have beene measured
            percMeasured = (np.sum(maskObject.mask)/maskObject.area)*100
            
            #Additional stopping condition for if there are no more linesToScan
            if scanMethod == 'linewise' and len(maskObject.linesToScan) == 0: completedRunFlag = True
               
            #Check the stopping condition before running the update step (triggers animation code)
            completedRunFlag = checkStopCondFuncThreshold(stopCondParams, stopCondFuncVal, percMeasured, maskObject, sample, iterNum)
    
            #Store information to the resultsObject
            result.update(threshold, percMeasured, reconImage, sample, maskObject, ERDValuesNP, iterNum)
            
            #Evaluate the stop condition value
            stopCondFuncVal = computeStopCondFuncVal(oldReconImage, reconImage, stopCondParams, info, stopCondFuncVal, newIdxs, iterNum, maskObject)

            #Update the progress bar
            pbar.n = round(percMeasured,2)
            pbar.refresh()

    return result

def findNewMeasurementIdxs(info, maskObject, sample, model, reconImage, ERDValuesNP, scanMethod):

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
            
            #Choose stopPerc locations on the line with maximized ERD
            newIdxs = np.asarray(newIdxs)[np.argsort(lineERDValues)][::-1][:int((stopPerc/100)*len(lineERDValues))]
            newIdxs = newIdxs.tolist()
        #==========================================

        #==========================================
        #PARTIAL LINE BY THRESHOLD
        #==========================================
        #Narrow the possible points for selection by the given threshold
        if lineERDSums[lineToScanIdx] > 0: newIdxs = [pos for pos in newIdxs if ERDValuesNP[tuple(pos)] >= threshold]
        #==========================================

        #==========================================
        #START/END POINTS BY THRESHOLD
        #==========================================
        #If not partial lines, select all ordered points between the first and last position with an ERD above threshold
        if (not partialLineFlag) and (len(newIdxs) > 1):
            newIdxs = np.asarray(newIdxs)
            orderedNewIdxs = newIdxs[np.argsort(newIdxs[:,0]*newIdxs[:,1])]
            startLocation, endLocation = orderedNewIdxs[0], orderedNewIdxs[len(orderedNewIdxs)-1]
            newIdxs = np.asarray(lineToScanIdxs[lineToScanIdxs.index(startLocation.tolist()):lineToScanIdxs.index(endLocation.tolist())])
        #==========================================

        #==========================================
        #SELECTION SAFEGUARD
        #==========================================
        #If there are not enough locations selected, just scan the whole remainder of the line with the greatest ERD; ensures model will reach termination
        if len(newIdxs) <= (0.01*maskObject.imageWidth): newIdxs = np.asarray(lineToScanIdxs).tolist()

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

def avgReconAndERD(sample, info, iterNum, maskObject, reconImage, ERDValuesNP, model, newIdxs):
    
    sample.avgImage = np.average(np.asarray(sample.measuredImages), axis=0, weights=sample.mzWeights)
    
    #On the first interation compute the full ERD and reconstruction and otherwise perform only partial updates
    if iterNum == 1:
        reconImage = computeRecons(info, sample, maskObject, True)
        ERDValuesNP = computeERD(info, sample, maskObject, reconImage, model)
    else:
        
        #Update the average image based on the measured images
        sample.avgImage = np.average(np.asarray(sample.measuredImages), axis=0, weights=sample.mzWeights)
        
        #Estimate a good radius
        suggestedRadius = int(np.sqrt((1/np.pi)*(maskObject.area*info.numNeighbors/np.sum(maskObject.mask))))
        
        #Minimum radius check
        updateRadiusTemp = np.max([suggestedRadius, 3]);
        
        #Maximum radius check
        updateRadius=int(np.min([10,updateRadiusTemp]));

        #Increase the radius until all of the unmeasured points are inside
        updateRadiusMat = np.zeros((maskObject.imageHeight, maskObject.imageWidth))
        while True:
            for newIdx in newIdxs:
                updateRadiusMat[max(newIdx[0]-updateRadius,0):min(newIdx[0]+updateRadius,maskObject.imageHeight)][:,max(newIdx[1]-updateRadius,0):min(newIdx[1]+updateRadius,maskObject.imageWidth)]=1
            updateIdxs = np.where(updateRadiusMat[maskObject.mask==0]==1)
            smallUnMeasuredIdxs = np.transpose(np.where(np.logical_and(maskObject.mask==0,updateRadiusMat==1)))
            if smallUnMeasuredIdxs.size==0:
                updateRadius=int(updateRadius*1.5)
            else:
                break

        #Retreive measured values
        idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
        measuredValues = sample.avgImage[np.asarray(idxsX), np.asarray(idxsY)]

        #Determine neighborhood information
        smallNeighborIndices, smallNeighborWeights, smallNeighborDistances = findNeighbors(info, maskObject, maskObject.measuredIdxs, smallUnMeasuredIdxs)
        smallNeighborValues = measuredValues[smallNeighborIndices]

        #Perform reconstruction
        smallReconValues = np.sum(smallNeighborValues*smallNeighborWeights, axis=1)
        reconImage[smallUnMeasuredIdxs[:,0], smallUnMeasuredIdxs[:,1]] = smallReconValues
        reconImage[maskObject.measuredIdxs[:,0], maskObject.measuredIdxs[:,1]] = measuredValues

        #Determine polyFeatures
        smallPolyFeatures = computeFeatures(maskObject, smallUnMeasuredIdxs, smallNeighborValues, smallNeighborWeights, smallNeighborDistances, info, reconImage)
        
        #Update ERD here
        smallERDValues = model.predict(smallPolyFeatures)
        ERDValuesNP[(np.logical_and(maskObject.mask==0, updateRadiusMat==1))] = smallERDValues

        #Set measured ERD values as 0
        for i in range(0, len(maskObject.measuredIdxs)): ERDValuesNP[maskObject.measuredIdxs[i][0], maskObject.measuredIdxs[i][1]] = 0
            
    return sample, reconImage, ERDValuesNP

def computeERD(info, sample, maskObject, reconImage, model):

    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
    measuredValues = sample.avgImage[np.asarray(idxsX), np.asarray(idxsY)]

    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(info, maskObject, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)
    neighborValues = measuredValues[neighborIndices]

    #Determine polyFeatures
    polyFeatures = computeFeatures(maskObject, maskObject.unMeasuredIdxs, neighborValues, neighborWeights, neighborDistances, info, reconImage)

    # Compute ERD
    ERDValues = model.predict(polyFeatures)
    
    #Rearrange ERD values into array; those that have already been measured have 0 ERD
    ERDValuesNP = np.zeros([maskObject.imageHeight, maskObject.imageWidth])
    for i in range(0, len(maskObject.unMeasuredIdxs)): ERDValuesNP[maskObject.unMeasuredIdxs[i][0], maskObject.unMeasuredIdxs[i][1]] = ERDValues[i]
    
    return ERDValuesNP

def checkStopCondFuncThreshold(stopCondParams, StopCondFuncVal, percMeasured, maskObject, sample, iterNum):

    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
    measuredValues = sample.avgImage[np.asarray(idxsX), np.asarray(idxsY)]

    if stopCondParams.threshold == 0:
        if round(percMeasured) >= stopCondParams.maxPercentage:
            return True
        else:
            return False
    else:
        if round(percMeasured) >= stopCondParams.maxPercentage:
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

    neigh = NearestNeighbors(n_neighbors=info.numNeighbors)
    neigh.fit(measuredIdxs)
    neighborDistances, neighborIndices = neigh.kneighbors(unMeasuredIdxs)
    neighborDistances = neighborDistances*info.resolution
    unNormNeighborWeights = 1/np.power(neighborDistances, info.neighborWeightsPower)
    sumOverRow = (np.sum(unNormNeighborWeights, axis=1))
    neighborWeights = unNormNeighborWeights/sumOverRow[:, np.newaxis]
    
    return neighborIndices, neighborWeights, neighborDistances

def performRecon(inputImage, maskObject, neighborIndices, neighborWeights, neighborDistances):
    
    #Create a blank image for the reconstruction
    reconImage = np.zeros((maskObject.imageHeight, maskObject.imageWidth))
    
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

def computeFeatures(maskObject, unMeasuredIdxs, neighborValues, neighborWeights, neighborDistances, info, inputImage):

    #Retreive input image (recon) values
    inputValues = inputImage[unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]]

    #Create array to hold features
    feature = np.zeros((np.shape(unMeasuredIdxs)[0],6))
    
    #Compute std div features
    diffVect = computeDifference(neighborValues, np.transpose(np.matlib.repmat(inputValues, np.shape(neighborValues)[1],1)), info.imageType)
    feature[:,0] = np.sum(neighborWeights*diffVect, axis=1)
    feature[:,1] = np.sqrt(np.sum(np.power(diffVect,2),axis=1))
    
    #Compute distance/density features
    cutoffDist = np.ceil(np.sqrt((info.featDistCutoff/100)*(maskObject.area/np.pi)))
    feature[:,2] = neighborDistances[:,0]
    feature[:,3] = (1+(np.pi*(np.power(cutoffDist, 2))))/(1+np.sum(neighborDistances <= cutoffDist, axis=1))

    #Compute gradient features; assume continuous features
    gradientImageX, gradientImageY = np.gradient(inputImage)
    feature[:,4] = abs(gradientImageY)[unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]]
    feature[:,5] = abs(gradientImageX)[unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]]

    #Convert any nan values to 0
    feature = np.nan_to_num(feature)

    #Fit features
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
    
    percents = np.linspace(min(np.hstack(perc_testingResults)), max(np.hstack(perc_testingResults)), int((max(np.hstack(perc_testingResults)) - min(np.hstack(perc_testingResults))) / precision + 1))
    newResults = [np.interp(percents, perc_testingResults[resultNum], results[resultNum]) for resultNum in range(0, len(results))]
    averageResults = np.average(newResults, axis=0)
    
    return percents, averageResults

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

def readScanData(folderLocation):
    images = []
    massRanges = []
    #Import each of the images according to their mz range order
    for imageFileName in natsort.natsorted(glob.glob(folderLocation+ '*.csv'), reverse=False):
        images.append(np.nan_to_num(np.loadtxt(imageFileName, delimiter=',')))
        massRanges.append([os.path.basename(imageFileName)[2:10], os.path.basename(imageFileName)[11:19]])
    imageHeight, imageWidth = images[0].shape

    return images, massRanges, imageHeight, imageWidth

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

