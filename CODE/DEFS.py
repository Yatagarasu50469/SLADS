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

@ray.remote
def recon_parhelper(maskObject, image):
    
    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(maskObject, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)
    
    #Perform reconstruction
    reconImage = performRecon(image, maskObject, neighborIndices, neighborWeights, neighborDistances)
    
    return reconImage

@ray.remote
def performRecon_parhelper(image, maskObject, neighborIndices, neighborWeights, neighborDistances):

    #Perform reconstruction
    reconImage = performRecon(image, maskObject, neighborIndices, neighborWeights, neighborDistances)

    return reconImage

#Generate/apply a gaussian kernel to a window of an image centered at an idx, summing result; window size according to sigma strength
def gaussianGenerator(inputImage, idx, sigma, maskObject):
    
    #Determine odd number >= 3 times the given sigma value for a reasonable window size to generate, (width, height)
    windowSize = int(np.ceil((np.ceil(sigma*3)//2)*2+1))
    
    #Penalize according to the aspect ratio
    if maskObject.imageWidth>maskObject.imageHeight:
        windowSize = [int(np.ceil(windowSize/maskObject.aspectRatio)), windowSize]
    else:
        windowSize = [windowSize, int(np.ceil(windowSize/maskObject.aspectRatio))]
    
    #Pad input image based on window size, to ensure no data is lost when splitting into windows
    paddedInputImage = np.pad(inputImage, [(int(np.floor(windowSize[0]/2)), ), (int(np.floor(windowSize[1]/2)), )], mode='constant')

    #Extract window around specified idx and calculate kernel
    window = viewW(paddedInputImage, (windowSize[0], windowSize[1]))[idx[0], idx[1]]
    kernel = np.outer(signal.gaussian(windowSize[0], std=sigma), signal.gaussian(windowSize[1], std=sigma))

    return np.sum(window*kernel)

#Perform gaussianGenerator for a set of sigma values
@ray.remote
def gaussian_parhelper(idxs, inputImage, sigmaValues, maskObject, indexes):
    return [gaussianGenerator(inputImage, idxs[index], sigmaValues[index], maskObject) for index in indexes]

def calcRD(maskObject, reconImage, cValue, avgGroundTruthImage):
    
    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(maskObject, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)

    #Calculate the sigma value for chosen c value, store directly into shared memory
    sigmaValues_id = ray.put(neighborDistances[:,0]/cValue)
    
    #Compute the difference between the original and reconstructed images, store directly into shared memory
    RDPP_id = ray.put(computeDifference(avgGroundTruthImage, reconImage))
    
    #Split computation sets to be run on multiple threads
    indexes = np.asarray(list(range(0, np.sum(maskObject.mask==0))))
    blockSize = int(np.ceil(len(indexes) / float(multiprocessing.cpu_count())))
    indexSets = np.split(indexes, np.arange(blockSize, len(indexes), blockSize))

    #Store indexes into shared memory
    unMeasuredIdxs_id = ray.put(maskObject.unMeasuredIdxs)

    #Store aspect ratio into shared memory
    maskObject_id = ray.put(maskObject)
    
    #Perform computation of RD values for each unmeasured point
    results = ray.get([gaussian_parhelper.remote(unMeasuredIdxs_id, RDPP_id, sigmaValues_id, maskObject_id, indexes) for indexes in indexSets])
    RDValues = [result for resultSet in results for result in resultSet]
    
    #Reassemble the values into a single image and min-max normalize (visualization consistency)
    RDImage = np.zeros((avgGroundTruthImage.shape))
    RDImage[np.where(maskObject.mask==0)] = RDValues
    RDImage = normalize(RDImage)

    return RDImage

#Singular result generated through runSLADS
class Result():
    def __init__(self, sample, maskObject, avgGroundTruthImage, bestCFlag, oracleFlag, simulationFlag, animationFlag):
        self.sample = sample
        self.avgGroundTruthImage = avgGroundTruthImage
        self.simulationFlag = simulationFlag
        self.animationFlag = animationFlag
        self.bestCFlag = bestCFlag
        self.oracleFlag = oracleFlag
        self.avgImages = []
        self.reconImages = []
        self.RDImages = []
        self.threshReconImages = []
        self.mzRecons = []
        self.maskObjects = []
        self.ERDValueNPs = []
        
        self.iouList = []
        self.MSEList = []
        self.SSIMList = []
        self.PSNRList = []
        self.TDList = []
        self.ERDPSNRList = []

        self.threshMSEList = []
        self.threshSSIMList = []
        self.threshPSNRList = []
        self.threshTDList = []        
        
        self.percMeasuredList = []

        #Threshold out the foreground
        self.threshMask = avgGroundTruthImage > threshold_triangle(avgGroundTruthImage)
        self.thresheldGroundTruth = self.threshMask*avgGroundTruthImage
        
        self.maximumValue = np.max(avgGroundTruthImage)

    def update(self, percMeasured, sample, maskObject, reconImage, ERDValuesNP, iterNum):

        #Save the model development
        self.maskObjects.append(copy.deepcopy(maskObject))
        self.ERDValueNPs.append(copy.deepcopy(ERDValuesNP))
        self.avgImages.append(copy.deepcopy(sample.avgImage))
        self.sample = sample
        self.percMeasuredList.append(percMeasured)
        if self.bestCFlag or self.oracleFlag: self.reconImages.append(reconImage)
    
    def complete(self, bestC): 
        if self.simulationFlag:
        
            if not self.bestCFlag and not self.oracleFlag:
                results = ray.get([recon_parhelper.remote(self.maskObjects[index], self.avgImages[index]) for index in range(0,len(self.maskObjects))])
                self.reconImages = [result for result in results]

            #Perform statistics extraction for all images
            for index in range(0, len(self.reconImages)):
                
                #Measure and save statistics
                difference = np.sum(computeDifference(self.avgGroundTruthImage, self.reconImages[index]))
                TD = difference/self.maskObjects[index].area
                MSE = mean_squared_error(self.avgGroundTruthImage, self.reconImages[index])
                SSIM = structural_similarity(self.avgGroundTruthImage, self.reconImages[index], data_range=self.reconImages[index].max() - self.reconImages[index].min())
                PSNR = compare_psnr(self.avgGroundTruthImage, self.reconImages[index], data_range=self.reconImages[index].max() - self.reconImages[index].min())

                self.TDList.append(TD)
                self.MSEList.append(MSE)
                self.SSIMList.append(SSIM)
                self.PSNRList.append(PSNR)
                
                #Find statistics of interest for thesheld image
                threshReconImage = self.threshMask*self.reconImages[index]
                self.threshReconImages.append(threshReconImage)
                
                #Measure and save statistics
                difference = np.sum(computeDifference(self.thresheldGroundTruth, threshReconImage))
                TD = difference/self.maskObjects[index].area
                MSE = mean_squared_error(self.thresheldGroundTruth, threshReconImage)
                SSIM = structural_similarity(self.thresheldGroundTruth, threshReconImage, data_range=threshReconImage.max() - threshReconImage.min())
                PSNR = compare_psnr(self.thresheldGroundTruth, threshReconImage, data_range=threshReconImage.max() - threshReconImage.min())

                self.threshTDList.append(TD)
                self.threshMSEList.append(MSE)
                self.threshSSIMList.append(SSIM)
                self.threshPSNRList.append(PSNR)

            #If determining the best c, return the area under the TD curve
            if self.bestCFlag: return self.PSNRList, self.percMeasuredList

            #Determine maximum value for color bar visualization
            self.maximumValue = np.max(self.reconImages)

            #Calculate the actual RD Image and relative error of the ERD for each of the masks; bestCFlag data must be returned before this subroutine
            for index in tqdm(range(0, len(self.maskObjects)), desc='RD Calc', leave = False, ascii=True):
                RDImage = calcRD(self.maskObjects[index], self.reconImages[index], bestC, self.avgGroundTruthImage)
                self.RDImages.append(RDImage)
                ERDPSNR = compare_psnr(RDImage, self.ERDValueNPs[index], data_range=self.ERDValueNPs[index].max() - self.ERDValueNPs[index].min())
                self.ERDPSNRList.append(ERDPSNR)

            for maskObject in self.maskObjects: self.iouList.append(iou(self.threshMask, maskObject.mask))
                    
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

            #If this was a simulation
            if self.simulationFlag:
                
                #Save each of the individual mass range reconstructions
                percSampled = "{:.2f}".format(self.percMeasuredList[-1])

                #Find neighbor information
                neighborIndices, neighborWeights, neighborDistances = findNeighbors(self.maskObjects[-1], self.maskObjects[-1].measuredIdxs, self.maskObjects[-1].unMeasuredIdxs)

                #Perform reconstructions and statistics extraction for all images
                results = ray.get([performRecon_parhelper.remote(self.sample.measuredImages[mzImageNum], self.maskObjects[-1], neighborIndices, neighborWeights, neighborDistances) for mzImageNum in range(0,len(self.sample.measuredImages))])
                self.mzRecons = [result for result in results]

                for massNum in tqdm(range(0, len(self.sample.massRanges)), desc='mz Images', leave = False, ascii=True):
                    
                    #mz image with only the actual measurements made
                    subMeasuredImage = self.sample.measuredImages[massNum].astype("float")
                    
                    #Retreive reconstruction for the specific mz image
                    subReconImage = self.mzRecons[massNum]

                    #Retreive ground truth for the specific mz image
                    mzImage = self.sample.images[massNum].astype("float")
                    
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
                for i in tqdm(range(0, len(self.maskObjects)), desc='Result Images', leave = False, ascii=True):
                    
                    #2x2 with ERD printout
                    #=====================
                    saveLocation = dir_AnimationFrames + 'stretched_2x2_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'

                    f = plt.figure(figsize=(35,15))
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
                    im = ax4.imshow(self.maskObjects[i].mask, cmap='gray', aspect='auto')
                    ax4.set_title('Measurement Mask')
                    cbar = f.colorbar(im, ax=ax4, orientation='vertical', pad=0.01)

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
                plt.close()

                #Save the averaged thresheld ground truth, no borders
                saveLocation = dir_AnimationFrames + 'final_thresh_groundTruth_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.thresheldGroundTruth, cmap='hot', aspect='auto', vmin=0, vmax=self.maximumValue)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()

                #Save the averaged final reconstruction, no borders
                saveLocation = dir_AnimationFrames + 'final_reconstruction_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.reconImages[-1], cmap='hot', aspect='auto', vmin=0, vmax=self.maximumValue)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()

                #Save the final mask, no borders
                saveLocation = dir_AnimationFrames + 'final_mask_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.maskObjects[-1].mask, cmap='gray', aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()
                
                #Save the averaged threshold mask, no borders
                saveLocation = dir_AnimationFrames + 'final_thresh_mask_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.threshMask, cmap='gray', aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()
                
            else: #Not a simulation
                sys.exit('ERROR! - Non simulation plots not yet fixed for color issue and modified selection')

#Each sample needs a mask object
class MaskObject():
    def __init__(self, imageWidth, imageHeight, initialPercToScan, scanMethod):
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.aspectRatio = imageWidth/imageHeight if imageWidth>imageHeight else imageHeight/imageWidth
        self.area = imageWidth*imageHeight
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
            
            #Generate a random initial mask that scans requested percent of sample pixels
            self.unMeasuredIdxs = np.transpose(np.where(self.initialMask == 0))
            np.random.shuffle(self.unMeasuredIdxs)
            for pt in self.unMeasuredIdxs[:int((initialPercToScan/100)*self.area)]: self.initialMask[tuple(pt)] = 1
        
        #Store the initially measured and unmeasured mask locations for the first measurement step
        self.initialMeasuredIdxs = np.transpose(np.where(self.initialMask == 1))
        self.initialUnMeasuredIdxs = np.transpose(np.where(self.initialMask == 0))
        
        #Reset the internal mask to match the initial
        self.mask = copy.deepcopy(self.initialMask)
        self.measuredIdxs = np.transpose(np.where(self.mask == 1))
        self.unMeasuredIdxs = np.transpose(np.where(self.mask == 0))
    
    #Update the mask given a set of new measurement locations
    def update(self, newIdxs):
        for pt in newIdxs: self.mask[tuple(pt)] = 1
        
        #SUGGESTED IMPROVEMENT: Should change this to add new Idxs to the list, rather than going through the whole list again
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
            self.mask = copy.deepcopy(self.initialMask)
            self.measuredIdxs = np.transpose(np.where(self.mask == 1))
            self.unMeasuredIdxs = np.transpose(np.where(self.mask == 0))

    #Remove entire lines from future consideration
    def delLine(self, index):
        if scanMethod == 'linewise':
            self.linesToScan = np.delete(self.linesToScan, index, 0).tolist()
        else:
            sys.exit('Error! - Attempted to delete a line with non-linewise scanning method')

    #Remove points from future consideration
    def delPoints(self, pts):
        if scanMethod == 'linewise':
            for lineNum in range(0, len(self.linesToScan)): self.linesToScan[lineNum] = [pt for ix, pt in enumerate(self.linesToScan[lineNum]) if pt not in pts]
            self.linesToScan = [x for x in self.linesToScan if x]
        else:
            sys.exit('Error! - Attempted to delete points from lines with non-linewise scanning method')

def computeERD(iterNum, measuredAvgImage, maskObject, model):
    
    #Compute feature image for network input
    featureImage = featureExtractor(maskObject, measuredAvgImage)

    #Send input through trained model
    inputTensor, padding = makeTensor(featureImage)
    ERD = model.predict(inputTensor, steps=1)[0,:,:,0]
    
    #Remove padding
    ERD = ERD[:measuredAvgImage.shape[0],:measuredAvgImage.shape[1]]
    
    #Ensure points already measured and those with values less than those have a 0 ERD
    for i in range(0, len(maskObject.measuredIdxs)): ERD[maskObject.measuredIdxs[i][0], maskObject.measuredIdxs[i][1]] = 0
    ERD[np.where((ERD < 0))] = 0
    
    #Normzalize
    ERD = normalize(ERD)
    
    return ERD
    
def runSLADS(samples, model, cValue, percToScan, stopPerc, sampleNum, simulationFlag, trainPlotFlag, animationFlag, tqdmHide, oracleFlag, bestCFlag):

    if simulationFlag: #Here sample.images contains the full ground-truth images
        sample = samples[sampleNum]
        avgGroundTruthImage = np.average(sample.images, axis=0, weights=sample.mzWeights)
    else: #Here sample.images contains only the initially measured ground-truth images
        sample = samples
        sample.measuredImages = sample.images
        avgGroundTruthImage = []

    maskObject = sample.maskObject

    #Reinitialize the mask to starting state
    maskObject.reset(simulationFlag)

    #Indicate the stopping condition has not yet been met
    completedRunFlag = False

    #Set the current iteration
    iterNum = 1

    #Perform the initial measurements
    if simulationFlag: sample, maskObject = performMeasurements(sample, maskObject, maskObject.initialMeasuredIdxs, simulationFlag)

    #Determine percentage pixels measured initially
    percMeasured = (np.sum(maskObject.mask)/maskObject.area)*100

    #Additional stopping condition for if there are no more linesToScan
    if scanMethod == 'linewise' and len(maskObject.linesToScan) == 0: completedRunFlag = True

    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(maskObject, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)

    #Perform weighted averaging of the multiple channels
    sample.avgImage = np.average(np.asarray(sample.measuredImages), axis=0, weights=sample.mzWeights)
    
    #Determine ERD or use the RD as the ERD if the oracleFlag or bestCFlag is enabled
    if oracleFlag or bestCFlag:
        reconImage = performRecon(sample.avgImage, maskObject, neighborIndices, neighborWeights, neighborDistances)
        ERDValuesNP = calcRD(maskObject, reconImage, cValue, avgGroundTruthImage)
    else:
        ERDValuesNP = computeERD(iterNum, sample.avgImage, maskObject, model)

    #Initialize a result object
    result = Result(sample, maskObject, avgGroundTruthImage, bestCFlag, oracleFlag, simulationFlag, animationFlag)

    if oracleFlag or bestCFlag:
        result.update(percMeasured, sample, maskObject, reconImage, ERDValuesNP, iterNum)
    else:
        result.update(percMeasured, sample, maskObject, None, ERDValuesNP, iterNum)

    #Check stopping criteria, just in case of a bad input
    if round(percMeasured) >= stopPerc: completedRunFlag = True
    if scanMethod == 'linewise' and len(maskObject.linesToScan) == 0: completedRunFlag = True

    #Until the stopping criteria has been met
    with tqdm(total = float(100), desc = '% Sampled', leave = False, ascii=True, disable=tqdmHide) as pbar:

        #Initialize progress bar state according to % measured
        pbar.n = round(percMeasured,2)
        pbar.refresh()

        #Until the program has completed
        while not completedRunFlag:
            
            #Step the iteration counter
            iterNum += 1
            
            #Find next measurement locations
            maskObject, newIdxs = findNewMeasurementIdxs(maskObject, sample, ERDValuesNP, percToScan, False)
            
            #Perform measurements
            sample, maskObject = performMeasurements(sample, maskObject, newIdxs, simulationFlag)
            
            #Update the percentage of pixels that have been measured
            percMeasured = (np.sum(maskObject.mask)/maskObject.area)*100
            
            #Check stopping conditions
            if round(percMeasured) >= stopPerc: completedRunFlag = True
            if scanMethod == 'linewise' and len(maskObject.linesToScan) == 0: completedRunFlag = True
    
            #Find neighbor information
            neighborIndices, neighborWeights, neighborDistances = findNeighbors(maskObject, maskObject.measuredIdxs, maskObject.unMeasuredIdxs)

            #Determine ERD or use the RD as the ERD if the oracleFlag or bestCFlag is enabled
            if oracleFlag or bestCFlag:
                reconImage = performRecon(sample.avgImage, maskObject, neighborIndices, neighborWeights, neighborDistances)
                ERDValuesNP = calcRD(maskObject, reconImage, cValue, avgGroundTruthImage)
                result.update(percMeasured, sample, maskObject, reconImage, ERDValuesNP, iterNum)
            else:
                ERDValuesNP = computeERD(iterNum, sample.avgImage, maskObject, model)
                result.update(percMeasured, sample, maskObject, None, ERDValuesNP, iterNum)

            #Update the progress bar
            pbar.n = round(percMeasured,2)
            pbar.refresh()

    return result

def findNewMeasurementIdxs(maskObject, sample, ERDValuesNP, percToScan, randomizedSelection):

    #Make sure ERDValuesNP is in np array
    ERDValuesNP = np.asarray(ERDValuesNP)
    
    if randomizedSelection:
        newIdxs = np.asarray(random.sample(maskObject.unMeasuredIdxs.tolist(), int((percToScan/100)*maskObject.area)))
    elif scanMethod == 'pointwise':
        
        #Obtain a list of all ERD values for unmeasured locations
        ERDValueList = [ERDValuesNP[tuple(pt)] for pt in maskObject.unMeasuredIdxs]
        
        #Sort the values in reverse order and choose the top 1% of values
        newIdxs = maskObject.unMeasuredIdxs[np.argsort(ERDValueList)][::-1][:int((percToScan/100)*maskObject.area)]

    elif scanMethod == 'linewise':
        #==========================================
        #OPTIMAL LINE DETERMINATION
        #==========================================

        #Set an initial threshold
        threshold = 0

        #Sum ERD for all lines
        lineERDSums = [np.nansum(ERDValuesNP[tuple([x[0] for x in line]), tuple([y[1] for y in line])]) for line in maskObject.linesToScan]

        #Choose the line with maximum ERD and extract the actual indices
        lineToScanIdx = np.nanargmax(lineERDSums)
        lineToScanIdxs = maskObject.linesToScan[lineToScanIdx]

        #Set the new Idxs to be all of the indexes identified
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
        if len(newIdxs) <= (0.01*maskObject.imageWidth): newIdxs = np.asarray(lineToScanIdxs).tolist()

        #==========================================
        #POINT CONSIDERATION UPDATE FOR LINEWISE
        #==========================================
        #Remove the selected points from further consideration, allows revisting lines
        if lineRevistFlag:
            maskObject.delPoints(newIdxs)
        else:
            #Remove the line selected from further consideration, does not allow revisiting
            maskObject.delLine(lineToScanIdx)
        #==========================================

    return maskObject, newIdxs

def findNeighbors(maskObject, measuredIdxs, unMeasuredIdxs):
    neigh = NearestNeighbors(n_neighbors=10, metric='asym', aspect=maskObject.aspectRatio)
    #neigh = NearestNeighbors(n_neighbors=10, algorithm='kd_tree', metric='asym', aspect=maskObject.aspectRatio, n_jobs=multiprocessing.cpu_count())
    neigh.fit(measuredIdxs)
    neighborDistances, neighborIndices = neigh.kneighbors(unMeasuredIdxs)
    unNormNeighborWeights = 1/np.power(neighborDistances, 2)
    sumOverRow = (np.sum(unNormNeighborWeights, axis=1))
    neighborWeights = unNormNeighborWeights/sumOverRow[:, np.newaxis]
    
    return neighborIndices, neighborWeights, neighborDistances

def performRecon(inputImage, maskObject, neighborIndices, neighborWeights, neighborDistances):

    #Create a blank image for the reconstruction
    reconImage = np.zeros((maskObject.imageHeight, maskObject.imageWidth))
    
    #Retreive measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
    measuredValues = inputImage[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Compute and save reconstruction values, combining with the measured values
    reconImage[maskObject.unMeasuredIdxs[:,0], maskObject.unMeasuredIdxs[:,1]] = np.sum(measuredValues[neighborIndices]*neighborWeights, axis=1)
    reconImage[maskObject.measuredIdxs[:,0], maskObject.measuredIdxs[:,1]] = measuredValues

    return reconImage

def normalize(feature):
    return (feature-np.min(feature))*((1.0-0.0)/(np.max(feature)-np.min(feature)))+0.0

def computeDifference(array1, array2):
    return abs(array1-array2)
        
def percResults(results, perc_testingResults, precision):

    percents = np.linspace(min(np.hstack(perc_testingResults)), max(np.hstack(perc_testingResults)), int((max(np.hstack(perc_testingResults)) - min(np.hstack(perc_testingResults))) / precision + 1))
    newResults = [np.interp(percents, perc_testingResults[resultNum], results[resultNum]) for resultNum in range(0, len(results))]
    averageResults = np.average(newResults, axis=0)
    
    return percents, averageResults

def performMeasurements(sample, maskObject, newIdxs, simulationFlag):

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

    #Perform weighted averaging of the multiple channels at original dimensionality
    sample.avgImage = np.average(np.asarray(sample.measuredImages), axis=0, weights=sample.mzWeights)

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

def identityBlock(inputData, numFilters):
    conv_1 = Conv2D(numFilters, (1,1), padding='same', kernel_initializer='he_normal')(inputData)
    conv_1 = LayerNormalization()(conv_1)
    conv_1 = ReLU()(conv_1)
    conv_1 = Conv2D(numFilters, (3,3), padding='same', kernel_initializer='he_normal')(conv_1)
    conv_1 = LayerNormalization()(conv_1)
    conv_1 = ReLU()(conv_1)
    conv_1 = Conv2D(numFilters, (1,1), padding='same', kernel_initializer='he_normal')(conv_1)
    conv_1 = LayerNormalization()(conv_1)
    conv_1 = concatenate([inputData, conv_1], axis=3)
    conv_1 = ReLU()(conv_1)
    
    return conv_1

#CNN
def cnn(numFilters, numChannels):
    inputs = Input(shape=(None,None,numChannels))
    conv_1 = identityBlock(inputs, numFilters)
    conv_2 = identityBlock(conv_1, numFilters)
    conv_3 = identityBlock(conv_2, numFilters)
    conv_4 = identityBlock(conv_3, numFilters)
    
    output = Conv2D(1, (1,1), activation='linear', padding='same', kernel_initializer='he_normal')(conv_4)
    
    return tf.keras.Model(inputs=inputs, outputs=output)

#U-Net
def unet(numFilters, numChannels):
    
    inputs = Input(shape=(None,None,numChannels))

    conv_1 = identityBlock(inputs, numFilters)
    down_1 = Conv2D(numFilters, (1,1), strides=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)

    conv_2 = identityBlock(down_1, numFilters*2)
    down_2 = Conv2D(numFilters*2, (1,1), strides=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)

    conv_3 = identityBlock(down_2, numFilters*4)
    down_3 = Conv2D(numFilters*4, (1,1), strides=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)
    
    conv_4 = identityBlock(down_3, numFilters*8)

    upScale_5 = Conv2D(numFilters*4, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(interpolation='nearest')(conv_4))
    skip_5 = concatenate([upScale_5, conv_3], axis=3)
    conv_5 = identityBlock(skip_5, numFilters*4)
    
    upScale_6 = Conv2D(numFilters*2, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(interpolation='nearest')(conv_5))
    skip_6 = concatenate([upScale_6, conv_2], axis=3)
    conv_6 = identityBlock(skip_6, numFilters*2)
    
    upScale_7 = Conv2D(numFilters, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(interpolation='nearest')(conv_6))
    skip_7 = concatenate([upScale_7, conv_1], axis=3)
    conv_7 = identityBlock(skip_7, numFilters*2)
    
    output = Conv2D(1, (1,1), activation='linear', padding='same', kernel_initializer='he_normal')(conv_7)
    
    return tf.keras.Model(inputs=inputs, outputs=output)

#RBDN approach with 3 branches and normalization
def rbdnWithNormalization(numFilters, numChannels):
    
    #Branch 0 start
    inputs = Input(shape=(None,None,numChannels))
    
    conv_0 = Conv2D(numFilters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv_0 = Conv2D(numFilters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_0)
    ln = LayerNormalization()(conv_0)
    maxPool_0 = MaxPooling2D(pool_size=(2,2))(ln)
    
    #Branch 1 start
    conv_10 = Conv2D(numFilters*2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(maxPool_0)
    conv_10 = Conv2D(numFilters*2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_10)
    ln = LayerNormalization()(conv_10)
    maxPool_1 = MaxPooling2D(pool_size=(2,2))(ln)
    
    #Branch 2 start
    conv_20 = Conv2D(numFilters*4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(maxPool_1)
    conv_20 = Conv2D(numFilters*4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_20)
    ln = LayerNormalization()(conv_20)
    maxPool_2 = MaxPooling2D(pool_size=(2,2))(ln)    
    
    #Branch 3 start
    conv_30 = Conv2D(numFilters*8, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(maxPool_2)
    conv_30 = Conv2D(numFilters*8, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_30)
    ln = LayerNormalization()(conv_30)
    maxPool_3 = MaxPooling2D(pool_size=(2,2))(ln)
    
    #Branch 3 finish
    conv_31 = Conv2D(numFilters*8, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(maxPool_3)
    conv_31 = Conv2D(numFilters*8, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_31)
    upscale_3 = Conv2D(numFilters*8, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(interpolation='nearest')(conv_31))
    ln = LayerNormalization()(upscale_3)
    
    #Branch 2 finish
    conc_2 = concatenate([ln, maxPool_2], axis=3)
    conv_21 = Conv2D(numFilters*4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_2)
    conv_21 = Conv2D(numFilters*4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_21)
    upscale_2 = Conv2D(numFilters*4, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(interpolation='nearest')(conv_21))
    ln = LayerNormalization()(upscale_2)
    
    #Branch 1 finish
    conc_1 = concatenate([ln, maxPool_1], axis=3)
    conv_11 = Conv2D(numFilters*2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_1)
    conv_11 = Conv2D(numFilters*2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_11)
    upscale_1 = Conv2D(numFilters*2, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(interpolation='nearest')(conv_11))
    ln = LayerNormalization()(upscale_1)
    
    #Branch 0 finish
    conc_0 = concatenate([ln, maxPool_0], axis=3)
    conv_01 = Conv2D(numFilters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conc_0)
    conv_01 = Conv2D(numFilters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv_01)
    upscale_0 = Conv2D(numFilters, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(interpolation='nearest')(conv_01))
    output = Conv2D(1, (1,1), activation='linear', padding='same', kernel_initializer='he_normal')(upscale_0)

    return tf.keras.Model(inputs=inputs, outputs=output)

#Extract features of interest from a reconstruction for propogation through network
def featureExtractor(maskObject, measuredImage):

    #Create blank image to build on for remaining visualized features
    featureImage = np.zeros([maskObject.imageHeight, maskObject.imageWidth])

    #Stack feature inputs as needed
    featureImage = np.dstack((featureImage, measuredImage))

    #Delete the blank first channel
    featureImage = np.delete(featureImage, 0, -1)
        
    return featureImage

def makeTensor(image):
    
    #Determine padding needed for the network architecture
    padX = int(np.ceil(image.shape[0]/depthFactor)*depthFactor) - image.shape[0]
    padY = int(np.ceil(image.shape[1]/depthFactor)*depthFactor) - image.shape[1]
    
    #NOTE- L2 (sklearn default) results in clustering of RD, (removed, as this is not a desired behavior; using min-max norm)
    #If there is more than one channel, normalize each on its own, then pad accordingly
    try:
        for channelNum in range(0, image.shape[2]): image[:, :, channelNum] = normalize(image[:, :, channelNum])
        image = np.pad(image, ((0, padX), (0, padY), (0,0)), mode='constant')
        image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    except:
        image = normalize(image)
        image = np.pad(image, ((0, padX), (0, padY)), mode='constant')
        image = image.reshape((1,image.shape[0],image.shape[1],1))

    #Convert for network input
    resultTensor = tf.convert_to_tensor(image)

    return resultTensor, [padX, padY]
