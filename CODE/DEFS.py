#==================================================================
#SLADS DEFINITIONS GENERAL
#==================================================================

#Perform summation and alignment over an mz range for a given line; mzFile cannot be pickled and passed into function
@ray.remote
def mzrange_parhelper(mzRange, scanFileName, mzMethod):
    data = mzFile(scanFileName)
    lineNum = int(scanFileName.split('line')[1].split('.')[0])-1
    
    if mzMethod == 'sum':
        lineData = np.asarray([np.sum(mz_range(data.scan(px, 'profile'), mzRange)) for px in range(data.scan_range()[0], data.scan_range()[1]+1)])
    elif mzMethod =='xic':
        lineData = np.asarray(list(map(list, data.xic(data.time_range()[0], data.time_range()[1], mzRange[0], mzRange[1]))))[:,1]
    else:
        sys.exit('Error! - mzMethod: ' + mzMethod + 'for: ' + scanFileName + 'has not been implemented')
 
    return lineNum, lineData

#All information pertaining to a sample
class Sample:
    def __init__(self, sampleFolder, ignoreMissingLines=False):
    
        #Should missing lines be ignored (to only be used in training and testing)
        self.ignoreMissingLines = ignoreMissingLines
    
        #Location of MSI data and sample name
        self.sampleFolder = sampleFolder + os.path.sep
        self.name = os.path.basename(sampleFolder)
    
        #Which files have already been read
        self.readScanFiles = []
        self.readLines = []

        #Read in data from sampleInfo.txt
        sampleInfo = open(self.sampleFolder+'sampleInfo.txt').readlines()
        self.numLines = int(sampleInfo[0].rstrip())
        self.lineExt = sampleInfo[1].rstrip()
        self.timeRes = float(sampleInfo[2].rstrip())
        self.maxTime = float(sampleInfo[3].rstrip())
        self.mzMethod = sampleInfo[4].rstrip()
        self.mzSpec = sampleInfo[5].rstrip()
        self.normMethod = sampleInfo[6].rstrip()
        if self.normMethod == 'standard' or self.mzSpec == 'value': self.mzTolerance = (float(sampleInfo[7].rstrip())*1e-03)/2
        
        #Read in all m/z locations/ranges (.csv)
        if self.mzSpec == 'value':
            mzLocations = np.loadtxt(self.sampleFolder+'mz.csv', delimiter=',')
            self.mzRanges = [[mzLocation-self.mzTolerance, mzLocation+self.mzTolerance] for mzLocation in mzLocations]
        elif self.mzSpec == 'range':
            self.mzRanges = np.loadtxt(self.sampleFolder+'mz.csv', delimiter=',')
                
        #Prepare variables to hold sample data
        self.mzWeights = np.ones(len(self.mzRanges))/len(self.mzRanges)
        self.numColumns = int(1/self.timeRes)
        self.mzImages = [np.zeros((self.numLines, self.numColumns)) for mzRangeNum in range(0, len(self.mzRanges))]
        self.measuredmzImages = copy.deepcopy(self.mzImages)
        self.newTimes = np.linspace(0, self.maxTime, self.numColumns)
        self.origTIC = [np.zeros(self.numColumns) for line in range(self.numLines+1)]
        self.origTimes = copy.deepcopy(self.origTIC) 

        #If using a standard, then also read in standards locations (.csv)
        if  self.normMethod == 'standard': 
            mzStandardLocations = np.loadtxt(self.sampleFolder+'mzStandards.csv', delimiter=',')
            if mzStandardLocations.shape != ():
                print('Warning! - Untested functionality, will sum the formed visualizations of the specified m/z ranges for the normalization')
                self.mzStandardRanges = [[mzStandardLocation-self.mzTolerance, mzStandardLocation+self.mzTolerance] for mzStandardLocation in mzStandardLocations]
            else:
                self.mzStandardRanges = [[mzStandardLocations-self.mzTolerance, mzStandardLocations+self.mzTolerance]]
            self.mzStandardImages = [copy.deepcopy(self.origTIC) for mzRangeNum in range(0, len(self.mzStandardRanges))]

        #Declare variables used in training/testing/implementation
        self.origNormArray = None
        self.normArray = None
        self.avgMeasuredImage = None
        self.avgGroundTruthImage = []
        self.maskObject = None
        self.resultsPath = ''
        self.measuredLines = []
        self.reconImage = None
        self.RDImage = None
        self.RDValues = None
        self.polyFeatures = None
        self.percMeasured = 0
        
    #Update mzImages, TIC, origTIC, and origTimes by information in the present line files
    def readScanData(self, lineRevistMethod):

        #Obtain and sort the available line files pertaining to the current scan
        scanFiles = natsort.natsorted(glob.glob(self.sampleFolder+'*'+self.lineExt), reverse=False)

        #Identify which files have not yet been scanned, if line revisiting is disabled (update not replace)
        if lineRevistMethod == False:
            scannedBool = [scanFile not in self.readScanFiles for scanFile in scanFiles]
            scanFiles = np.asarray(scanFiles)[scannedBool].tolist()

        #Read in each of the lines, obtain the TIC and original sampling locations (times)
        for scanFileName in scanFiles:
            
            #Add file name to those already scanned
            self.readScanFiles.append(scanFileName)

            #Establish file pointer and line number (1 indexed) for the specific scan
            data = mzFile(scanFileName)
            lineNum = int(scanFileName.split('line')[1].split('.')[0])-1
            self.readLines.append(lineNum)
            
            #Obtain the total ion chromatogram over all mz available
            xicData = np.asarray(data.xic(data.time_range()[0], data.time_range()[1]))
            self.origTimes[lineNum] = xicData[:,0]
            self.origTIC[lineNum] = xicData[:,1]

        #Check and setup for normalization
        if self.normMethod == 'tic':
            self.origNormArray = self.origTIC
        elif self.normMethod == 'standard':
            for mzRangeNum in range(0, len(self.mzStandardRanges)): 
                results = ray.get([mzrange_parhelper.remote(self.mzStandardRanges[mzRangeNum], scanFileName, self.mzMethod) for scanFileName in scanFiles])
                for result in results: self.mzStandardImages[mzRangeNum][result[0]] = result[1]
                self.origNormArray = np.sum(self.mzStandardImages, axis=0)
        elif self.normMethod == 'none':
            self.origNormArray = None
        else:
            sys.exit('Error! - Unknown normalization method: ' + self.normMethod + ' specified for sample: ' + self.name)

        #Align the normalization array for visualization
        self.normArray = np.asarray([np.interp(self.newTimes, self.origTimes[lineNum], self.origNormArray[lineNum]) for lineNum in range(0, self.numLines)])

        #For each mzRange generate a visualization
        for mzRangeNum in range(0, len(self.mzRanges)): 
            results = ray.get([mzrange_parhelper.remote(self.mzRanges[mzRangeNum], scanFileName, self.mzMethod) for scanFileName in scanFiles])
            for result in results: 
                #Perform normalization if specified
                if self.normMethod != 'none': 
                    self.mzImages[mzRangeNum][result[0]] = np.interp(self.newTimes, self.origTimes[result[0]], np.nan_to_num(result[1]/self.origNormArray[result[0]], nan=0, posinf=0, neginf=0))
                else:
                    self.mzImages[mzRangeNum][result[0]] = np.interp(self.newTimes, self.origTimes[result[0]], result[1])
        
        #If missing lines are to be ignored, then delete them
        if self.ignoreMissingLines: 
            missingLines = list(set(np.arange(1, self.numLines).tolist()) - set(self.readLines))
            self.mzImages = np.delete(self.mzImages, missingLines, axis=1)
            if self.normMethod == 'standard': self.mzStandardImages = np.delete(self.mzStandardImages, missingLines, axis=1)
            self.origTIC = np.delete(self.origTIC, missingLines, axis=0)
            self.origTimes = np.delete(self.origTIC, missingLines, axis=0)
            self.origNormArray = np.delete(self.origNormArray, missingLines, axis=0)
            self.normArray = np.delete(self.normArray, missingLines, axis=0)
            self.numLines -= len(missingLines)
        
    #Scan new locations in the sample
    def performMeasurements(self, newIdxs, simulationFlag):

        #Update the maskObject according to the newIdxs
        if len(newIdxs) != 0: self.maskObject.update(newIdxs)

        #If this is not a simulation then inform equipment what points to scan, wait, read in new data, update images
        if not simulationFlag:
            with open('./INPUT/IMP/UNLOCK', 'w') as filehandle: filehandle.writelines(str(tuple(newIdxs[0])) + ', ' + str(tuple(newIdxs[len(newIdxs)-1])))
            equipWait()
            self.readScanData()
            self.measuredmzImages = copy.deepcopy(self.mzImages)
        else:
            #Obtain masked values from the stored image information
            for imageNum in range(0,len(self.measuredmzImages)):
                temp = np.asarray(self.mzImages[imageNum]).copy()
                temp[self.maskObject.mask == 0] = 0
                self.measuredmzImages[imageNum] = temp.copy()

        #Perform averaging of the multiple channels and subsequent image normalization
        self.avgMeasuredImage = np.average(np.asarray(self.measuredmzImages), axis=0, weights=self.mzWeights)
        self.avgMeasuredImage = MinMaxScaler().fit_transform(self.avgMeasuredImage.reshape(-1, 1)).reshape(self.avgMeasuredImage.shape)
        
        #Update percentage pixels measured
        self.percMeasured = (np.sum(self.maskObject.mask)/self.maskObject.area)*100

#Singular result generated through runSLADS
class Result():
    def __init__(self, sample, maskObject, avgGroundTruthImage, bestCFlag, oracleFlag, simulationFlag, animationFlag):
        self.sample = copy.deepcopy(sample)
        self.avgGroundTruthImage = copy.deepcopy(avgGroundTruthImage)
        self.simulationFlag = copy.deepcopy(simulationFlag)
        self.animationFlag = copy.deepcopy(animationFlag)
        self.bestCFlag = copy.deepcopy(bestCFlag)
        self.oracleFlag = copy.deepcopy(oracleFlag)
        self.avgImages = []
        self.reconImages = []
        self.RDImages = []
        self.mzRecons = []
        self.samples = []
        self.ERDValueNPs = []
        
        self.MSEList = []
        self.SSIMList = []
        self.PSNRList = []
        self.TDList = []
        self.ERDPSNRList = []   
        
        self.percMeasuredList = []

    def update(self, percMeasured, sample, maskObject, reconImage, ERDValuesNP, iterNum):

        #Save the model development
        self.ERDValueNPs.append(copy.deepcopy(ERDValuesNP))
        self.avgImages.append(copy.deepcopy(sample.avgMeasuredImage))
        self.samples.append(copy.deepcopy(sample))
        self.percMeasuredList.append(copy.deepcopy(percMeasured))
        self.reconImages.append(copy.deepcopy(reconImage))
    
    def complete(self, optimalC): 
        if self.simulationFlag:

            #Perform statistics extraction for all images
            for index in range(0, len(self.reconImages)):
                
                #Measure and save statistics
                difference = np.sum(computeDifference(self.avgGroundTruthImage, self.reconImages[index]))
                TD = difference/self.samples[index].maskObject.area
                MSE = mean_squared_error(self.avgGroundTruthImage, self.reconImages[index])
                SSIM = structural_similarity(self.avgGroundTruthImage, self.reconImages[index], data_range=1)
                PSNR = compare_psnr(self.avgGroundTruthImage, self.reconImages[index], data_range=1)

                self.TDList.append(TD)
                self.MSEList.append(MSE)
                self.SSIMList.append(SSIM)
                self.PSNRList.append(PSNR)
                
            #If determining the best c, return the area under the PSNR curve
            if self.bestCFlag: return self.PSNRList, self.percMeasuredList

            #Calculate the actual RD Image; bestCFlag data should be returned before this subroutine
            #currMaxRD = 0
            for index in tqdm(range(0, len(self.samples)), desc='RD Calc', leave = False, ascii=True):
                RDImage = computeRD(self.samples[index], optimalC)
                self.RDImages.append(RDImage)
                self.ERDPSNRList.append(compare_psnr(RDImage, self.ERDValueNPs[index], data_range=1))
                #if np.max(RDImage) > currMaxRD: currMaxRD = np.max(RDImage)
            
            #Normalize RD Images across sampling series and find ERD PSNR difference
            #for index in range(0, len(self.samples)): 
            #    self.RDImages[index] = self.RDImages[index]/currMaxRD
            #    self.ERDPSNRList.append(compare_psnr(self.RDImages[index], self.ERDValueNPs[index], data_range=1))
            
        #If an animation will be produced and the run has completed
        if self.animationFlag:

            #Setup directory addresses
            dir_mzResults = self.samples[-1].resultsPath + 'mzResults/'
            dir_mzSampleResults = dir_mzResults + self.samples[-1].name + '/'

            dir_Animations = self.samples[-1].resultsPath+ 'Animations/'
            dir_AnimationVideos = dir_Animations + 'Videos/'
            dir_AnimationFrames = dir_Animations + self.samples[-1].name + '/'

            #Clean sub-directories
            if os.path.exists(dir_AnimationFrames): shutil.rmtree(dir_AnimationFrames)
            os.makedirs(dir_AnimationFrames)

            if os.path.exists(dir_mzSampleResults): shutil.rmtree(dir_mzSampleResults)
            os.makedirs(dir_mzSampleResults)

            #If this was a simulation
            if self.simulationFlag:
                
                #Save each of the individual mass range reconstructions
                percSampled = "{:.2f}".format(self.percMeasuredList[-1])
                
                #Perform reconstructions and statistics extraction for all images
                results = ray.get([performRecon_parhelper.remote(self.samples[-1].measuredmzImages[mzImageNum], self.samples[-1].maskObject) for mzImageNum in range(0,len(self.samples[-1].measuredmzImages))])
                self.mzRecons = [result for result in results]

                for massNum in tqdm(range(0, len(self.samples[-1].mzRanges)), desc='mz Images', leave = False, ascii=True):
                    
                    #mz image with only the actual measurements made
                    subMeasuredImage = self.samples[-1].measuredmzImages[massNum]
                    
                    #Retrieve reconstruction for the specific mz image
                    subReconImage = self.mzRecons[massNum]

                    #Retrieve ground truth for the specific mz image
                    mzImage = self.samples[-1].mzImages[massNum]
                    
                    #MSE relative to the measured
                    measureMSE = '{:.2e}'.format(mean_squared_error(mzImage, subMeasuredImage))
                    
                    #MSE relative to the reconstruction
                    reconMSE = '{:.2e}'.format(mean_squared_error(mzImage, subReconImage))

                    #Mass range string
                    massRange = str(self.samples[-1].mzRanges[massNum][0]) + '-' + str(self.samples[-1].mzRanges[massNum][1])

                    mzMaxValue = np.max([np.max(mzImage), np.max(subReconImage)])

                    #Measured mz image
                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(40,8))
                    f.subplots_adjust(top = 0.80)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle('Percent Sampled: ' + percSampled + '  Measurement mz: ' + massRange + '\nMeasured MSE: ' + measureMSE + '    Reconstruction MSE : ' + reconMSE, fontsize=20, fontweight='bold', y = 0.95)

                    sub = f.add_subplot(1,4,1)
                    sub.imshow(self.samples[-1].maskObject.mask, cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')

                    sub = f.add_subplot(1,4,2)
                    sub.imshow(mzImage, cmap='hot', aspect='auto', vmin=0, vmax=mzMaxValue)
                    sub.set_title('Ground-Truth')

                    sub = f.add_subplot(1,4,3)
                    sub.imshow(subMeasuredImage, cmap='hot', aspect='auto', vmin=0, vmax=mzMaxValue)
                    sub.set_title('Measured')

                    sub = f.add_subplot(1,4,4)
                    sub.imshow(subReconImage, cmap='hot', aspect='auto', vmin=0, vmax=mzMaxValue)
                    sub.set_title('Reconstruction')

                    saveLocation = dir_mzSampleResults + massRange +'.png'

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()

                #Generate each of the frames
                for i in tqdm(range(0, len(self.samples)), desc='Result Images', leave = False, ascii=True):
                    
                    minERDRDValue, maxERDRDValue = np.min([self.ERDValueNPs[i], self.RDImages[i]]), np.max([self.ERDValueNPs[i], self.RDImages[i]])
                    
                    saveLocation = dir_AnimationFrames + 'progression_' + self.samples[-1].name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'

                    f = plt.figure(figsize=(35,15))
                    f.subplots_adjust(top = 0.85)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle("Percent Sampled: %.2f,  Iteration: %.0f\nRecon PSNR: %.2f, ERD PSNR: %.2f" % (self.percMeasuredList[i], i+1, self.PSNRList[i], self.ERDPSNRList[i]), fontsize=20, fontweight='bold', y = 0.95)

                    ax1 = plt.subplot2grid(shape=(2,3), loc=(0,0))
                    im = ax1.imshow(self.avgGroundTruthImage, cmap='hot', aspect='auto', vmin=0, vmax=1)
                    ax1.set_title('Ground-Truth')
                    cbar = f.colorbar(im, ax=ax1, orientation='vertical', pad=0.01)

                    ax2 = plt.subplot2grid((2,3), (0,1))
                    im = ax2.imshow(self.reconImages[i], cmap='hot', aspect='auto', vmin=0, vmax=1)
                    ax2.set_title('Reconstruction')
                    cbar = f.colorbar(im, ax=ax2, orientation='vertical', pad=0.01)

                    ax3 = plt.subplot2grid((2,3), (0,2))
                    im = ax3.imshow(abs(self.avgGroundTruthImage-self.reconImages[i]), cmap='hot', aspect='auto', vmin=0, vmax=1)
                    ax3.set_title('Absolute Difference')
                    cbar = f.colorbar(im, ax=ax3, orientation='vertical', pad=0.01)

                    ax4 = plt.subplot2grid((2,3), (1,0))
                    im = ax4.imshow(self.samples[i].maskObject.mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
                    ax4.set_title('Measurement Mask')
                    cbar = f.colorbar(im, ax=ax4, orientation='vertical', pad=0.01)
                    
                    ax5 = plt.subplot2grid((2,3), (1,1))
                    im = ax5.imshow(self.ERDValueNPs[i], cmap='viridis', vmin=minERDRDValue, vmax=maxERDRDValue, aspect='auto')
                    ax5.set_title('ERD')
                    cbar = f.colorbar(im, ax=ax5, orientation='vertical', pad=0.01)

                    ax6 = plt.subplot2grid((2,3), (1,2))
                    im = ax6.imshow(self.RDImages[i], cmap='viridis', vmin=minERDRDValue, vmax=maxERDRDValue, aspect='auto')
                    ax6.set_title('RD')
                    cbar = f.colorbar(im, ax=ax6, orientation='vertical', pad=0.01)

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()
                    #=====================

                    #No border saves
                    #=====================
                    #Save the reconstruction, no borders
                    saveLocation = dir_AnimationFrames + 'reconstruction_' + self.samples[-1].name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.reconImages[i], cmap='hot', aspect='auto', vmin=0, vmax=1)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()

                    #Save the ERD, no borders
                    saveLocation = dir_AnimationFrames + 'erd_' + self.samples[-1].name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.ERDValueNPs[i], aspect='auto', vmin=minERDRDValue, vmax=maxERDRDValue)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()
                    
                    #Save the RD, no borders
                    saveLocation = dir_AnimationFrames + 'rd_' + self.samples[-1].name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.RDImages[i], aspect='auto', vmin=minERDRDValue, vmax=maxERDRDValue)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()

                    #Save the mask, no borders
                    saveLocation = dir_AnimationFrames + 'mask_' + self.samples[-1].name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.samples[i].maskObject.mask, cmap='gray', aspect='auto')
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()
                    #=====================
                
                #Combine images into animation
                dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'progression_*.png'))
                height, width, layers = cv2.imread(dataFileNames[0]).shape
                animation = cv2.VideoWriter(dir_AnimationVideos + 'progression_' + self.samples[-1].name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
                for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                animation.release()
                animation = None
                                
                #Save the averaged ground truth, no borders
                saveLocation = dir_AnimationFrames + 'final_groundTruth_' + self.samples[-1].name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.avgGroundTruthImage, cmap='hot', aspect='auto', vmin=0, vmax=1)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()

                #Save the averaged final reconstruction, no borders
                saveLocation = dir_AnimationFrames + 'final_reconstruction_' + self.samples[-1].name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.reconImages[-1], cmap='hot', aspect='auto', vmin=0, vmax=1)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()

                #Save the final mask, no borders
                saveLocation = dir_AnimationFrames + 'final_mask_' + self.samples[-1].name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.samples[-1].maskObject.mask, cmap='gray', aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()
                
            else: #Not a simulation
                print('Warning! - Non simulation plots not yet fixed for color issue and modified selection')
        if self.simulationFlag:

            #Perform statistics extraction for all images
            for index in range(0, len(self.reconImages)):
                
                #Measure and save statistics
                difference = np.sum(computeDifference(self.avgGroundTruthImage, self.reconImages[index]))
                TD = difference/self.samples[index].maskObject.area
                MSE = mean_squared_error(self.avgGroundTruthImage, self.reconImages[index])
                SSIM = structural_similarity(self.avgGroundTruthImage, self.reconImages[index], data_range=1)
                PSNR = compare_psnr(self.avgGroundTruthImage, self.reconImages[index], data_range=1)

                self.TDList.append(TD)
                self.MSEList.append(MSE)
                self.SSIMList.append(SSIM)
                self.PSNRList.append(PSNR)
                
            #If determining the best c, return the area under the PSNR curve
            if self.bestCFlag: return self.PSNRList, self.percMeasuredList

            #Calculate the actual RD Image; bestCFlag data should be returned before this subroutine
            #currMaxRD = 0
            for index in tqdm(range(0, len(self.samples)), desc='RD Calc', leave = False, ascii=True):
                RDImage = computeRD(self.samples[index], optimalC)
                self.RDImages.append(RDImage)
                self.ERDPSNRList.append(compare_psnr(RDImage, self.ERDValueNPs[index], data_range=1))
                #if np.max(RDImage) > currMaxRD: currMaxRD = np.max(RDImage)
            
            #Normalize RD Images across sampling series and find ERD PSNR difference
            #for index in range(0, len(self.samples)): 
            #    self.RDImages[index] = self.RDImages[index]/currMaxRD
            #    self.ERDPSNRList.append(compare_psnr(self.RDImages[index], self.ERDValueNPs[index], data_range=1))
            
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
                
                #Perform reconstructions and statistics extraction for all images
                results = ray.get([performRecon_parhelper.remote(self.sample.measuredmzImages[mzImageNum], self.samples[-1].maskObject) for mzImageNum in range(0,len(self.sample.measuredmzImages))])
                self.mzRecons = [result for result in results]

                for massNum in tqdm(range(0, len(self.sample.mzRanges)), desc='mz Images', leave = False, ascii=True):
                    
                    #mz image with only the actual measurements made
                    subMeasuredImage = self.sample.measuredmzImages[massNum].astype("float")
                    
                    #Retreive reconstruction for the specific mz image
                    subReconImage = self.mzRecons[massNum]

                    #Retreive ground truth for the specific mz image
                    mzImage = self.sample.mzImages[massNum].astype("float")
                    
                    #MSE relative to the measured
                    measureMSE = "{:.2f}".format(mean_squared_error(mzImage, subMeasuredImage))
                    
                    #MSE relative to the reconstruction
                    reconMSE = "{:.2f}".format(mean_squared_error(mzImage, subReconImage))

                    #Mass range string
                    massRange = str(self.sample.mzRanges[massNum][0]) + '-' + str(self.sample.mzRanges[massNum][1])

                    mzMaxValue = np.max([np.max(mzImage), np.max(subReconImage)])

                    #Measured mz image
                    font = {'size' : 18}
                    plt.rc('font', **font)
                    f = plt.figure(figsize=(25,10))
                    f.subplots_adjust(top = 0.80)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle('Percent Sampled: ' + percSampled + '  Measurement mz: ' + massRange + '\nMeasured MSE: ' + measureMSE + ' Reconstruction MSE : ' + reconMSE, fontsize=20, fontweight='bold', y = 0.95)

                    sub = f.add_subplot(1,4,1)
                    sub.imshow(self.samples[-1].maskObject.mask, cmap='gray', aspect='auto')
                    sub.set_title('Sampled Mask')

                    sub = f.add_subplot(1,4,2)
                    sub.imshow(mzImage, cmap='hot', aspect='auto', vmin=0, vmax=mzMaxValue)
                    sub.set_title('Ground-Truth')

                    sub = f.add_subplot(1,4,3)
                    sub.imshow(subMeasuredImage, cmap='hot', aspect='auto', vmin=0, vmax=mzMaxValue)
                    sub.set_title('Measured')

                    sub = f.add_subplot(1,4,4)
                    sub.imshow(subReconImage, cmap='hot', aspect='auto', vmin=0, vmax=mzMaxValue)
                    sub.set_title('Reconstruction')

                    saveLocation = dir_mzSampleResults + massRange +'.png'

                    plt.savefig(saveLocation, bbox_inches='tight')
                    plt.close()

                #Generate each of the frames
                for i in tqdm(range(0, len(self.samples)), desc='Result Images', leave = False, ascii=True):
                    
                    minERDRDValue, maxERDRDValue = np.min([self.ERDValueNPs[i], self.RDImages[i]]), np.max([self.ERDValueNPs[i], self.RDImages[i]])
                    
                    saveLocation = dir_AnimationFrames + 'progression_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'

                    f = plt.figure(figsize=(35,15))
                    f.subplots_adjust(top = 0.85)
                    f.subplots_adjust(wspace=0.15, hspace=0.2)
                    plt.suptitle("Percent Sampled: %.2f,  Iteration: %.0f\nRecon PSNR: %.2f, ERD PSNR: %.2f" % (self.percMeasuredList[i], i+1, self.PSNRList[i], self.ERDPSNRList[i]), fontsize=20, fontweight='bold', y = 0.95)

                    ax1 = plt.subplot2grid(shape=(2,3), loc=(0,0))
                    im = ax1.imshow(self.avgGroundTruthImage, cmap='hot', aspect='auto', vmin=0, vmax=1)
                    ax1.set_title('Ground-Truth')
                    cbar = f.colorbar(im, ax=ax1, orientation='vertical', pad=0.01)

                    ax2 = plt.subplot2grid((2,3), (0,1))
                    im = ax2.imshow(self.reconImages[i], cmap='hot', aspect='auto', vmin=0, vmax=1)
                    ax2.set_title('Reconstruction')
                    cbar = f.colorbar(im, ax=ax2, orientation='vertical', pad=0.01)

                    ax3 = plt.subplot2grid((2,3), (0,2))
                    im = ax3.imshow(abs(self.avgGroundTruthImage-self.reconImages[i]), cmap='hot', aspect='auto', vmin=0, vmax=1)
                    ax3.set_title('Absolute Difference')
                    cbar = f.colorbar(im, ax=ax3, orientation='vertical', pad=0.01)

                    ax4 = plt.subplot2grid((2,3), (1,0))
                    im = ax4.imshow(self.samples[i].maskObject.mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
                    ax4.set_title('Measurement Mask')
                    cbar = f.colorbar(im, ax=ax4, orientation='vertical', pad=0.01)
                    
                    ax5 = plt.subplot2grid((2,3), (1,1))
                    im = ax5.imshow(self.ERDValueNPs[i], cmap='viridis', vmin=minERDRDValue, vmax=maxERDRDValue, aspect='auto')
                    ax5.set_title('ERD')
                    cbar = f.colorbar(im, ax=ax5, orientation='vertical', pad=0.01)

                    ax6 = plt.subplot2grid((2,3), (1,2))
                    im = ax6.imshow(self.RDImages[i], cmap='viridis', vmin=minERDRDValue, vmax=maxERDRDValue, aspect='auto')
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
                    plt.imshow(self.reconImages[i], cmap='hot', aspect='auto', vmin=0, vmax=1)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()

                    #Save the ERD, no borders
                    saveLocation = dir_AnimationFrames + 'erd_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.ERDValueNPs[i], aspect='auto', vmin=minERDRDValue, vmax=maxERDRDValue)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()
                    
                    #Save the RD, no borders
                    saveLocation = dir_AnimationFrames + 'rd_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.RDImages[i], aspect='auto', vmin=minERDRDValue, vmax=maxERDRDValue)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()

                    #Save the mask, no borders
                    saveLocation = dir_AnimationFrames + 'mask_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.samples[i].maskObject.mask, cmap='gray', aspect='auto')
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()
                    #=====================
                
                #Combine images into animation
                dataFileNames = natsort.natsorted(glob.glob(dir_AnimationFrames + 'progression_*.png'))
                height, width, layers = cv2.imread(dataFileNames[0]).shape
                animation = cv2.VideoWriter(dir_AnimationVideos + 'progression_' + self.sample.name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
                for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                animation.release()
                animation = None
                                
                #Save the averaged ground truth, no borders
                saveLocation = dir_AnimationFrames + 'final_groundTruth_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.avgGroundTruthImage, cmap='hot', aspect='auto', vmin=0, vmax=1)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()

                #Save the averaged final reconstruction, no borders
                saveLocation = dir_AnimationFrames + 'final_reconstruction_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.reconImages[-1], cmap='hot', aspect='auto', vmin=0, vmax=1)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()

                #Save the final mask, no borders
                saveLocation = dir_AnimationFrames + 'final_mask_' + self.sample.name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.samples[-1].maskObject.mask, cmap='gray', aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()
                
            else: #Not a simulation
                print('Warning! - Non simulation plots not yet fixed for color issue and modified selection')

#Each sample needs a mask object
class MaskObject():
    def __init__(self, imageWidth, imageHeight, initialPercToScan, scanMethod):
        self.imageWidth = copy.deepcopy(imageWidth)
        self.imageHeight = copy.deepcopy(imageHeight)
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
                
                if lineMethod == 'percLine':
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

def iou(groundTruth, prediction):
    return np.sum(np.logical_and(groundTruth, prediction)) / np.sum(np.logical_or(groundTruth, prediction))

@ray.remote
def performRecon_parhelper(image, maskObject):
    return performRecon(image, maskObject)

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

def computeRD(sample, cValue):

    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(sample.maskObject)
    
    #Calculate the sigma value for chosen c value, store directly into shared memory
    sigmaValues_id = ray.put(neighborDistances[:,0]/cValue)
    
    #Compute the difference between the original and reconstructed images, store directly into shared memory
    RDPP_id = ray.put(computeDifference(sample.avgGroundTruthImage, sample.reconImage))
    
    #Split computation sets to be run on multiple `
    indexes = np.asarray(list(range(0, np.sum(sample.maskObject.mask==0))))
    blockSize = int(np.ceil(len(indexes) / float(multiprocessing.cpu_count())))
    indexSets = np.split(indexes, np.arange(blockSize, len(indexes), blockSize))
    
    #Store indexes into shared memory
    unMeasuredIdxs_id = ray.put(sample.maskObject.unMeasuredIdxs)
    
    #Store aspect ratio into shared memory
    maskObject_id = ray.put(sample.maskObject)
    
    #Perform computation of RD values for each unmeasured point
    results = ray.get([gaussian_parhelper.remote(unMeasuredIdxs_id, RDPP_id, sigmaValues_id, maskObject_id, indexes) for indexes in indexSets])
    RDValues = [result for resultSet in results for result in resultSet]
    
    #Reassemble the values into a single image
    RDImage = np.zeros((sample.avgGroundTruthImage.shape))
    RDImage[np.where(sample.maskObject.mask==0)] = RDValues
    
    #Normalize
    RDImage = MinMaxScaler().fit_transform(RDImage.reshape(-1, 1)).reshape(RDImage.shape)

    return RDImage

def computePolyFeatures(maskObject, reconImage):
    
    #Retreive recon values
    inputValues = reconImage[maskObject.unMeasuredIdxs[:,0], maskObject.unMeasuredIdxs[:,1]]
    
    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
    measuredValues = reconImage[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(maskObject)
    neighborValues = measuredValues[neighborIndices]
    
    #Create array to hold features
    feature = np.zeros((np.shape(maskObject.unMeasuredIdxs)[0],6))
    
    #Compute std div features
    diffVect = computeDifference(neighborValues, np.transpose(np.matlib.repmat(inputValues, np.shape(neighborValues)[1],1)))
    feature[:,0] = np.sum(neighborWeights*diffVect, axis=1)
    feature[:,1] = np.sqrt(np.sum(np.power(diffVect,2),axis=1))
    
    #Compute distance/density features
    cutoffDist = np.ceil(np.sqrt((featDistCutoff/100)*(maskObject.area/np.pi)))
    feature[:,2] = neighborDistances[:,0]
    neighborsInCircle = np.sum(neighborDistances<=cutoffDist,axis=1)
    feature[:,3] = (1+(np.pi*(np.square(cutoffDist))))/(1+neighborsInCircle)
    
    #Compute gradient features; assume continuous features
    gradientImageX, gradientImageY = np.gradient(reconImage)
    feature[:,4] = abs(gradientImageY)[maskObject.unMeasuredIdxs[:,0], maskObject.unMeasuredIdxs[:,1]]
    feature[:,5] = abs(gradientImageX)[maskObject.unMeasuredIdxs[:,0], maskObject.unMeasuredIdxs[:,1]]
    
    #Fit polynomial features to the determined array
    polyFeatures = PolynomialFeatures(degree=2).fit_transform(feature)
    
    return polyFeatures

def computeERD(iterNum, sample, model):
    
    if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net':
        
        #Compute ERD values
        polyFeatures = computePolyFeatures(sample.maskObject, sample.reconImage)
        ERDValues = model.predict(polyFeatures)
        
        #Rearrange ERD values into array; those that have already been measured have 0 ERD
        ERD = np.zeros([sample.maskObject.imageHeight, sample.maskObject.imageWidth])
        ERD[sample.maskObject.unMeasuredIdxs[:, 0], sample.maskObject.unMeasuredIdxs[:, 1]] = ERDValues
        
        #Remove values that are less than those already scanned (0 ERD)
        ERD[np.where((ERD < 0))] = 0
    
    elif erdModel == 'DLADS':
        
        #Form measured image
        measuredImage = np.zeros((sample.maskObject.mask.shape))
        measuredImage[np.where(sample.maskObject.mask)] = sample.avgMeasuredImage[np.where(sample.maskObject.mask)]

        #Compute feature image for network input
        featureImage = featureExtractor(sample.maskObject, measuredImage, sample.reconImage)

        #Send input through trained model
        inputImage, originalShape = makeCompatible(featureImage)
        ERD = model.predict(inputImage, steps=1)[0,:,:,0]
        
        #Revert to the original shape
        ERD = resize(ERD, (originalShape), order=0)

        #Ensure points already measured and those with values less than those have a 0 ERD
        ERD[sample.maskObject.measuredIdxs[:, 0], sample.maskObject.measuredIdxs[:, 1]] = 0
        ERD[np.where((ERD < 0))] = 0
    
    return ERD
    
def runSLADS(samples, model, scanMethod, cValue, percToScan, stopPerc, sampleNum, simulationFlag, trainPlotFlag, animationFlag, tqdmHide, oracleFlag, bestCFlag):

    if simulationFlag: #Here sample.mzImages contains the full ground-truth
        sample = samples[sampleNum]        
    else: #Here sample.mzImages contains only the initially measured images
        sample = samples
        sample.measuredmzImages = sample.mzImages
        
    #Reinitialize the mask to starting state
    sample.maskObject.reset(simulationFlag)

    #Indicate the stopping condition has not yet been met
    completedRunFlag = False

    #Set the current iteration
    iterNum = 1

    #Perform the initial measurements
    if simulationFlag: sample.performMeasurements(sample.maskObject.initialMeasuredIdxs, simulationFlag)

    #Calculate the reconstruction image
    sample.reconImage = performRecon(sample.avgMeasuredImage, sample.maskObject)
    
    #Determine ERD or use the RD as the ERD if the oracleFlag or bestCFlag is enabled
    if oracleFlag or bestCFlag:
        ERDValuesNP = computeRD(sample, cValue)
    else:
        ERDValuesNP = computeERD(iterNum, sample, model)

    #Initialize and perform first update for a result object
    result = Result(sample, sample.maskObject, sample.avgGroundTruthImage, bestCFlag, oracleFlag, simulationFlag, animationFlag)
    result.update(sample.percMeasured, sample, sample.maskObject, sample.reconImage, ERDValuesNP, iterNum)

    #Check stopping criteria, just in case of a bad input
    if (scanMethod == 'pointwise' or not lineVisitAll) and (round(sample.percMeasured) >= stopPerc): completedRunFlag = True
    if scanMethod == 'linewise' and len(sample.maskObject.linesToScan) == 0: completedRunFlag = True

    #Until the stopping criteria has been met
    with tqdm(total = float(100), desc = '% Sampled', leave = False, ascii=True, disable=tqdmHide) as pbar:

        #Initialize progress bar state according to % measured
        pbar.n = round(sample.percMeasured,2)
        pbar.refresh()

        #Until the program has completed
        while not completedRunFlag:
            
            #Step the iteration counter
            iterNum += 1
            
            #Find next measurement locations
            newIdxs = findNewMeasurementIdxs(sample, ERDValuesNP, percToScan, scanMethod)
            
            #Perform measurements
            sample.performMeasurements(newIdxs, simulationFlag)
            
            #Check stopping conditions
            if (scanMethod == 'pointwise' or not lineVisitAll) and (round(sample.percMeasured) >= stopPerc): completedRunFlag = True
            if scanMethod == 'linewise' and len(sample.maskObject.linesToScan) == 0: completedRunFlag = True

            #Calculate the reconstruction
            sample.reconImage = performRecon(sample.avgMeasuredImage, sample.maskObject)
            
            #Determine ERD or use the RD as the ERD if the oracleFlag or bestCFlag is enabled
            if oracleFlag or bestCFlag:
                ERDValuesNP = computeRD(sample, cValue)
            else:
                ERDValuesNP = computeERD(iterNum, sample, model)

            #Update the result
            result.update(sample.percMeasured, sample, sample.maskObject, sample.reconImage, ERDValuesNP, iterNum)

            #Update the progress bar
            pbar.n = round(sample.percMeasured,2)
            pbar.refresh()

    return result

def findNewMeasurementIdxs(sample, ERDValuesNP, percToScan, scanMethod):

    #Make sure ERDValuesNP is in np array
    ERDValuesNP = np.asarray(ERDValuesNP)
    
    if scanMethod == 'random':
        newIdxs = np.asarray(random.sample(sample.maskObject.unMeasuredIdxs.tolist(), int((percToScan/100)*sample.maskObject.area)))
    elif scanMethod == 'pointwise':
        
        #Obtain a list of all ERD values for unmeasured locations
        ERDValueList = [ERDValuesNP[tuple(pt)] for pt in sample.maskObject.unMeasuredIdxs]
        
        #Sort the values in reverse order and choose the top 1% of values
        newIdxs = sample.maskObject.unMeasuredIdxs[np.argsort(ERDValueList)][::-1][:int((percToScan/100)*sample.maskObject.area)]

    elif scanMethod == 'linewise':
        #==========================================
        #OPTIMAL LINE DETERMINATION
        #==========================================
        
        lineERDSums = [np.nansum(ERDValuesNP[tuple([x[0] for x in line]), tuple([y[1] for y in line])]) for line in sample.maskObject.linesToScan]
        
        #Choose the line with maximum ERD and extract the actual indices
        lineToScanIdx = np.nanargmax(lineERDSums)
        lineToScanIdxs = sample.maskObject.linesToScan[lineToScanIdx]
        
        #Set the default new Idxs to be all of the indexes identified
        newIdxs = lineToScanIdxs.copy()
        
        #Obtain the ERD values in the chosen line
        lineERDValues = [ERDValuesNP[tuple(pt)] for pt in newIdxs]
        
        #==========================================
        
        #==========================================
        #PARTIAL LINE BY PERCENT
        #==========================================
        #Scan stopPerc locations on the line with maximized ERD
        if lineMethod == 'percLine':
            newIdxs = np.asarray(newIdxs)[np.argsort(lineERDValues)][::-1][:int((stopPerc/100)*len(lineERDValues))]
            newIdxs = newIdxs.tolist()
        #==========================================
        
        #==========================================
        #PARTIAL LINE BY START/END POINTS
        #==========================================
        #Choose segment to scan on line which contains at least stopPerc locations with maximal ERD
        if lineMethod == 'startEndPoints':
            newIdxs = np.asarray(newIdxs)[np.argsort(lineERDValues)][::-1][:int((stopPerc/100)*len(lineERDValues))]
            orderedNewIdxs = newIdxs[np.argsort(newIdxs[:,0]*newIdxs[:,1])]
            startLocation, endLocation = orderedNewIdxs[0], orderedNewIdxs[len(orderedNewIdxs)-1]
            newIdxs = np.asarray(lineToScanIdxs[lineToScanIdxs.index(startLocation.tolist()):lineToScanIdxs.index(endLocation.tolist())])
            
        #==========================================
        
        #==========================================
        #SELECTION SAFEGUARD
        #==========================================
        #If there are not enough locations selected, just scan the whole remainder of the line with the greatest ERD; ensures model will reach termination
        if len(newIdxs) < int(0.01*len(lineERDValues)): newIdxs = np.asarray(lineToScanIdxs).tolist()
        #==========================================

        #==========================================
        #POINT CONSIDERATION UPDATE
        #==========================================
        #Remove the selected points from further consideration, allows revisting lines
        if lineRevistMethod:
            sample.maskObject.delPoints(newIdxs)
        else:
            #Remove the line selected from further consideration, does not allow revisiting
            sample.maskObject.delLine(lineToScanIdx)
        #==========================================

    return newIdxs

def findNeighbors(maskObject):

    #Penalize neighbor distances according to the image aspect ratio
    if asymPenalty:
        unMeasuredIdxs = np.copy(maskObject.unMeasuredIdxs)
        measuredIdxs = np.copy(maskObject.measuredIdxs)
        if maskObject.imageWidth>maskObject.imageHeight:
            unMeasuredIdxs[:,0] = unMeasuredIdxs[:,0]*maskObject.aspectRatio
            measuredIdxs[:,0] = measuredIdxs[:,0]*maskObject.aspectRatio
        elif maskObject.imageWidth<maskObject.imageHeight:
            unMeasuredIdxs[:,1] = unMeasuredIdxs[:,1]*maskObject.aspectRatio
            measuredIdxs[:,1] = measuredIdxs[:,1]*maskObject.aspectRatio

    #Calculate knn
    neigh = NearestNeighbors(n_neighbors=numNeighbors)
    neigh.fit(measuredIdxs)
    neighborDistances, neighborIndices = neigh.kneighbors(unMeasuredIdxs)

    #Determine inverse distance weights
    unNormNeighborWeights = 1.0/np.square(neighborDistances)
    neighborWeights = unNormNeighborWeights/(np.sum(unNormNeighborWeights, axis=1))[:, np.newaxis]

    return neighborIndices, neighborWeights, neighborDistances

#Perform the reconstruction without 0-padding
def performRecon(inputImage, maskObject):

    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(maskObject)

    #Create a blank image for the reconstruction
    reconImage = np.zeros((maskObject.imageHeight, maskObject.imageWidth))

    #Retreive measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in maskObject.measuredIdxs]))
    measuredValues = inputImage[np.asarray(idxsX), np.asarray(idxsY)]

    #Compute reconstruction values using IDW (inverse distance weighting)
    reconImage[maskObject.unMeasuredIdxs[:,0], maskObject.unMeasuredIdxs[:,1]] = np.sum(measuredValues[neighborIndices]*neighborWeights, axis=1)

    #Combine measured values back into the reconstruction image
    reconImage[maskObject.measuredIdxs[:,0], maskObject.measuredIdxs[:,1]] = measuredValues

    return reconImage


#Perform the reconstruction with 0-padding; removes stretching in initial measurements
#def performRecon(inputImage, maskObject):
#
#    #Pad input image with a 0-border (known 0 values around the image)
#    inputImage = np.pad(inputImage, [(1, 1), (1, 1)], mode='constant', constant_values=0)
#
#    #Duplicate the mask object, pad a 1-border (scanned idxs) around the mask, account for the padding in internal variables
#    tempMaskObject = copy.deepcopy(maskObject)
#    tempMaskObject.mask = np.pad(tempMaskObject.mask, [(1, 1), (1, 1)], mode='constant', constant_values=1)
#    tempMaskObject.imageHeight+=2
#    tempMaskObject.imageWidth+=2
#    tempMaskObject.measuredIdxs = np.transpose(np.where(tempMaskObject.mask == 1))
#    tempMaskObject.unMeasuredIdxs = np.transpose(np.where(tempMaskObject.mask == 0))
#
#    #Find neighbor information
#    neighborIndices, neighborWeights, neighborDistances = findNeighbors(tempMaskObject)
#
#    #Create a blank image for the reconstruction
#    reconImage = np.zeros((tempMaskObject.imageHeight, tempMaskObject.imageWidth))
#
#    #Retreive measured values
#    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in tempMaskObject.measuredIdxs]))
#    measuredValues = inputImage[np.asarray(idxsX), np.asarray(idxsY)]
#
#    #Compute reconstruction values using IDW (inverse distance weighting)
#    reconImage[tempMaskObject.unMeasuredIdxs[:,0], tempMaskObject.unMeasuredIdxs[:,1]] = np.sum(measuredValues[neighborIndices]*neighborWeights, axis=1)
#
#    #Combine measured values back into the reconstruction image
#    reconImage[tempMaskObject.measuredIdxs[:,0], tempMaskObject.measuredIdxs[:,1]] = measuredValues
#
#    #Remove 0-padding
#    reconImage = reconImage[1:maskObject.imageHeight+1, 1:maskObject.imageWidth+1]
#
#    return reconImage

def percResults(results, perc_testingResults, precision):

    percents = np.linspace(min(np.hstack(perc_testingResults)), max(np.hstack(perc_testingResults)), int((max(np.hstack(perc_testingResults)) - min(np.hstack(perc_testingResults))) / precision + 1))
    newResults = [np.interp(percents, perc_testingResults[resultNum], results[resultNum]) for resultNum in range(0, len(results))]
    averageResults = np.average(newResults, axis=0)
    
    return percents, averageResults

def sectionTitle(title):
    print('\n' + ('#' * int(consoleColumns)))
    print(title)
    print(('#' * int(consoleColumns)) + '\n')

#Convert bytes into human readable format
def sizeFunc(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0: return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def addClip(model, numFilters, numChannels):
    inputs = Input(shape=(None,None,numChannels))
    modelResult = model(inputs, training=False)
    output = tf.clip_by_value(modelResult, 0, 1)

    return tf.keras.Model(inputs=inputs, outputs=output)

#CNN
def cnn(numFilters, numChannels):
    inputs = Input(shape=(None,None,numChannels))
    
    conv = Conv2D(numFilters, (3,3), padding='same', kernel_initializer='he_normal')(inputs)
    conv = LayerNormalization()(conv)
    conv = LeakyReLU()(conv)
    
    conv = Conv2D(numFilters, (3,3), padding='same', kernel_initializer='he_normal')(conv)
    conv = LayerNormalization()(conv)
    conv = LeakyReLU()(conv)
    
    conv = Conv2D(numFilters, (1,1), padding='same', kernel_initializer='he_normal')(conv)
    conv = LayerNormalization()(conv)
    conv = LeakyReLU()(conv)
    
    output = Conv2D(1, (1,1), activation='linear', padding='same', kernel_initializer='he_normal')(conv)
    output = tfp.math.clip_by_value_preserve_gradient(output, 0, 1)

    return tf.keras.Model(inputs=inputs, outputs=output)

def mlp(numFilters, numChannels):
    inputs = Input(shape=(None,None,numChannels))
    dense_1 = Dense(numFilters, activation='relu', kernel_initializer='he_normal')(inputs)
    dense_2 = Dense(numFilters, activation='relu', kernel_initializer='he_normal')(dense_1)
    dense_3 = Dense(numFilters, activation='relu', kernel_initializer='he_normal')(dense_2)
    dense_4 = Dense(numFilters, activation='relu', kernel_initializer='he_normal')(dense_3)
    dense_5 = Dense(numFilters, activation='relu', kernel_initializer='he_normal')(dense_4)
    output = Dense(1, activation='linear', kernel_initializer='he_normal')(dense_5)
    output = tfp.math.clip_by_value_preserve_gradient(output, 0, 1)

    return tf.keras.Model(inputs=inputs, outputs=output)

#U-Net
def unet(numFilters, numChannels):
    
    inputs = Input(shape=(None,None,numChannels))

    conv_1 = Conv2D(numFilters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    down_1 = Conv2D(numFilters, (1,1), strides=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv_1)

    conv_2 = Conv2D(numFilters*2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(down_1)
    down_2 = Conv2D(numFilters*2, (1,1), strides=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv_2)

    conv_3 = Conv2D(numFilters*4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(down_2)
    down_3 = Conv2D(numFilters*4, (1,1), strides=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv_3)
    
    conv_4 = Conv2D(numFilters*8, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(down_3)

    upScale_5 = Conv2D(numFilters*4, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(interpolation='nearest')(conv_4))
    skip_5 = concatenate([upScale_5, conv_3], axis=3)
    conv_5 = Conv2D(numFilters*4, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(skip_5)
    
    upScale_6 = Conv2D(numFilters*2, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(interpolation='nearest')(conv_5))
    skip_6 = concatenate([upScale_6, conv_2], axis=3)
    conv_6 = Conv2D(numFilters*2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(skip_6)
    
    upScale_7 = Conv2D(numFilters, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(interpolation='nearest')(conv_6))
    skip_7 = concatenate([upScale_7, conv_1], axis=3)
    conv_7 = Conv2D(numFilters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(skip_7)
    
    conv_8 = Conv2D(numFilters, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_7)
    output = Conv2D(1, (1,1), activation='linear', padding='same', kernel_initializer='he_normal')(conv_8)
    output = tfp.math.clip_by_value_preserve_gradient(output, 0, 1)
    
    return tf.keras.Model(inputs=inputs, outputs=output)

#Extract features of interest from a reconstruction for propogation through network
def featureExtractor(maskObject, measuredImage, reconImage):

    #Create blank image to build on for remaining visualized features
    featureImage = np.zeros([maskObject.imageHeight, maskObject.imageWidth])

    #Stack measured value image
    featureImage = np.dstack((featureImage, measuredImage))

    #Stack recon values for only unmeasured locations
    tempReconImage = copy.deepcopy(reconImage)
    tempReconImage[maskObject.measuredIdxs[:,0], maskObject.measuredIdxs[:,1]] = 0
    featureImage = np.dstack((featureImage, tempReconImage))
    
    #Delete the blank first channel
    featureImage = np.delete(featureImage, 0, -1)

    return featureImage

#Convert image into TF model compatible shapes/tensors
def makeCompatible(image):
    
    #Save the original image dimensions
    originalShape = image.shape[:2]
    
    #Resize with nearest neighbor for the network architecture
    image = resize(image, (int(np.ceil(image.shape[0]/depthFactor)*depthFactor), int(np.ceil(image.shape[1]/depthFactor)*depthFactor)), order=0)

    #If there is more than one channel, then pad accordingly
    try:
        image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    except:
        image = image.reshape((1,image.shape[0],image.shape[1],1))

    return image, originalShape

def computeDifference(array1, array2):
    return abs(array1-array2)

#Metric for model; compute PSNR between two tensors
def PSNR(imageTrue, imagePred): 
    return tf.reduce_mean(tf.image.psnr(imageTrue, imagePred, max_val=1.0))
