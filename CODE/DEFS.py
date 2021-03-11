#==================================================================
#SLADS DEFINITIONS GENERAL
#==================================================================

#Perform summation and alignment over an mz range for a given line; mzFile cannot be pickled and passed into function
@ray.remote
def mzrange_parhelper(mzRange, scanFileName, mzMethod, ignoreMissingLines, missingLines):
    data = mzFile(scanFileName)
    lineNum = int(scanFileName.split('line')[1].split('.')[0])-1
    
    #If ignoring missing lines, then determine the offset for correct indexing
    if ignoreMissingLines: lineNum -= np.sum([lineNum > missingLine for missingLine in missingLines])
    
    if mzMethod == 'sum':
        lineData = np.asarray([np.sum(mz_range(data.scan(px, 'profile'), mzRange)) for px in range(data.scan_range()[0], data.scan_range()[1]+1)])
    elif mzMethod =='xic':
        lineData = np.asarray(list(map(list, data.xic(data.time_range()[0], data.time_range()[1], mzRange[0], mzRange[1]))))[:,1]
    else:
        sys.exit('Error! - mzMethod: ' + mzMethod + 'for: ' + scanFileName + 'has not been implemented')
 
    return lineNum, lineData

#All information pertaining to a sample
class Sample:
    def __init__(self, sampleFolder, initialPercToScan, scanMethod, ignoreMissingLines=False):
    
        #Should missing lines be ignored (to only be used in training and testing)
        self.ignoreMissingLines = ignoreMissingLines
    
        #Location of MSI data and sample name
        self.sampleFolder = sampleFolder + os.path.sep
        self.name = os.path.basename(sampleFolder)
    
        #Which files have already been read
        self.readScanFiles = []
        self.readLines = []

        #Read in data from sampleInfo.txt
        lineIndex = 0
        sampleInfo = open(self.sampleFolder+'sampleInfo.txt').readlines()
        
        #Read the max number of lines that are expected 
        numLines = int(sampleInfo[lineIndex].rstrip())
        lineIndex += 1
        
        #Read the sample width
        self.sampleWidth = float(sampleInfo[lineIndex].rstrip())
        lineIndex += 1
        
        #Read the sample height
        self.sampleHeight = float(sampleInfo[lineIndex].rstrip())
        lineIndex += 1
        
        #Read in mapping method
        self.scanRate = float(sampleInfo[lineIndex].rstrip())
        lineIndex += 1
        
        #Read how the mz ranges should be visualized; sum or xic
        self.mzMethod = sampleInfo[lineIndex].rstrip()
        lineIndex += 1
        
        #Read how m/z ranges should be specified; range, or value
        self.mzSpec = sampleInfo[lineIndex].rstrip()
        lineIndex += 1
        
        #Read the normalization method; tic, standard, or none
        self.normMethod = sampleInfo[lineIndex].rstrip()
        lineIndex += 1
        
        #If specifying m/z ranges by value, or using a standard normalization, read what the tolerance window should be
        if self.normMethod == 'standard' or self.mzSpec == 'value': self.mzTolerance = (float(sampleInfo[lineIndex].rstrip())*1e-03)/2
        
        #Read in all m/z locations/ranges (.csv)
        if self.mzSpec == 'value':
            mzLocations = np.loadtxt(self.sampleFolder+'mz.csv', delimiter=',')
            self.mzRanges = [[mzLocation-self.mzTolerance, mzLocation+self.mzTolerance] for mzLocation in mzLocations]
        elif self.mzSpec == 'range':
            self.mzRanges = np.loadtxt(self.sampleFolder+'mz.csv', delimiter=',')
        
        #Prepare variables to hold sample data
        self.mzWeights = np.ones(len(self.mzRanges))/len(self.mzRanges)

        #If using a standard, then also read in standards locations (.csv)
        if self.normMethod == 'standard': 
            mzStandardLocations = np.loadtxt(self.sampleFolder+'mzStandards.csv', delimiter=',')
            if mzStandardLocations.shape != ():
                print('Warning! - Untested functionality, will sum the formed visualizations of the specified m/z ranges for the normalization')
                self.mzStandardRanges = [[mzStandardLocation-self.mzTolerance, mzStandardLocation+self.mzTolerance] for mzStandardLocation in mzStandardLocations]
            else:
                self.mzStandardRanges = [[mzStandardLocations-self.mzTolerance, mzStandardLocations+self.mzTolerance]]
        
        #Declare variables used in training/testing/implementation; NOT SURE IF THESE ARE STILL NEEDED
        self.origNormArray = None
        self.normArray = None
        self.avgMeasuredImage = None
        self.avgGroundTruthImage = []
        self.resultsPath = ''
        self.measuredLines = []
        self.mzReconImages = None
        self.RDImage = None
        self.RDValues = None
        self.polyFeatures = None
        self.percMeasured = 0
        
        #Get the MSI file extension automatically      
        extensions = list(map(lambda x:x.lower(), np.unique([filename.split('.')[-1] for filename in natsort.natsorted(glob.glob(self.sampleFolder+'*'), reverse=False)])))
        if 'd' in extensions: self.lineExt = '.d'
        elif 'raw' in extensions: self.lineExt = '.raw'
        else: sys.exit('Error! - Unknown MSI filetype being used for sample: ' + self.name)
        
        #Store final dimensions for physical domain, determining the number of columns for row-alignment interpolations
        self.finalDim = [numLines, int(round((self.sampleWidth*1e3)/self.scanRate))]
        
        #If missing lines are to be ignored, then setup to do so (only possible to perform in simulated operation)
        if self.ignoreMissingLines: 
            scanFiles = natsort.natsorted(glob.glob(self.sampleFolder+'*'+self.lineExt), reverse=False)
            self.missingLines = list(set(np.arange(1, self.finalDim[0]).tolist()) - set([int(scanFile.split('line')[1].split('.')[0]) for scanFile in scanFiles]))
            self.missingLines = [missingLine-1 for missingLine in self.missingLines]
            self.finalDim[0] -= len(self.missingLines)
        else: 
            self.missingLines = []
        
        #Setup empty arrays for actual measurements; used in simulations since mzImages contains complete data
        self.measuredmzImages = np.asarray([np.zeros((self.finalDim[0], self.finalDim[1])) for mzRangeNum in range(0, len(self.mzRanges))])

        #Establish the total sample area; for determination of percMeasured
        self.area = int(round(self.finalDim[0]*self.finalDim[1]))

        #Setup objects for storing raw MSI data
        self.newTimes = np.linspace(0, (self.finalDim[1]-1)/60, self.finalDim[1])
        self.mzImages = [np.zeros((self.finalDim[0], self.finalDim[1])) for mzRangeNum in range(0, len(self.mzRanges))]
        self.origTIC = [[] for line in range(0, self.finalDim[0])]
        self.origTimes = copy.deepcopy(self.origTIC) 
        if self.normMethod == 'standard': self.mzStandardImages = [copy.deepcopy(self.origTIC) for mzRangeNum in range(0, len(self.mzStandardRanges))]
        
        #Determine image dimensions that will produce square pixels (consistent vertical/horizontal resolution)
        if(self.finalDim[1]/self.sampleWidth) > (self.finalDim[0]/self.sampleHeight): 
            self.squareDim = [int(round((self.finalDim[1]*self.sampleHeight)/self.sampleWidth)), self.finalDim[1]]
        elif (self.finalDim[1]/self.sampleWidth) < (self.finalDim[0]/self.sampleHeight):
            self.squareDim = [self.finalDim[0], int(round((self.finalDim[0]*self.sampleWidth)/self.sampleHeight))]
        else:
            self.squareDim = self.finalDim
    
        #Initial mask variables for first scan and reseting
        self.initialMask = np.zeros(self.finalDim)
        
        #List of what points/lines should be initially measured
        self.initialSets = []
        
        #If scanning with line-bounded constraint
        if scanMethod == 'linewise':
        
            #Create list of arrays containing points to measure on each line
            self.linesToScan = []
            for rowNum in np.arange(0, self.finalDim[0], 1): self.linesToScan.append([tuple([rowNum, columnNum]) for columnNum in np.arange(0, self.finalDim[1], 1)])
        
            #Fix internal formatting
            self.linesToScan = np.asarray(self.linesToScan).tolist()
            
            #Set initial lines to scan
            lineIndexes = [int((self.finalDim[1]-1)*0.50)]
            
            #Obtain points in the specified lines and add them to the initial scan list
            for lineIndex in lineIndexes:
                
                #If only a percentage should be scanned, then randomly select points, otherwise select all
                if lineMethod == 'percLine':
                    newIdxs = [pt for pt in self.linesToScan[lineIndex]]
                    np.random.suffle(newIdxs)
                    newIdxs = newIdxs[:int((stopPerc/100)*self.finalDim[1])]
                else: 
                    newIdxs = [pt for pt in self.linesToScan[lineIndex]]
                    
                #Add the points to the initial mask
                for pt in [tuple(pt) for pt in newIdxs]: self.initialMask[pt] = 1
                
                #Delete the lines/points specified from remaining potentials; in case of overlapping 'line' geometries
                for lineIndexNum in range(0, len(lineIndexes)): self.delLine(lineIndexes[lineIndex]-lineIndexNum)
                
                #Add positions to initial list
                self.initialSets.append(newIdxs)
                
        elif scanMethod == 'pointwise':
        
            #Randomly select points to initially scan
            self.unMeasuredIdxs = np.transpose(np.where(self.initialMask==0))
            newIdxs = np.asarray(random.sample(self.unMeasuredIdxs.tolist(), int((initialPercToScan/100)*self.area)))
            
            #Add the points to the initial mask
            for pt in [tuple(pt) for pt in newIdxs]: self.initialMask[pt] = 1
            
            #Add positions to initial list
            self.initialSets.append(newIdxs)
            
        #Store the initially measured and unmeasured locations and mask for the first iteration
        self.mask = copy.deepcopy(self.initialMask)
        self.measuredIdxs = np.transpose(np.where(self.initialMask==1))
        self.unMeasuredIdxs = np.transpose(np.where(self.initialMask==0))
        
        #Now reshape and extract measured and unmeasured locations for the square pixel mask
        self.squareMask = resize(self.initialMask, (self.squareDim[0], self.squareDim[1]), order=0)
        self.squareMeasuredIdxs = np.transpose(np.where(self.squareMask==1))
        self.squareUnMeasuredIdxs = np.transpose(np.where(self.squareMask==0))
        
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
            
            #If ignoring missing lines, then determine the offset for correct indexing
            if self.ignoreMissingLines: lineNum -= np.sum([lineNum > missingLine for missingLine in self.missingLines])
            
            #Obtain the total ion chromatogram over all mz available
            xicData = np.asarray(data.xic(data.time_range()[0], data.time_range()[1]))
            self.origTimes[lineNum] = xicData[:,0]
            self.origTIC[lineNum] = xicData[:,1]
        
        #Check and setup for normalization
        if self.normMethod == 'tic':
            self.origNormArray = self.origTIC
        elif self.normMethod == 'standard':
            for mzRangeNum in range(0, len(self.mzStandardRanges)): 
                results = ray.get([mzrange_parhelper.remote(self.mzStandardRanges[mzRangeNum], scanFileName, self.mzMethod, self.ignoreMissingLines, self.missingLines) for scanFileName in scanFiles])
                for result in results: self.mzStandardImages[mzRangeNum][result[0]] = result[1]
                self.origNormArray = np.sum(self.mzStandardImages, axis=0)
        elif self.normMethod == 'none':
            self.origNormArray = None
        else:
            sys.exit('Error! - Unknown normalization method: ' + self.normMethod + ' specified for sample: ' + self.name)

        #Align the normalization array for visualization
        self.normArray = np.asarray([np.interp(self.newTimes, self.origTimes[lineNum], self.origNormArray[lineNum]) for lineNum in range(0, self.finalDim[0])])

        #For each mzRange generate a visualization, performing normalization as specified
        for mzRangeNum in range(0, len(self.mzRanges)): 
            results = ray.get([mzrange_parhelper.remote(self.mzRanges[mzRangeNum], scanFileName, self.mzMethod, self.ignoreMissingLines, self.missingLines) for scanFileName in scanFiles])
            for result in results: 
                if self.normMethod != 'none': 
                    self.mzImages[mzRangeNum][result[0]] = np.interp(self.newTimes, self.origTimes[result[0]], 
                    np.nan_to_num(result[1]/self.origNormArray[result[0]], nan=0, posinf=0, neginf=0))
                else:
                    self.mzImages[mzRangeNum][result[0]] = np.interp(self.newTimes, self.origTimes[result[0]], result[1])
        
        #Determine ground-truth average images
        self.avgGroundTruthImage = np.average(np.asarray(self.mzImages), axis=0, weights=self.mzWeights)
        self.avgGroundTruthImage = MinMaxScaler().fit_transform(self.avgGroundTruthImage.reshape(-1, 1)).reshape(self.avgGroundTruthImage.shape)
        self.avgSquareGroundTruthImage = resize(self.avgGroundTruthImage, (self.squareDim[0], self.squareDim[1]), order=0)
        
        #Create corresponding square variable to mzImages
        self.squaremzImages = np.asarray([resize(mzImage, (self.squareDim[0], self.squareDim[1]), order=0) for mzImage in self.mzImages])

    #Scan new locations in the sample
    def performMeasurements(self, newIdxs, simulationFlag, fromRecon):

        #Update the masks according to the newIdxs; need to do so in squareMask for nearest neighbor search
        if len(newIdxs) != 0: 
            for pt in newIdxs: self.mask[tuple(pt)] = 1
            self.measuredIdxs = np.transpose(np.where(self.mask==1))
            self.unMeasuredIdxs = np.transpose(np.where(self.mask==0))
            
            self.squareMask = resize(self.mask, (self.squareDim[0], self.squareDim[1]), order=0)
            self.squareMeasuredIdxs = np.transpose(np.where(self.squareMask==1))
            self.squareUnMeasuredIdxs = np.transpose(np.where(self.squareMask==0))
        
        #If the measurements are not to be taken from the reconstruction image(s) (as is done for groupwise percToScan acquisition)
        if not fromRecon:
            #If this is not a simulation then inform equipment, wait/read/update; otherwise used stored information
            if not simulationFlag:
                with open('./INPUT/IMP/UNLOCK', 'w') as filehandle: filehandle.writelines(str(tuple(newIdxs[0])) + ', ' + str(tuple(newIdxs[len(newIdxs)-1])))
                equipWait()
                self.readScanData()
                self.measuredmzImages = copy.deepcopy(self.mzImages)
            else:
                for imageNum in range(0,len(self.measuredmzImages)):
                    temp = copy.deepcopy(np.asarray(self.mzImages[imageNum]))
                    temp[self.mask == 0] = 0
                    self.measuredmzImages[imageNum] = temp.copy()

            #Update the square mz images
            self.squareMeasuredmzImages = np.asarray([resize(measuredmzImage, (self.squareDim[0], self.squareDim[1]), order=0) for measuredmzImage in self.measuredmzImages])
            
            #Perform averaging of the multiple channels and subsequent image normalization
            self.avgMeasuredImage = np.average(np.asarray(self.measuredmzImages), axis=0, weights=self.mzWeights)
            self.avgMeasuredImage = MinMaxScaler().fit_transform(self.avgMeasuredImage.reshape(-1, 1)).reshape(self.avgMeasuredImage.shape)

        else: #When fromRecon, only a single value is presented at a time in a list
            if averageReconInput or erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
                self.avgMeasuredImage[newIdxs[0][0], newIdxs[0][1]] = self.avgReconImage[newIdxs[0][0], newIdxs[0][1]]
                
            elif percToScan != None and erdModel == 'DLADS':
                self.measuredmzImages[:, newIdxs[0][0], newIdxs[0][1]] = self.mzReconImages[:, newIdxs[0][0], newIdxs[0][1]]
                
                #Update the square mz images
                self.squareMeasuredmzImages = np.asarray([resize(measuredmzImage, (self.squareDim[0], self.squareDim[1]), order=0) for measuredmzImage in self.measuredmzImages])
                
                #Perform averaging of the multiple channels and subsequent image normalization
                self.avgMeasuredImage = np.average(np.asarray(self.measuredmzImages), axis=0, weights=self.mzWeights)
                self.avgMeasuredImage = MinMaxScaler().fit_transform(self.avgMeasuredImage.reshape(-1, 1)).reshape(self.avgMeasuredImage.shape)

        self.avgSquareMeasuredImage = resize(self.avgMeasuredImage, (self.squareDim[0], self.squareDim[1]), order=0)
        
        #Update percentage pixels measured
        self.percMeasured = (np.sum(self.mask)/self.area)*100

    #Reset the smaple mask and linesToScan
    def maskReset(self, simulationFlag):
        
        #If this is a simulation, then set to a blank mask (no initial scan performed), otherwise set to state after initial measurements
        if simulationFlag: self.mask = np.zeros(self.finalDim)
        else: self.mask = copy.deepcopy(self.initialMask)
            
        #Set the measured/unmeasured location lists so it matches the intended state
        self.measuredIdxs = np.transpose(np.where(self.mask==1))
        self.unMeasuredIdxs = np.transpose(np.where(self.mask==0))
            
        #Now reshape and extract measured and unmeasured locations for the square pixel mask
        self.squareMask = resize(self.initialMask, (self.squareDim[0], self.squareDim[1]), order=0)
        self.squareMeasuredIdxs = np.transpose(np.where(self.squareMask==1))
        self.squareUnMeasuredIdxs = np.transpose(np.where(self.squareMask==0))
    
    #Remove entire lines from future consideration
    def delLine(self, index):
        if scanMethod == 'linewise': self.linesToScan = np.delete(self.linesToScan, index, 0).tolist()
        else: sys.exit('Error! - Attempted to delete a line with non-linewise scanning method')
        
    def delPoints(self, pts):
        if scanMethod == 'linewise':
            for lineNum in range(0, len(self.linesToScan)): self.linesToScan[lineNum] = [pt for ix, pt in enumerate(self.linesToScan[lineNum]) if pt not in pts]
            self.linesToScan = [x for x in self.linesToScan if x]
        else:
            sys.exit('Error! - Attempted to delete points from lines with non-linewise scanning method')
        
#Singular result generated through runSLADS
class Result():
    def __init__(self, sample, avgGroundTruthImage, bestCFlag, oracleFlag, simulationFlag, animationFlag):
        self.sample = copy.deepcopy(sample)
        self.avgGroundTruthImage = copy.deepcopy(avgGroundTruthImage)
        self.simulationFlag = copy.deepcopy(simulationFlag)
        self.animationFlag = copy.deepcopy(animationFlag)
        self.bestCFlag = copy.deepcopy(bestCFlag)
        self.oracleFlag = copy.deepcopy(oracleFlag)
        self.avgImages = []
        self.reconImages = []
        self.RDImages = []
        self.mzReconImages = []
        self.samples = []
        self.ERDValueNPs = []
        
        self.MSEList = []
        self.SSIMList = []
        self.PSNRList = []
        self.TDList = []
        self.ERDPSNRList = []   
        
        self.percMeasuredList = []

    def update(self, sample, ERDValuesNP):

        #Save the model development
        self.ERDValueNPs.append(copy.deepcopy(ERDValuesNP))
        self.avgImages.append(copy.deepcopy(sample.avgMeasuredImage))
        self.samples.append(copy.deepcopy(sample))
        self.percMeasuredList.append(copy.deepcopy(sample.percMeasured))
    
    def complete(self, optimalC): 
    
        #If average reconstructions were not computed during runSLADS operations, then compute them now
        if not averageReconInput and erdModel == 'DLADS':
            results = ray.get([computeRecon_parhelper.remote(sample.avgGroundTruthImage, sample) for sample in self.samples])
            for resultNum in range(0, len(results)): self.samples[resultNum].avgReconImage = resize(results[resultNum], tuple(sample.finalDim), order=0)
        
        if self.simulationFlag:

            #Perform statistics extraction for all images
            for index in range(0, len(self.samples)):
                
                #Measure and save statistics
                difference = np.sum(computeDifference(self.avgGroundTruthImage, self.samples[index].avgReconImage))
                TD = difference/self.samples[index].area
                MSE = mean_squared_error(self.avgGroundTruthImage, self.samples[index].avgReconImage)
                SSIM = structural_similarity(self.avgGroundTruthImage, self.samples[index].avgReconImage, data_range=1)
                PSNR = compare_psnr(self.avgGroundTruthImage, self.samples[index].avgReconImage, data_range=1)

                self.TDList.append(TD)
                self.MSEList.append(MSE)
                self.SSIMList.append(SSIM)
                self.PSNRList.append(PSNR)
                
            #If determining the best c, return the area under the PSNR curve
            if self.bestCFlag: return self.PSNRList, self.percMeasuredList

            #Calculate the actual RD Image; bestCFlag data should be returned before this subroutine
            for index in tqdm(range(0, len(self.samples)), desc='RD Calc', leave = False, ascii=True):
                RDImage = computeRD(self.samples[index], optimalC)
                self.RDImages.append(RDImage)
                self.ERDPSNRList.append(compare_psnr(RDImage, self.ERDValueNPs[index], data_range=1))
            

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
                if averageReconInput or erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
                    results = ray.get([computeRecon_parhelper.remote(self.samples[-1].squareMeasuredmzImages[mzImageNum], self.samples[-1]) for mzImageNum in range(0,len(self.samples[-1].squareMeasuredmzImages))])
                    self.mzReconImages = np.asarray([resize(result, tuple(self.samples[-1].finalDim), order=0) for result in results])
                
                for massNum in tqdm(range(0, len(self.samples[-1].mzRanges)), desc='mz Images', leave = False, ascii=True):
                    
                    #mz image with only the actual measurements made
                    subMeasuredImage = self.samples[-1].measuredmzImages[massNum]
                    
                    #Retrieve reconstruction for the specific mz image
                    subReconImage = self.mzReconImages[massNum]

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
                    sub.imshow(self.samples[-1].mask, cmap='gray', aspect='auto')
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
                    im = ax2.imshow(self.samples[i].avgReconImage, cmap='hot', aspect='auto', vmin=0, vmax=1)
                    ax2.set_title('Reconstruction')
                    cbar = f.colorbar(im, ax=ax2, orientation='vertical', pad=0.01)

                    ax3 = plt.subplot2grid((2,3), (0,2))
                    im = ax3.imshow(abs(self.avgGroundTruthImage-self.samples[i].avgReconImage), cmap='hot', aspect='auto', vmin=0, vmax=1)
                    ax3.set_title('Absolute Difference')
                    cbar = f.colorbar(im, ax=ax3, orientation='vertical', pad=0.01)

                    ax4 = plt.subplot2grid((2,3), (1,0))
                    im = ax4.imshow(self.samples[i].mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
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
                    plt.imshow(self.samples[i].avgReconImage, cmap='hot', aspect='auto', vmin=0, vmax=1)
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
                    plt.imshow(self.samples[i].mask, cmap='gray', aspect='auto')
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
                plt.imshow(self.samples[-1].avgReconImage, cmap='hot', aspect='auto', vmin=0, vmax=1)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()

                #Save the final mask, no borders
                saveLocation = dir_AnimationFrames + 'final_mask_' + self.samples[-1].name + '_iter_' + str(i+1) + '_perc_' + str(self.percMeasuredList[i]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.samples[-1].mask, cmap='gray', aspect='auto')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()
                
            else: #Not a simulation
                print('Warning! - Non simulation plots not yet fixed for color issue and modified selection')

def iou(groundTruth, prediction):
    return np.sum(np.logical_and(groundTruth, prediction)) / np.sum(np.logical_or(groundTruth, prediction))

@ray.remote
def computeRecon_parhelper(image, sample):
    return computeRecon(image, sample)

#Generate/apply a gaussian kernel to a window of an image centered at an idx, summing result; window size according to sigma strength
def gaussianGenerator(inputImage, idx, sigma):
    
    #Determine odd number >= 3 times the given sigma value for a reasonable window size to generate, (width, height)
    windowSize = int(np.ceil((np.ceil(sigma*3)//2)*2+1))
    
    #Pad input image based on window size, to ensure no data is lost when splitting into windows
    paddedInputImage = np.pad(inputImage, [(int(np.floor(windowSize/2)), ), (int(np.floor(windowSize/2)), )], mode='constant')
    
    #Extract window around specified idx and calculate kernel
    window = viewW(paddedInputImage, (windowSize, windowSize))[idx[0], idx[1]]
    kernel = np.outer(signal.gaussian(windowSize, std=sigma), signal.gaussian(windowSize, std=sigma))
    
    return np.sum(window*kernel)

#Perform gaussianGenerator for a set of sigma values
@ray.remote
def gaussian_parhelper(idxs, inputImage, sigmaValues, indexes):
    return [gaussianGenerator(inputImage, idxs[index], sigmaValues[index]) for index in indexes]

def computeRD(sample, cValue):
    
    if RDMethod == 'var':
        #Compute the difference between the ground-truth mz images and their reconstructions
        difference = abs(sample.squaremzImages-sample.squaremzReconImages)
        #Flatten difference stack by finding the variance at each pixel 
        RDPP_id = ray.put(np.var(difference, axis=0))
    elif RDMethod == 'max':
        #Compute the difference between the ground-truth mz images and their reconstructions
        difference = abs(sample.squaremzImages-sample.squaremzReconImages)
        #Flatten difference stack by finding the maximum value at each pixel
        RDPP_id = ray.put(np.max(difference, axis=0))
    else:
        RDPP_id = ray.put(computeDifference(sample.avgSquareGroundTruthImage, sample.avgSquareReconImage))
        
    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(sample.squareMeasuredIdxs, sample.squareUnMeasuredIdxs)
    
    #Calculate the sigma value for chosen c value, store directly into shared memory
    sigmaValues_id = ray.put(neighborDistances[:,0]/cValue)
    
    #Split computation sets to be run on multiple processes
    indexes = np.asarray(list(range(0, len(sample.squareUnMeasuredIdxs))))
    blockSize = int(np.ceil(len(indexes) / float(multiprocessing.cpu_count())))
    indexSets = np.split(indexes, np.arange(blockSize, len(indexes), blockSize))
    
    #Store indexes into shared memory
    unMeasuredIdxs_id = ray.put(sample.squareUnMeasuredIdxs)
    
    #Perform computation of RD values for each unmeasured point
    results = ray.get([gaussian_parhelper.remote(unMeasuredIdxs_id, RDPP_id, sigmaValues_id, indexes) for indexes in indexSets])
    RDValues = [result for resultSet in results for result in resultSet]
    
    #Reassemble the values into a single image
    RDImage = np.zeros((sample.squareMask.shape))
    RDImage[np.where(sample.squareMask==0)] = RDValues
    
    #Normalize
    RDImage = MinMaxScaler().fit_transform(RDImage.reshape(-1, 1)).reshape(RDImage.shape)

    #Resize to physical domain dimensionality
    RDImage = resize(RDImage, tuple(sample.finalDim), order=0)

    return RDImage

def computePolyFeatures(sample, reconImage):
    
    #Retreive recon values
    inputValues = reconImage[sample.squareUnMeasuredIdxs[:,0], sample.squareUnMeasuredIdxs[:,1]]
    
    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in sample.squareMeasuredIdxs]))
    measuredValues = reconImage[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(sample.squareMeasuredIdxs, sample.squareUnMeasuredIdxs)
    neighborValues = measuredValues[neighborIndices]
    
    #Create array to hold features
    feature = np.zeros((np.shape(sample.squareUnMeasuredIdxs)[0],6))
    
    #Compute std div features
    diffVect = computeDifference(neighborValues, np.transpose(np.matlib.repmat(inputValues, np.shape(neighborValues)[1],1)))
    feature[:,0] = np.sum(neighborWeights*diffVect, axis=1)
    feature[:,1] = np.sqrt(np.sum(np.power(diffVect,2),axis=1))
    
    #Compute distance/density features
    cutoffDist = np.ceil(np.sqrt((featDistCutoff/100)*(sample.area/np.pi)))
    feature[:,2] = neighborDistances[:,0]
    neighborsInCircle = np.sum(neighborDistances<=cutoffDist,axis=1)
    feature[:,3] = (1+(np.pi*(np.square(cutoffDist))))/(1+neighborsInCircle)
    
    #Compute gradient features; assume continuous features
    gradientImageX, gradientImageY = np.gradient(reconImage)
    feature[:,4] = abs(gradientImageY)[sample.squareUnMeasuredIdxs[:,0], sample.squareUnMeasuredIdxs[:,1]]
    feature[:,5] = abs(gradientImageX)[sample.squareUnMeasuredIdxs[:,0], sample.squareUnMeasuredIdxs[:,1]]
    
    #Fit polynomial features to the determined array
    polyFeatures = PolynomialFeatures(degree=2).fit_transform(feature)
    
    return polyFeatures

def computeERD(sample, model):
    
    if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net':
        
        #Compute ERD values
        polyFeatures = computePolyFeatures(sample, sample.avgSquareReconImage)
        ERDValues = model.predict(polyFeatures)
        
        #Rearrange ERD values into array; those that have already been measured have 0 ERD
        ERD = np.zeros(sample.squareDim)
        ERD[sample.squareUnMeasuredIdxs[:, 0], sample.squareUnMeasuredIdxs[:, 1]] = ERDValues
        
        #Remove values that are less than those already scanned (0 ERD)
        ERD[np.where((ERD < 0))] = 0
        
        #Resize to physical domain dimensionality
        ERD = resize(ERD, tuple(sample.finalDim), order=0)
    
    elif erdModel == 'DLADS':
        
        #Send input through trained model
        if averageReconInput: 
            measuredImage = copy.deepcopy(sample.avgSquareMeasuredImage)
            #measuredImage[sample.squareMask==0] = 0
            inputImage, originalShape = makeCompatible(featureExtractor(sample, measuredImage, sample.avgSquareReconImage))
        else:
            inputImage, originalShape = makeCompatible(np.stack(sample.squareMeasuredmzImages, axis=-1))

        #Compute ERD and resize to finalDim
        ERD = resize(model.predict(inputImage, steps=1)[0,:,:,0], tuple(sample.finalDim), order=0)

    return ERD
    
def runSLADS(samples, model, scanMethod, cValue, percToScan, stopPerc, sampleNum, simulationFlag, trainPlotFlag, animationFlag, tqdmHide, oracleFlag, bestCFlag):

    if simulationFlag: #Here sample.mzImages contains the full ground-truth
        sample = samples[sampleNum]        
    else: #Here sample.mzImages contains only the initially measured images
        sample = samples
        sample.measuredmzImages = sample.mzImages
        sample.squareMeasuredmzImages = sample.squaremzImages
        
    #Reinitialize the mask to starting state
    sample.maskReset(simulationFlag)

    #Indicate the stopping condition has not yet been met
    completedRunFlag = False
    
    #Perform the initial measurements
    if simulationFlag: sample.performMeasurements(np.transpose(np.where(sample.initialMask == 1)), simulationFlag, False)

    #Calculate the reconstruction(s)
    if averageReconInput or erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
        sample.avgSquareReconImage = computeRecon(sample.avgSquareMeasuredImage, sample)
        sample.avgReconImage = resize(sample.avgSquareReconImage, tuple(sample.finalDim), order=0)
    elif percToScan != None and erdModel == 'DLADS':
        sample.squaremzReconImages = np.asarray(ray.get([computeRecon_parhelper.remote(squareMeasuredmzImage, sample) for squareMeasuredmzImage in sample.squareMeasuredmzImages]))
        sample.mzReconImages = [resize(squaremzReconImage, tuple(sample.finalDim), order=0) for squaremzReconImage in sample.squaremzReconImages]

    #Determine ERD or use the RD as the ERD if the oracleFlag or bestCFlag is enabled
    if oracleFlag or bestCFlag:
        ERDValuesNP = computeRD(sample, cValue)
    else:
        ERDValuesNP = computeERD(sample, model)

    #Initialize and perform first update for a result object
    result = Result(sample, sample.avgGroundTruthImage, bestCFlag, oracleFlag, simulationFlag, animationFlag)
    result.update(sample, ERDValuesNP)

    #Check stopping criteria, just in case of a bad input
    if (scanMethod == 'pointwise' or not lineVisitAll) and (round(sample.percMeasured) >= stopPerc): completedRunFlag = True
    if scanMethod == 'linewise' and len(sample.linesToScan) == 0: completedRunFlag = True

    #Until the stopping criteria has been met
    with tqdm(total = float(stopPerc), desc = '% Sampled', leave = False, ascii=True, disable=tqdmHide) as pbar:

        #Initialize progress bar state according to % measured
        pbar.n = round(sample.percMeasured,2)
        pbar.refresh()

        #Until the program has completed
        while not completedRunFlag:
            
            #Find next measurement locations
            newIdxs = findNewMeasurementIdxs(sample, result, model, ERDValuesNP, scanMethod, cValue, simulationFlag, oracleFlag, bestCFlag, percToScan)
            
            #Perform measurements
            sample.performMeasurements(newIdxs, simulationFlag, False)
            
            #Calculate the reconstruction(s)
            if averageReconInput or erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
                sample.avgSquareReconImage = computeRecon(sample.avgSquareMeasuredImage, sample)
                sample.avgReconImage = resize(sample.avgSquareReconImage, tuple(sample.finalDim), order=0)
            elif percToScan != None and erdModel == 'DLADS':
                sample.squaremzReconImages = np.asarray(ray.get([computeRecon_parhelper.remote(squareMeasuredmzImage, sample) for squareMeasuredmzImage in sample.squareMeasuredmzImages]))
                sample.mzReconImages = [resize(squaremzReconImage, tuple(sample.finalDim), order=0) for squaremzReconImage in sample.squaremzReconImages]
                    
            #Determine ERD or use the RD as the ERD if the oracleFlag or bestCFlag is enabled
            if oracleFlag or bestCFlag: ERDValuesNP = computeRD(sample, cValue)
            else: ERDValuesNP = computeERD(sample, model)
            
            #Check stopping conditions
            if (scanMethod == 'pointwise' or not lineVisitAll) and (round(sample.percMeasured) >= stopPerc): completedRunFlag = True
            if scanMethod == 'linewise' and len(sample.linesToScan) == 0: completedRunFlag = True

            #If viz limit, only update when percToViz has been met; otherwise update every iteration
            if ((percToViz != None) and ((sample.percMeasured - result.samples[-1].percMeasured) >= percToViz)) or (percToViz == None):
                result.update(sample, ERDValuesNP)
                performUpdate = False

            #Update the progress bar
            pbar.n = round(sample.percMeasured,2)
            pbar.refresh()

    return result

def findNewMeasurementIdxs(sample, result, model, ERDValuesNP, scanMethod, cValue, simulationFlag, oracleFlag, bestCFlag, percToScan):

    #Make sure ERDValuesNP is in np array
    ERDValuesNP = np.asarray(ERDValuesNP)
    
    if scanMethod == 'random':
        newIdxs = np.asarray(random.sample(sample.unMeasuredIdxs.tolist(), int((percToScan/100)*sample.area)))
    elif scanMethod == 'pointwise':
        
        #If performing a groupwise scan, use reconstruction as the "performed" measurement, until reaching target
        if percToScan != None:
        
            #Create a list to hold the chosen scanning locations
            newIdxs = []
            
            #Until the percToScan has been reached, substitute reconstruction values for actual measurements
            while True:

                #Find next measurement location and store the chosen scanning location for later, actual measurement
                newIdxs.append(sample.unMeasuredIdxs[np.argmax(ERDValuesNP[sample.mask==0])])
                
                #Perform the measurement, using values from reconstruction 
                sample.performMeasurements([newIdxs[-1]], simulationFlag, True)
                
                #When enough new locations have been determined, break from loop
                if (sample.percMeasured-result.samples[-1].percMeasured) >= percToScan: break
                
                #Re-compute ERD/RD; ensure in an array
                if oracleFlag or bestCFlag: ERDValuesNP = np.asarray(computeRD(sample, cValue))
                else: ERDValuesNP = np.asarray(computeERD(sample, model))
        else:
            #Identify the unmeasured location with the highest ERD value; return in a list to ensure it is iterable
            newIdxs = [sample.unMeasuredIdxs[np.argmax(ERDValuesNP[sample.mask==0])]]

    elif scanMethod == 'linewise':
        #==========================================
        #OPTIMAL LINE DETERMINATION
        #==========================================

        #Choose the line with maximum ERD and extract the actual indices
        lineERDSums = [np.nansum(ERDValuesNP[tuple([x[0] for x in line]), tuple([y[1] for y in line])]) for line in sample.linesToScan]
        lineToScanIdx = np.nanargmax(lineERDSums)
        lineToScanIdxs = sample.linesToScan[lineToScanIdx]
        
        #Obtain the ERD values in the chosen line
        lineERDValues = [ERDValuesNP[tuple(pt)] for pt in lineToScanIdxs]
        
        #==========================================
        #PARTIAL LINE BY PERCENT
        #==========================================
        #Scan stopPerc locations on the line with maximized ERD
        if lineMethod == 'percLine':
        
            #Create a list to hold the chosen scanning locations
            newIdxs = []
            
            #Until the stopPerc has been reached, substitute reconstruction values for actual measurements
            while True:
            
                #Identify the next scanning location and store it for later, actual measurement
                nextIndex = np.argmax(lineERDValues)
                newIdxs.append(lineToScanIdxs[nextIndex])
                
                #Remove that location from further consideration in this loop
                lineToScanIdxs = np.delete(lineToScanIdxs, nextIndex)
                
                #Perform the measurement, using values from reconstruction 
                sample.performMeasurements([newIdxs[-1]], simulationFlag, True)
                
                #When enough new locations have been determined, break from loop
                if (sample.percMeasured-result.samples[-1].percMeasured) >= stopPerc: break
                
                #Re-compute ERD/RD; ensure in an array
                if oracleFlag or bestCFlag: ERDValuesNP = np.asarray(computeRD(sample, cValue))
                else: ERDValuesNP = np.asarray(computeERD(sample, model))
                
                #Obtain the ERD values in the chosen line
                lineERDValues = [ERDValuesNP[tuple(pt)] for pt in lineToScanIdxs]

        #==========================================
        
        #==========================================
        #PARTIAL LINE BY START/END POINTS
        #==========================================
        #Choose segment to scan on line which contains at least stopPerc locations with maximal ERD
        elif lineMethod == 'startEndPoints':
            newIdxs = lineToScanIdxs.copy()
            newIdxs = np.asarray(newIdxs)[np.argsort(lineERDValues)][::-1][:int((stopPerc/100)*len(lineERDValues))]
            orderedNewIdxs = newIdxs[np.argsort(newIdxs[:,0]*newIdxs[:,1])]
            startLocation, endLocation = orderedNewIdxs[0], orderedNewIdxs[len(orderedNewIdxs)-1]
            newIdxs = np.asarray(lineToScanIdxs[lineToScanIdxs.index(startLocation.tolist()):lineToScanIdxs.index(endLocation.tolist())])
        else:
            sys.error('Error! - Unknown line method specified in configuation: ' + lineMethod)
            
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
            sample.delPoints(newIdxs)
        else:
            #Remove the line selected from further consideration, does not allow revisiting
            sample.delLine(lineToScanIdx)
        #==========================================

    return newIdxs

def findNeighbors(measuredIdxs, unMeasuredIdxs):

    #Calculate knn
    neigh = NearestNeighbors(n_neighbors=numNeighbors)
    neigh.fit(measuredIdxs)
    neighborDistances, neighborIndices = neigh.kneighbors(unMeasuredIdxs)

    #Determine inverse distance weights
    unNormNeighborWeights = 1.0/np.square(neighborDistances)
    neighborWeights = unNormNeighborWeights/(np.sum(unNormNeighborWeights, axis=1))[:, np.newaxis]

    return neighborIndices, neighborWeights, neighborDistances

#Perform the reconstruction without 0-padding
def computeRecon(inputImage, sample):

    #Find neighbor information
    neighborIndices, neighborWeights, neighborDistances = findNeighbors(sample.squareMeasuredIdxs, sample.squareUnMeasuredIdxs)

    #Create a blank image for the reconstruction
    reconImage = np.zeros(sample.squareDim)

    #Retreive measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in sample.squareMeasuredIdxs]))
    measuredValues = inputImage[np.asarray(idxsX), np.asarray(idxsY)]

    #Compute reconstruction values using IDW (inverse distance weighting)
    reconImage[sample.squareUnMeasuredIdxs[:,0], sample.squareUnMeasuredIdxs[:,1]] = np.sum(measuredValues[neighborIndices]*neighborWeights, axis=1)

    #Combine measured values back into the reconstruction image
    reconImage[sample.squareMeasuredIdxs[:,0], sample.squareMeasuredIdxs[:,1]] = measuredValues

    return reconImage

#Perform the reconstruction with 0-padding; removes stretching in initial measurements
#def computeRecon(inputImage, maskObject):
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

def unet(numFilters, numChannels):

    inputs = Input(shape=(None,None,numChannels))
    
    conv1 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    down1 = Conv2D(numFilters, 2, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    
    conv2 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down1)
    conv2 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    down2 = Conv2D(numFilters, 2, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    
    conv3 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down2)
    conv3 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    down3 = Conv2D(numFilters, 2, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    
    conv4 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down3)
    conv4 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    up7 = Conv2D(numFilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(numFilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(numFilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    output = Conv2D(1, (1,1), activation='linear', padding='same')(conv9)
    output = tfp.math.clip_by_value_preserve_gradient(output, 0, 1)

    return tf.keras.Model(inputs=inputs, outputs=output)

#Extract features of interest from a reconstruction for propogation through network
def featureExtractor(sample, measuredImage, reconImage):

    #Stack recon values for only unmeasured locations
    tempReconImage = copy.deepcopy(reconImage)
    tempReconImage[sample.squareMask==1] = 0
    featureImage = np.dstack((measuredImage, tempReconImage))

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
