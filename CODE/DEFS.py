#==================================================================
#SLADS DEFINITIONS GENERAL
#==================================================================

#Function to generate metadata for multiple samples
@ray.remote
def SampleData_parhelper(sampleFolder, initialPercToScan, stopPerc, scanMethod, RDMethod, ignoreMissingLines, lineRevist, simulationFlag):
    return SampleData(sampleFolder, initialPercToScan, stopPerc, scanMethod, RDMethod, ignoreMissingLines, lineRevist, simulationFlag)

#Object for initializing and storing sample metadata
class SampleData:
    def __init__(self, sampleFolder, initialPercToScan, stopPerc, scanMethod, RDMethod, ignoreMissingLines, lineRevist, simulationFlag):
        
        #Save options as internal variables
        self.scanMethod = scanMethod
        self.initialPercToScan = initialPercToScan
        self.stopPerc = stopPerc
        self.ignoreMissingLines = ignoreMissingLines
        self.lineRevist = lineRevist
        self.simulationFlag = simulationFlag
        self.lineExt = None
        
        #Store location of MSI data and sample name
        self.sampleFolder = sampleFolder
        self.name = os.path.basename(sampleFolder)
        
        #Note which files have already been read
        self.readScanFiles = []
        self.readLines = []
        
        #Storage location for matching sequentially generated indexes with physical line numbers
        self.physicalLineNums = {}
        
        #Read in data from sampleInfo.txt
        lineIndex = 0
        sampleInfo = open(sampleFolder+os.path.sep+'sampleInfo.txt').readlines()

        #Read the max number of lines that are expected 
        self.numLines = int(sampleInfo[lineIndex].rstrip())
        lineIndex += 1

        #Read the sample width
        self.sampleWidth = float(sampleInfo[lineIndex].rstrip())
        lineIndex += 1

        #Read the sample height
        self.sampleHeight = float(sampleInfo[lineIndex].rstrip())
        lineIndex += 1

        #Read in scan rate
        self.scanRate = float(sampleInfo[lineIndex].rstrip())
        lineIndex += 1

        #Read in acquistion rate
        self.acqRate = float(sampleInfo[lineIndex].rstrip())
        lineIndex += 1

        #Read in monoisotopic mz (-1 will indicate None)
        self.mzMonoValue = float(sampleInfo[lineIndex].rstrip())
        lineIndex += 1
        
        #Read window tolerance (ppm)
        self.ppm = float(sampleInfo[lineIndex].rstrip())*1e-6
        lineIndex += 1

        #Number of decimals for mz precision
        self.mzResOrder = int(sampleInfo[lineIndex].rstrip())
        lineIndex += 1

        #Check if files are numbered sequentially, or according to physical position
        fileNumbering = int(sampleInfo[lineIndex].rstrip())
        lineIndex += 1

        #Process the read information as needed for regular use cases
        if overrideTIC: self.mzMonoValue = -1
        self.ppmPos, self.ppmNeg = 1+self.ppm, 1-self.ppm
        self.mzMonoRange = [self.mzMonoValue*self.ppmNeg, self.mzMonoValue*self.ppmPos]
        if fileNumbering==0: self.unorderedNames = False
        elif fileNumbering==0: self.unorderedNames = True
        else: sys.exit('Error - File Numbering parameter used in sampleInfo is not an acceptable value.')

        #Get mz ranges to use for visualizations
        try: self.mzValues = np.loadtxt(self.sampleFolder+os.path.sep+'mz.csv', delimiter=',')
        except: self.mzValues = np.loadtxt('mz.csv', delimiter=',')
        self.mzRanges = np.round(np.column_stack((self.mzValues*self.ppmNeg, self.mzValues*self.ppmPos)), self.mzResOrder)
        
        #Store final dimensions for physical domain, determining the number of columns for row-alignment interpolations
        self.finalDim = [self.numLines, int(round(((self.sampleWidth*1e3)/self.scanRate)*self.acqRate))]
        
        #If this is a simulation, then get MSI extension and check for missing lines if applicable
        if self.simulationFlag:
            extensions = list(map(lambda x:x, np.unique([filename.split('.')[-1] for filename in natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'), reverse=False)])))
            if 'd' in extensions: self.lineExt = '.d'
            elif 'D' in extensions: self.lineExt = '.D'
            elif 'raw' in extensions: self.lineExt = '.raw'
            elif 'RAW' in extensions: self.lineExt = '.RAW'
            else: sys.exit('Error! - Either no MSI files are present, or an unknown MSI filetype being used for sample: ' + self.name)
            scanFiles = natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'+self.lineExt), reverse=False)
            if self.ignoreMissingLines:
                self.missingLines = np.asarray(list(set(np.arange(1, self.finalDim[0]).tolist()) - set([int(scanFileName.split('line-')[1].split('.')[0].lstrip('0')) for scanFileName in scanFiles])))-1
                self.finalDim[0] -= len(self.missingLines)
            else: self.missingLines = np.asarray([])
        
        #Establish the total sample area; for determination of percMeasured
        self.area = int(round(self.finalDim[0]*self.finalDim[1]))
        
        #Setup objects for storing raw MSI data
        self.newTimes = np.linspace(0, ((self.sampleWidth*1e3)/self.scanRate)/60, self.finalDim[1])
        self.mzImages = np.zeros((len(self.mzRanges), self.finalDim[0], self.finalDim[1]))
        self.TIC = np.zeros((self.finalDim))
        if self.mzMonoValue != -1: self.mzMono = np.zeros((self.finalDim))
        
        #Determine image dimensions that will produce square pixels (consistent vertical/horizontal resolution)
        if(self.finalDim[1]/self.sampleWidth) > (self.finalDim[0]/self.sampleHeight): 
            self.squareDim = [int(round((self.finalDim[1]*self.sampleHeight)/self.sampleWidth)), self.finalDim[1]]
        elif (self.finalDim[1]/self.sampleWidth) < (self.finalDim[0]/self.sampleHeight):
            self.squareDim = [self.finalDim[0], int(round((self.finalDim[0]*self.sampleWidth)/self.sampleHeight))]
        else:
            self.squareDim = self.finalDim
            
        #Setup initial sets immediately
        self.generateInitialSets(self.scanMethod)
        
        #If a simulation, then just read all the data outright
        if self.simulationFlag: self.readScanData()
        
    #Generate initial scanning mask; allows changing the scan method without rescanning all of the data
    def generateInitialSets(self, scanMethod):
    
        #Update the scan method
        self.scanMethod = scanMethod
    
        #List of what points/lines should be initially measured
        self.initialSets = []
        
        #If scanning with line-bounded constraint
        if self.scanMethod == 'linewise':
        
            #Create list of arrays containing points to measure on each line
            self.linesToScan = np.asarray([[tuple([rowNum, columnNum]) for columnNum in np.arange(0, self.finalDim[1], 1)] for rowNum in np.arange(0, self.finalDim[0], 1)]).tolist()

            #Set initial lines to scan
            lineIndexes = [int(round((self.finalDim[0]-1)*0.50))]
            
            #Obtain points in the specified lines and add them to the initial scan list
            for lineIndex in lineIndexes:
                
                #If only a percentage should be scanned, then randomly select points, otherwise select all
                if lineMethod == 'percLine':
                    newIdxs = copy.deepcopy(self.linesToScan[lineIndex])
                    np.random.shuffle(newIdxs)
                    newIdxs = newIdxs[:int(np.ceil((self.stopPerc/100)*self.finalDim[1]))]
                else: 
                    newIdxs = [pt for pt in self.linesToScan[lineIndex]]
                
                #Add positions to initial list
                self.initialSets.append(newIdxs)
                
        elif self.scanMethod == 'pointwise' or self.scanMethod == 'random':
        
            #Randomly select points to initially scan
            newIdxs = np.transpose(np.where(np.zeros(self.finalDim)==0))
            np.random.shuffle(newIdxs)
            newIdxs = newIdxs[:int(np.ceil(((self.initialPercToScan/100)*self.area)))]
            
            #Add positions to initial list
            self.initialSets.append(newIdxs)
    
    def readScanData(self, mask=None):
    
        #Get the MSI file extension automatically if it isn't already known
        if self.lineExt == None:
            extensions = list(map(lambda x:x, np.unique([filename.split('.')[-1] for filename in natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'), reverse=False)])))
            if 'd' in extensions: self.lineExt = '.d'
            elif 'D' in extensions: self.lineExt = '.D'
            elif 'raw' in extensions: self.lineExt = '.raw'
            elif 'RAW' in extensions: self.lineExt = '.RAW'
            else: sys.exit('Error! - Either no MSI files are present, or an unknown MSI filetype being used for sample: ' + self.name)
        
        #Obtain and sort the available line files pertaining to the current scan
        scanFiles = natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'+self.lineExt), reverse=False)
        
        #Identify which files have not yet been scanned, if line revisiting is disabled (update not replace)
        if self.lineRevist == False: scanFiles = list(set(scanFiles)-set(self.readScanFiles))
        
        #For each of the MSI files identified
        for scanFileName in scanFiles:

            #Add a flag to indicate 'good'/'bad' data file (primarily checking for files without data)
            readErrorFlag = False

            #Establish file pointer and line number (1 indexed) for the specific scan
            try: data = mzFile(scanFileName)
            except: readErrorFlag = True
            
            #If the data file is 'good' then continue processing
            if not readErrorFlag:
            
                #Add file name to those already scanned
                self.readScanFiles.append(scanFileName)
                
                #Extract line number from the filename, removing leading zeros, subtract 1 for zero indexing
                lineNum = int(scanFileName.split('line-')[1].split('.')[0].lstrip('0'))-1
                
                #If the line numbers are not the physical row numbers, then obtain correct number from stored LUT
                if self.unorderedNames and impModel and scanMethod == 'linewise': lineNum = self.physicalLineNums[lineNum+1]
                
                #Record that the line number specified has been read previously
                self.readLines.append(lineNum)
                
                #If ignoring missing lines, then determine the offset for correct indexing
                if self.ignoreMissingLines and len(self.missingLines) > 0: lineNum -= int(np.sum(lineNum > self.missingLines))
                
                #Obtain the total ion chromatogram and extract original times
                ticData = np.asarray(data.xic(data.time_range()[0], data.time_range()[1]))
                origTimes, TICData = ticData[:,0], ticData[:,1]
                
                #If the data is being sparesly acquired, then the listed times in the file need to be shifted; convert np.float to float for method compatability
                if impModel and impOffset and scanMethod == 'linewise' and lineMethod == 'segLine': origTimes += (np.argwhere(mask[lineNum]==1).min()/self.finalDim[1])*(((self.sampleWidth*1e3)/self.scanRate)/60)
                elif impModel and impOffset: sys.exit('Error - Using implementation mode with an offset but not segmented-linewise operation is not a supported configuration.')
                
                #If normalizing by internal standard, then obtain the relevant data
                if self.mzMonoValue != -1: mzMonoData = np.asarray(data.xic(data.time_range()[0], data.time_range()[1], float(self.mzMonoRange[0]), float(self.mzMonoRange[1])))[:,1]
                
                #Read in specified mz ranges, normalize as specified, and interpolate to new times; convert np.float to float for method compatability
                for mzRangeNum in range(0, len(self.mzRanges)): 
                    mzData = np.asarray(data.xic(data.time_range()[0], data.time_range()[1], float(self.mzRanges[mzRangeNum][0]), float(self.mzRanges[mzRangeNum][1])))[:,1]
                    if self.mzMonoValue == -1: mzData = np.nan_to_num(mzData/TICData, nan=0, posinf=0, neginf=0)
                    else: mzData = np.nan_to_num(mzData/mzMonoData, nan=0, posinf=0, neginf=0)
                    self.mzImages[mzRangeNum, lineNum, :] = np.interp(self.newTimes, origTimes, np.nan_to_num(mzData, nan=0, posinf=0, neginf=0))
                
                #Interpolate TIC and internal standard (if applicable) to final new times for visualization
                self.TIC[lineNum] = np.interp(self.newTimes, origTimes, TICData) 
                if self.mzMonoValue != -1: self.mzMono[lineNum] = np.interp(self.newTimes, origTimes, mzMonoData)
        
        #Find the maximum value in each mz image for easy referencing
        self.mzImagesMax = np.max(self.mzImages, axis=(1,2))
        
        #Calculate the average mz image
        self.mzAvgImage = np.mean(self.mzImages, axis=0)
        
        #If only using a single mz as input to SLADS, then set average mz as the first indexed mz image
        if mzSingle: self.mzAvgImage = copy.deepcopy(self.mzImages[0,:,:])
        
        #Resize for square dimensions
        self.squaremzAvgImage = resize(self.mzAvgImage, tuple(self.squareDim), order=0)
        self.squaremzImages = np.moveaxis(resize(np.moveaxis(self.mzImages, 0, -1), tuple(self.squareDim), order=0), -1, 0)

#Relevant sample data at each time step; static information should be held in corresponding SampleData object
class Sample:
    def __init__(self, sampleData):
        
        #Setup variables that are expected to exist
        self.mask = np.zeros((sampleData.finalDim))
        self.squareRD = np.zeros((sampleData.squareDim))
        self.percMeasured = 0
        self.iteration = 0
        
    def performMeasurements(self, sampleData, newIdxs, model, cValue, bestCFlag, oracleFlag, fromRecon):
        
        #Ensure newIdxs are indexible in 2 dimensions
        newIdxs = np.atleast_2d(newIdxs)
        
        #Update mask of measured locations
        self.mask[newIdxs[:,0], newIdxs[:,1]] = 1
        
        #Update which positions have not yet been measured
        self.unMeasuredIdxs = np.transpose(np.where(self.mask==0))
        
        #If not taking values from a reconstruction, get from equipment or ground-truth; else get from the reconstruction
        if not fromRecon:
            #If not simulation, then read from equipment, otherwise mask ground-truth mz images by what should have been scanned
            if not sampleData.simulationFlag:
                print('Writing UNLOCK')
                with open(dir_ImpDataFinal + 'UNLOCK', 'w') as filehandle: _ = [filehandle.writelines(str(tuple([pos[0]+1, (pos[1]*sampleData.scanRate)/sampleData.acqRate]))+'\n') for pos in newIdxs.tolist()]
                if sampleData.unorderedNames and impModel and scanMethod == 'linewise': sampleData.physicalLineNums[len(sampleData.physicalLineNums.keys())+1] = int(newIdxs[0][0])
                equipWait()
                sampleData.readScanData(self.mask)
                self.mzImages = copy.deepcopy(sampleData.mzImages)
                self.mzAvgImage = copy.deepcopy(sampleData.mzAvgImage)
            else: 
                self.mzImages = sampleData.mzImages*self.mask
                self.mzAvgImage = sampleData.mzAvgImage*self.mask
        else:
            self.mzImages[:, newIdxs[:,0], newIdxs[:,1]] = self.mzReconImages[:, newIdxs[:,0], newIdxs[:,1]]
            self.mzAvgImage[newIdxs[:,0], newIdxs[:,1]] = self.mzAvgReconImage[newIdxs[:,0], newIdxs[:,1]]
            
        #Update percentage pixels measured; only when not fromRecon
        self.percMeasured = (np.sum(self.mask)/sampleData.area)*100
        
        #Resize and extract measured and unmeasured locations for the square pixel mask
        self.squareMask = resize(self.mask==1, tuple(sampleData.squareDim), order=0)
        squareMeasuredIdxs, squareUnMeasuredIdxs = np.transpose(np.where(self.squareMask==1)), np.transpose(np.where(self.squareMask==0))

        #Determine neighbor information for unmeasured locations
        neighborIndices, neighborWeights, neighborDistances = findNeighbors(squareMeasuredIdxs, squareUnMeasuredIdxs)
        
        self.distanceMap = np.zeros(tuple(sampleData.squareDim))
        self.distanceMap[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]] = neighborDistances[:,0]
        
        #Compute the reconstructions with square pixels if new data is acquired
        if not fromRecon:
        
            #Update the iteration counter
            self.iteration += 1
        
            squaremzImages = np.moveaxis(resize(np.moveaxis(self.mzImages, 0, -1), tuple(sampleData.squareDim), order=0), -1, 0)
            self.squaremzReconImages = computeRecon(squaremzImages, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
            self.mzReconImages = np.moveaxis(resize(np.moveaxis(self.squaremzReconImages , 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)        
            
            squaremzAvgImage = resize(self.mzAvgImage, tuple(sampleData.squareDim), order=0)
            self.squaremzAvgReconImage = computeRecon(squaremzAvgImage, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
            self.mzAvgReconImage = resize(self.squaremzAvgReconImage, tuple(sampleData.finalDim), order=0)
            
        #If needed, or might be needed, compute information for SLADS models; not needed in DLADS
        if (oracleFlag or bestCFlag or erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net'): 
            self.polyFeatures = computePolyFeatures(sampleData, self.squaremzAvgReconImage, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights, neighborDistances)
            self.squareRDValues = self.squareRD[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]]
            
        #Compute RD/ERD
        if oracleFlag or bestCFlag:
        
            #If this is a full measurement step, compute the RDPP; for original, compute difference of collapsed mz stack, otherwise collapse mz difference stack 
            if not fromRecon:
                if RDMethod == 'var': self.RDPP = np.var(abs(sampleData.squaremzImages-self.squaremzReconImages ), axis=0)
                elif RDMethod == 'max': self.RDPP = np.max(abs(sampleData.squaremzImages-self.squaremzReconImages ), axis=0)
                elif RDMethod == 'avg': self.RDPP = np.mean(abs(sampleData.squaremzImages-self.squaremzReconImages ), axis=0)
                elif RDMethod == 'sum': self.RDPP = np.sum(abs(sampleData.squaremzImages-self.squaremzReconImages ), axis=0)
                elif RDMethod == 'original': self.RDPP = computeDifference(sampleData.squaremzAvgImage, self.squaremzAvgReconImage)
                else: sys.exit('Error! - Unknown RD Method specified in configuration: ' + RDMethod)
                
            computeRD(self, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborDistances, cValue, bestCFlag, fromRecon)
            self.RD = resize(self.squareRD, tuple(sampleData.finalDim), order=0)
            self.ERD = self.RD
        else: self.ERD = computeERD(self, sampleData, model, squareUnMeasuredIdxs, squareMeasuredIdxs)

#Sample scanning progress and final results processing
class Result:
    def __init__(self, sampleData, liveOutputFlag, dir_Results, bestCFlag, cValue):
    
        self.sampleData = sampleData
        self.cValue = cValue
        self.bestCFlag = copy.deepcopy(bestCFlag)
        self.samples = []
        self.cSelectionList = []
        self.startTime = time.time()
        self.lastMask = None
        self.percsMeasured = []
        self.liveOutputFlag = liveOutputFlag
        self.dir_Results = dir_Results
        
        if animationGen and dir_Results != None:

            #Setup/clean base sample directory
            self.dir_sampleResults = self.dir_Results + self.sampleData.name + os.path.sep
            if os.path.exists(self.dir_sampleResults): shutil.rmtree(self.dir_sampleResults)
            os.makedirs(self.dir_sampleResults)
            
            #Prepare subdirectories; for frames and videos of mz progressions
            self.dir_mzProgression = self.dir_sampleResults + 'mz' + os.path.sep
            os.makedirs(self.dir_mzProgression)
            self.dir_mzProgressions = [self.dir_mzProgression + str(self.sampleData.mzRanges[mzNum][0]) + '-' + str(self.sampleData.mzRanges[mzNum][1]) + os.path.sep for mzNum in range(0, len(self.sampleData.mzRanges))]
            for dir_mzProgressionSub in self.dir_mzProgressions: os.makedirs(dir_mzProgressionSub)
            self.dir_avgProgression = self.dir_sampleResults + 'Average' + os.path.sep
            os.makedirs(self.dir_avgProgression)
            self.dir_videos= self.dir_sampleResults + 'Videos' + os.path.sep
            os.makedirs(self.dir_videos)
            
        if liveOutputFlag: self.visualize_groundTruth()
        
    def update(self, sample):
    
        #Update measurement mask and percentage of FOV measured at this step 
        self.lastMask = copy.deepcopy(sample.mask)
        self.percsMeasured.append(copy.deepcopy(sample.percMeasured))
        
        #If outputs should be produced at every update step, then do so, determining related metrics as needed
        if self.liveOutputFlag: 
            if self.sampleData.simulationFlag: self.extractSimulationData(sample)
            visualize_serial(sample, self.sampleData, self.dir_avgProgression, self.dir_mzProgressions, False)
        
        #If evaluating a c parameter, then find the PSNR of the current reconstructions, otherwise save a copy of the measurement step for later evaluation
        if self.bestCFlag: 
            self.cSelectionList.append(np.mean([compare_psnr(self.sampleData.mzImages[index], sample.mzReconImages[index], data_range=self.sampleData.mzImagesMax[index]) for index in range(0, len(self.sampleData.mzImages))]))
        else:
            self.samples.append(copy.deepcopy(sample))
            self.finalTime = time.time()-self.startTime
    
    #For a given measurement step find PSNR/SSIM of reconstructions, compute the RD, find PSNR of ERD
    def extractSimulationData(self, sample):
    
        #Resize and extract measured and unmeasured locations for the square pixel mask
        squareMeasuredIdxs, squareUnMeasuredIdxs = np.transpose(np.where(sample.squareMask==1)), np.transpose(np.where(sample.squareMask==0))
        
        #Determine neighbor information for unmeasured locations
        neighborIndices, neighborWeights, neighborDistances = findNeighbors(squareMeasuredIdxs, squareUnMeasuredIdxs)
        
        #If using a single mz channel as the average image in SLADS operation
        if mzSingle: 
            
            #Calculate actual averaged mz image
            sample.mzAvgImage = self.sampleData.mzAvgImage*sample.mask
            squaremzAvgImage = resize(sample.mzAvgImage, tuple(self.sampleData.squareDim), order=0)
            
            #Perform reconstruction of corrected averaged image
            sample.squaremzAvgReconImage = computeRecon(squaremzAvgImage, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
            sample.mzAvgReconImage = resize(sample.squaremzAvgReconImage, tuple(self.sampleData.finalDim), order=0)
        
        sample.mzImagePSNRList = [compare_psnr(self.sampleData.mzImages[index], sample.mzReconImages[index], data_range=self.sampleData.mzImagesMax[index]) for index in range(0, len(self.sampleData.mzImages))]
        sample.avgmzImagePSNR = compare_psnr(self.sampleData.mzAvgImage, sample.mzAvgReconImage, data_range=np.max(self.sampleData.mzAvgImage))
        sample.mzImageSSIMList = [compare_ssim(self.sampleData.mzImages[index], sample.mzReconImages[index], data_range=self.sampleData.mzImagesMax[index]) for index in range(0, len(self.sampleData.mzImages))]
        sample.avgmzImageSSIM = compare_ssim(self.sampleData.mzAvgImage, sample.mzAvgReconImage, data_range=np.max(self.sampleData.mzAvgImage))
        
        #Compute the RDPP; for original, compute difference of collapsed mz stack, otherwise collapse mz difference stack 
        if RDMethod == 'var': sample.RDPP = np.var(abs(self.sampleData.squaremzImages-sample.squaremzReconImages ), axis=0)
        elif RDMethod == 'max': sample.RDPP = np.max(abs(self.sampleData.squaremzImages-sample.squaremzReconImages ), axis=0)
        elif RDMethod == 'avg': sample.RDPP = np.mean(abs(self.sampleData.squaremzImages-sample.squaremzReconImages ), axis=0)
        elif RDMethod == 'sum': sample.RDPP = np.sum(abs(self.sampleData.squaremzImages-sample.squaremzReconImages ), axis=0)
        elif RDMethod == 'original': sample.RDPP = computeDifference(self.sampleData.squaremzAvgImage, sample.squaremzAvgReconImage)
        else: sys.exit('Error! - Unknown RD Method specified in configuration: ' + RDMethod)
        
        #Compute the actual square RD
        computeRD(sample, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborDistances, self.cValue, self.bestCFlag, False)
        
        #Rescale both the ERD and RD
        #sample.squareRD = (sample.squareRD-np.min(sample.squareRD))/(np.max(sample.squareRD)-np.min(sample.squareRD))
        #sample.ERD = (sample.ERD-np.min(sample.ERD))/(np.max(sample.ERD)-np.min(sample.ERD))
        
        #Determine SSIM/PSNR between RD and ERD
        #sample.ERDPSNR = compare_psnr(sample.squareRD, sample.ERD, data_range=1)
        #sample.ERDSSIM = compare_ssim(sample.squareRD, sample.ERD, data_range=1)

        #Determine SSIM/PSNR between RD and ERD
        maxRangeValue = np.max([sample.squareRD, sample.ERD])
        sample.ERDPSNR = compare_psnr(sample.squareRD, sample.ERD, data_range=maxRangeValue)
        sample.ERDSSIM = compare_ssim(sample.squareRD, sample.ERD, data_range=maxRangeValue)
        
        #Resize for visualization
        sample.ERD = resize(sample.ERD, tuple(self.sampleData.finalDim), order=0)
        sample.RD = resize(sample.squareRD, tuple(self.sampleData.finalDim), order=0)
    
    #Generate visualiations/metrics as needed at the end of scanning
    def complete(self):
        
        #Make sure samples is writable
        self.samples = copy.deepcopy(self.samples)
        
        #If the filenames were unordered, then save the mapping from filename to physical row
        if self.sampleData.unorderedNames: np.savetxt(self.dir_sampleResults+'physicalLineNums.csv', np.asarray(list(self.sampleData.physicalLineNums.items())), delimiter=',', fmt='%d')
        
        #Save a copy of the final measurement mask
        np.savetxt(self.dir_sampleResults+'measuredMask.csv', self.samples[-1].mask, delimiter=',', fmt='%d')
        
        #If was using a single mz as network input, then reset the sampleData average input to actual value
        if mzSingle: self.sampleData.mzAvgImage = np.mean(self.sampleData.mzImages, axis=0)
        
        #If this is a simulation, then can compare against ground-truth information
        if self.sampleData.simulationFlag:
            
            #If not done during acquisiton, then for each of the measurement steps find PSNR/SSIM of reconstructions, compute the RD, find PSNR of ERD
            if not self.liveOutputFlag: 
                for sample in self.samples: self.extractSimulationData(sample)
            
            #Summarize scores for testing printout
            self.mzAvgPSNRList = [np.mean(sample.mzImagePSNRList) for sample in self.samples]
            self.avgPSNRList = [sample.avgmzImagePSNR for sample in self.samples]
            self.ERDPSNRList = [sample.ERDPSNR for sample in self.samples]
            self.mzAvgSSIMList = [np.mean(sample.mzImageSSIMList) for sample in self.samples]
            self.avgSSIMList = [sample.avgmzImageSSIM for sample in self.samples]
            self.ERDSSIMList = [sample.ERDSSIM for sample in self.samples]
        
        #If an animation will be produced and the run has completed
        if animationGen:
            
            #Generate visualizations if they are not created during operation
            if not self.liveOutputFlag:
                self.visualize_groundTruth()
                if parallelization:
                    futures = [visualize_parhelper.remote(sample, self.sampleData, self.dir_avgProgression, self.dir_mzProgressions, True) for sample in self.samples]
                    _ = ray.get(futures)
                else:
                    _ = [visualize_serial(sample, self.sampleData, self.dir_avgProgression, self.dir_mzProgressions, False) for sample in tqdm(self.samples, desc='Steps', leave=False, ascii=True)]
            
            #Combine mz images into animations
            for mzNum in tqdm(range(0, len(self.sampleData.mzRanges)), desc='mz Videos', leave = False, ascii=True): 
                dataFileNames = natsort.natsorted(glob.glob(self.dir_mzProgressions[mzNum] + 'progression_*.png'))
                height, width, layers = cv2.imread(dataFileNames[0]).shape
                animation = cv2.VideoWriter(self.dir_videos + str(self.sampleData.mzRanges[mzNum][0]) + '-' + str(self.sampleData.mzRanges[mzNum][1]) + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
                for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                animation.release()
                animation = None

            #Combine average images into animation
            dataFileNames = natsort.natsorted(glob.glob(self.dir_avgProgression + 'progression_*.png'))
            height, width, layers = cv2.imread(dataFileNames[0]).shape
            animation = cv2.VideoWriter(self.dir_videos + 'average.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
            for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
            animation.release()
            animation = None

    #Visualize ground-truth data for simulations
    def visualize_groundTruth(self):
        #Ground truth borderless avg image
        if self.sampleData.simulationFlag:
            saveLocation = self.dir_avgProgression + 'avgGroundTruth.png'
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
            plt.axis('off')
            if not sysLogNorm: plt.imshow(self.sampleData.mzAvgImage, cmap='hot', aspect='auto')
            if sysLogNorm: plt.imshow(self.sampleData.mzAvgImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(self.sampleData.mzAvgImage)+3*np.std(self.sampleData.mzAvgImage), base=10, vmin=np.min(self.sampleData.mzAvgImage), vmax=np.max(self.sampleData.mzAvgImage)))
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(saveLocation, bbox_inches=extent)
            plt.close()
        
        #Ground truth borderless mz images
        if self.sampleData.simulationFlag:
            for mzNum in range(0, len(self.sampleData.mzRanges)):
                saveLocation = self.dir_mzProgressions[mzNum] + 'groundTruth_mz_' + str(self.sampleData.mzRanges[mzNum][0]) + '-' + str(self.sampleData.mzRanges[mzNum][1]) + '.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                if not sysLogNorm: plt.imshow(self.sampleData.mzImages[mzNum], cmap='hot', aspect='auto')
                if sysLogNorm: plt.imshow(self.sampleData.mzImages[mzNum], cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=np.mean(self.sampleData.mzImages[mzNum])+3*np.std(self.sampleData.mzImages[mzNum]), base=10, vmin=np.min(self.sampleData.mzImages[mzNum]), vmax=np.max(self.sampleData.mzImages[mzNum])))
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()

#Visualize multiple sample progression steps at once
@ray.remote
def visualize_parhelper(sample, sampleData, dir_avgProgression, dir_mzProgressions, parallel):
    return visualize_serial(sample, sampleData, dir_avgProgression, dir_mzProgressions, parallel)

#Visualize single sample progression step
def visualize_serial(sample, sampleData, dir_avgProgression, dir_mzProgressions, parallel):

    #If in a parallel thread, re-import libraries inside of thread to set plotting backend as non-interactive
    if parallel:
        import matplotlib
        matplotlib.use('agg')

    #Turn percent measured into a string
    percMeasured = "{:.2f}".format(sample.percMeasured)
    
    #Turn metrics into strings
    if sampleData.simulationFlag: 
        avgmzImagePSNR = "{:.2f}".format(sample.avgmzImagePSNR)
        avgmzImageSSIM = "{:.2f}".format(sample.avgmzImageSSIM)
        erdPSNR = "{:.2f}".format(sample.ERDPSNR)
        erdSSIM = "{:.2f}".format(sample.ERDSSIM)
        mzImageAvgPSNR = "{:.2f}".format(np.mean(sample.mzImagePSNRList))
        mzImageAvgSSIM = "{:.2f}".format(np.mean(sample.mzImageSSIMList))

    #For each of the mz ranges, generate visuals
    for mzNum in range(0, len(sampleData.mzRanges)):
        
        mzMinValue, mzMaxValue, mzLinThreshValue = np.min(sampleData.mzImages[mzNum]), np.max(sampleData.mzImages[mzNum]), np.mean(sampleData.mzImages[mzNum])+3*np.std(sampleData.mzImages[mzNum])

        #Turn metrics into strings
        massRange = str(sampleData.mzRanges[mzNum][0]) + '-' + str(sampleData.mzRanges[mzNum][1])
        if sampleData.simulationFlag: 
            mzImagePSNR = "{:.2f}".format(sample.mzImagePSNRList[mzNum])
            mzImageSSIM = "{:.2f}".format(sample.mzImageSSIMList[mzNum])
        
        #Measured mz image
        if sampleData.simulationFlag: f = plt.figure(figsize=(20,5.3865))
        else:  f = plt.figure(figsize=(15,5.3865))
    
        if sampleData.simulationFlag: plt.suptitle(r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ mz:\ }$" + massRange + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured + '\n' + r"$\bf{PSNR - mz\ Recon:\ }$" + mzImagePSNR + r"$\bf{\ \ Average\ mz\ Recon:\ }$" + avgmzImagePSNR+ '\n' + r"$\bf{SSIM - mz\ Recon:\ }$" + mzImageSSIM + r"$\bf{\ \ Average\ mz\ Recon:\ }$" + avgmzImageSSIM)
        else: plt.suptitle(r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ mz:\ }$" + massRange + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured)

        if sampleData.simulationFlag: ax = plt.subplot2grid(shape=(1,3), loc=(0,0))
        else: ax = plt.subplot2grid(shape=(1,2), loc=(0,0))
        im = ax.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Sampled Mask')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

        if sampleData.simulationFlag:
            ax = plt.subplot2grid(shape=(1,3), loc=(0,1))
            if not sysLogNorm: im = ax.imshow(sampleData.mzImages[mzNum], cmap='hot', aspect='auto', vmin=mzMinValue, vmax=mzMaxValue)
            if sysLogNorm: im = ax.imshow(sampleData.mzImages[mzNum], cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=mzLinThreshValue, base=10, vmin=mzMinValue, vmax=mzMaxValue))
            ax.set_title('Ground-Truth')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        if sampleData.simulationFlag: ax = plt.subplot2grid(shape=(1,3), loc=(0,2))
        else: ax = plt.subplot2grid(shape=(1,2), loc=(0,1))
        if not sysLogNorm: im = ax.imshow(sample.mzReconImages[mzNum], cmap='hot', aspect='auto', vmin=mzMinValue, vmax=mzMaxValue)
        if sysLogNorm: im = ax.imshow(sample.mzReconImages[mzNum], cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=mzLinThreshValue, base=10, vmin=mzMinValue, vmax=mzMaxValue))
        ax.set_title('Reconstruction')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        #Save
        f.tight_layout()
        f.subplots_adjust(top = 0.75)
        saveLocation = dir_mzProgressions[mzNum] + 'progression_mz_' + massRange + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) +'.png'
        plt.savefig(saveLocation, bbox_inches='tight')
        plt.close()
        
        #Do borderless saves for each mz image here; mask will be the same as produced in the average output
        saveLocation = dir_mzProgressions[mzNum] + 'reconstruction_mz_' + massRange + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        if not sysLogNorm: plt.imshow(sample.mzReconImages[mzNum], cmap='hot', aspect='auto', vmin=mzMinValue, vmax=mzMaxValue)
        if sysLogNorm: plt.imshow(sample.mzReconImages[mzNum], cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=mzLinThreshValue, base=10, vmin=mzMinValue, vmax=mzMaxValue))
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()
        
        saveLocation = dir_mzProgressions[mzNum] + 'measured_mz_' + massRange + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        if not sysLogNorm: plt.imshow(sample.mzImages[mzNum], cmap='hot', aspect='auto', vmin=mzMinValue, vmax=mzMaxValue)
        if sysLogNorm: plt.imshow(sample.mzImages[mzNum], cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=mzLinThreshValue, base=10, vmin=mzMinValue, vmax=mzMaxValue))
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()
        
    #For the average, generate visual
    avgMinValue, avgMaxValue, avgLinThreshValue = np.min(sampleData.mzAvgImage), np.max(sampleData.mzAvgImage), np.mean(sampleData.mzAvgImage)+3*np.std(sampleData.mzAvgImage)
    if sampleData.simulationFlag: f = plt.figure(figsize=(20,10))
    else: f = plt.figure(figsize=(20,5.3865))
    
    if sampleData.simulationFlag: plt.suptitle(r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured + '\n' + r"$\bf{PSNR - Average\ Recon: }$" + avgmzImagePSNR + r"$\bf{\ \ Average\ mz\ Recon:\ }$" + mzImageAvgPSNR + r"$\bf{\ \ ERD:\ }$" + erdPSNR + '\n' + r"$\bf{SSIM - Average\ Recon: }$" + avgmzImageSSIM + r"$\bf{\ \ Average\ mz\ Recon:\ }$" + mzImageAvgSSIM + r"$\bf{\ \ ERD:\ }$" + erdSSIM)
    else: plt.suptitle(r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured)
    
    if sampleData.simulationFlag: 
        ax = plt.subplot2grid(shape=(2,3), loc=(0,0))
        if not sysLogNorm: im = ax.imshow(sampleData.mzAvgImage, cmap='hot', aspect='auto', vmin=avgMinValue, vmax=avgMaxValue)
        if sysLogNorm: im = ax.imshow(sampleData.mzAvgImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=avgLinThreshValue, base=10, vmin=avgMinValue, vmax=avgMaxValue))
        ax.set_title('Ground-Truth')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if sampleData.simulationFlag: ax = plt.subplot2grid((2,3), (0,1))
    else: ax = plt.subplot2grid((1,3), (0,0))
    if not sysLogNorm: im = ax.imshow(sample.mzAvgReconImage, cmap='hot', aspect='auto', vmin=avgMinValue, vmax=avgMaxValue)
    if sysLogNorm: im = ax.imshow(sample.mzAvgReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=avgLinThreshValue, base=10, vmin=avgMinValue, vmax=avgMaxValue))
    ax.set_title('Reconstruction')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if sampleData.simulationFlag: 
        ax = plt.subplot2grid((2,3), (0,2))
        if not sysLogNorm: im = ax.imshow(abs(sampleData.mzAvgImage-sample.mzAvgReconImage), cmap='hot', aspect='auto', vmin=avgMinValue, vmax=avgMaxValue)
        if sysLogNorm: im = ax.imshow(abs(sampleData.mzAvgImage-sample.mzAvgReconImage), cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=avgLinThreshValue, base=10, vmin=avgMinValue, vmax=avgMaxValue))
        ax.set_title('Absolute Difference')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if sampleData.simulationFlag: ax = plt.subplot2grid((2,3), (1,0))
    else: ax = plt.subplot2grid((1,3), (0,1))
    im = ax.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Measurement Mask')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    if sampleData.simulationFlag: ax = plt.subplot2grid((2,3), (1,1))
    else: ax = plt.subplot2grid((1,3), (0,2))
    #im = ax.imshow(sample.ERD, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    im = ax.imshow(sample.ERD, cmap='viridis', vmin=0, aspect='auto')
    ax.set_title('ERD')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if sampleData.simulationFlag: 
        ax = plt.subplot2grid((2,3), (1,2))
        #im = ax.imshow(sample.RD, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        im = ax.imshow(sample.RD, cmap='viridis', vmin=0, aspect='auto')
        ax.set_title('RD')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    #Save
    f.tight_layout()
    f.subplots_adjust(top = 0.85)
    saveLocation = dir_avgProgression + 'progression' + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '_avg.png'
    plt.savefig(saveLocation, bbox_inches='tight')
    plt.close()

    #Borderless saves
    saveLocation = dir_avgProgression + 'reconstruction' + '_iter_' + str(sample.iteration) +  '_perc_' + str(sample.percMeasured) + '.png'
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    if not sysLogNorm: plt.imshow(sample.mzAvgReconImage, cmap='hot', aspect='auto', vmin=avgMinValue, vmax=avgMaxValue)
    if sysLogNorm: plt.imshow(sample.mzAvgReconImage, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=avgLinThreshValue, base=10, vmin=avgMinValue, vmax=avgMaxValue))
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(saveLocation, bbox_inches=extent)
    plt.close()
    
    saveLocation = dir_avgProgression + 'mask_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(saveLocation, bbox_inches=extent)
    plt.close()
    
    saveLocation = dir_avgProgression + 'ERD_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    #plt.imshow(sample.ERD, aspect='auto', vmin=0, vmax=1)
    plt.imshow(sample.ERD, aspect='auto', vmin=0)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(saveLocation, bbox_inches=extent)
    plt.close()
    
    saveLocation = dir_avgProgression + 'measured_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    if not sysLogNorm: plt.imshow(sample.mzAvgReconImage*sample.mask, cmap='hot', aspect='auto', vmin=avgMinValue, vmax=avgMaxValue)
    if sysLogNorm: plt.imshow(sample.mzAvgReconImage*sample.mask, cmap='hot', aspect='auto', norm=matplotlib.colors.SymLogNorm(linthresh=avgLinThreshValue, base=10, vmin=avgMinValue, vmax=avgMaxValue))
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(saveLocation, bbox_inches=extent)
    plt.close()
    
    if sampleData.simulationFlag:
        saveLocation = dir_avgProgression + 'RD_iter_' + str(sample.iteration) + 'perc_' + str(sample.percMeasured) + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        #plt.imshow(sample.RD, aspect='auto', vmin=0, vmax=1)
        plt.imshow(sample.RD, aspect='auto', vmin=0)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()

@ray.remote
def runSLADS_parhelper(sampleData, cValue, modelAvailable, percToScan, percToViz, bestCFlag, oracleFlag, lineVisitAll, liveOutputFlag, dir_Results, tqdmHide):
    
    #If in a parallel thread, re-import libraries inside of thread to set plotting backend as non-interactive
    import matplotlib
    matplotlib.use('agg')
    
    return runSLADS(sampleData, cValue, modelAvailable, percToScan, percToViz, bestCFlag, oracleFlag, lineVisitAll, liveOutputFlag, dir_Results, tqdmHide)

def runSLADS(sampleData, cValue, modelAvailable, percToScan, percToViz, bestCFlag, oracleFlag, lineVisitAll, liveOutputFlag, dir_Results, tqdmHide):
    
    if modelAvailable:
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': model = np.load(dir_TrainingResults+'model_cValue_'+str(cValue)+'.npy', allow_pickle=True).item()
        #elif erdModel == 'DLADS' and lossFunc == 'Dice': model = tf.keras.models.load_model(dir_TrainingResults+'model_cValue_'+str(cValue), custom_objects={'loss': DiceLoss()})
        #elif erdModel == 'DLADS' and lossFunc == 'Jaccard': model = tf.keras.models.load_model(dir_TrainingResults+'model_cValue_'+str(cValue), custom_objects={'loss': JaccardLoss()})
        elif erdModel == 'DLADS': model = tf.keras.models.load_model(dir_TrainingResults+'model_cValue_'+str(cValue))
    else: model = None
    
    #Make sure random selection is consistent
    if consistentSeed: np.random.seed(0)
    
    #If groupwise is active, specify how many points should be scanned each step if pointwise, or random
    if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random') and percToScan!=None: sampleData.pointsToScan = int(np.ceil(((stopPerc/100)*sampleData.area)/(stopPerc/percToScan)))
    
    #If linewise acquisiton, sepcify how many points should be scanned on each line 
    if sampleData.scanMethod == 'linewise': sampleData.pointsToScan = int(np.ceil((stopPerc/100)*sampleData.finalDim[1]))
    
    #Create a new sample object to hold current information
    sample = Sample(sampleData)
    
    #Indicate that the stopping condition has not yet been met
    completedRunFlag = False
    
    #Create a new result object to hold scanning progression
    result = Result(sampleData, liveOutputFlag, dir_Results, bestCFlag, cValue)
    
    #Scan initial sets
    for initialSet in sampleData.initialSets: sample.performMeasurements(sampleData, initialSet, model, cValue, bestCFlag, oracleFlag, False)
    
    #Check stopping criteria, just in case of a bad input
    if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random' or not lineVisitAll) and (sample.percMeasured >= sampleData.stopPerc): completedRunFlag = True
    elif sampleData.scanMethod == 'linewise' and sampleData.finalDim[0]-np.sum(np.sum(sample.mask, axis=1)>0) == 0: completedRunFlag = True
    if np.sum(sample.ERD) == 0: completedRunFlag = True
    
    #Perform the first update for the result
    result.update(sample)
    
    if not lineVisitAll or scanMethod != 'linewise': maxProgress = stopPerc
    else: maxProgress = 100
    
    #Until the stopping criteria has been met
    with tqdm(total = float(maxProgress), desc = '% Sampled', leave=False, ascii=True, disable=tqdmHide) as pbar:

        #Initialize progress bar state according to % measured
        pbar.n = np.clip(round(sample.percMeasured,2), 0, maxProgress)
        pbar.refresh()
        
        #Until the program has completed
        while not completedRunFlag:
        
            #Find next measurement locations
            newIdxs = findNewMeasurementIdxs(sample, sampleData, result, model, cValue, percToScan, oracleFlag, bestCFlag)
            
            #Perform measurements, reconstructions and ERD/RD computations
            if len(newIdxs) != 0: sample.performMeasurements(sampleData, newIdxs, model, cValue, bestCFlag, oracleFlag, False)
            else: completedRunFlag = True
            
            #Check stopping criteria
            if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random' or not lineVisitAll) and (sample.percMeasured >= sampleData.stopPerc): completedRunFlag = True
            elif sampleData.scanMethod == 'linewise' and sampleData.finalDim[0]-np.sum(np.sum(sample.mask, axis=1)>0) == 0: completedRunFlag = True
            if np.sum(sample.ERD) == 0: completedRunFlag = True
            
            #If viz limit, only update when percToViz has been met; otherwise update every iteration
            if ((percToViz != None) and ((sample.percMeasured-result.percsMeasured[-1]) >= percToViz)) or (percToViz == None) or sampleData.scanMethod == 'linewise' or completedRunFlag: result.update(sample)
            
            #Update the progress bar
            pbar.n = np.clip(round(sample.percMeasured,2), 0, maxProgress)
            pbar.refresh()
            
    return result
     
#Section of computeRDValue that is supported for Numba acceleration
@jit(nopython=True)
def secondComputeRDValue(image, location, radius, windowSize, gaussianValues):
    
    #Initiate the window with zeros (in case of edge overlap)
    window = np.zeros((windowSize, windowSize))

    #Determine indexing locations for the window and image, considering possible edge overlap
    if location[1]-radius < 0:
        windowXStart = -(location[1]-radius)
        imageXStart = 0
    else:
        windowXStart = 0
        imageXStart = location[1]-radius
    if location[1]+radius > image.shape[1]:
        windowXStop = windowSize-(location[1]+radius-image.shape[1])
        imageXStop = image.shape[1]
    else:
        windowXStop = windowSize
        imageXStop = location[1]+radius
    if location[0]-radius < 0:
        windowYStart = -(location[0]-radius)
        imageYStart = 0
    else:
        windowYStart = 0
        imageYStart = location[0]-radius
    if location[0]+radius > image.shape[0]:
        windowYStop = windowSize-(location[0]+radius-image.shape[0])
        imageYStop = image.shape[0]
    else:
        windowYStop = windowSize
        imageYStop = location[0]+radius

    #Extract window from image
    window[int(windowYStart):int(windowYStop), int(windowXStart):int(windowXStop)] = image[int(np.ceil(imageYStart)):int(np.ceil(imageYStop)), int(np.ceil(imageXStart)):int(np.ceil(imageXStop))]
    
    return np.sum(window*np.outer(gaussianValues, gaussianValues))
    
#Compute RD values around location by radius 3 times the given sigma value
def computeRDValue(image, location, sigma):
    if legacyFlag: return secondComputeRDValue(image, location, 7.5, 15, signal.gaussian(15, sigma))
    windowSize = int(np.ceil(sigma*3*2))
    if windowSize%2==0: windowSize+=1
    radius = windowSize/2
    return secondComputeRDValue(image, location, radius, windowSize, signal.gaussian(windowSize, sigma))
   
#Perform gaussianGenerator for a set of sigma values
@ray.remote
def gaussian_parhelper(RDPP, idxs, sigmaValues, indexes):
    return [computeRDValue(RDPP, idxs[index], sigmaValues[index]) for index in indexes]

#Perform Reduction in Distortion computation
def computeRD(sample, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborDistances, cValue, bestCFlag, update):
    
    #If a full calculation of RD then use the squareUnMeasured locations, otherwise find those that should be updated
    if not update: 
        unMeasuredLocations = squareUnMeasuredIdxs
        neighborDistances = neighborDistances[:,0]
    else:
        unMeasuredLocations = np.empty((0,2)).astype(int)
        updateLocations = np.argwhere(sample.prevSquareMask-sample.squareMask)
        
        #Prepare variables for indexing
        updateLocations_list = updateLocations.tolist()
        squareMeasuredIdxs_list = squareMeasuredIdxs.tolist()
        neighborIndices = neighborIndices[:,0].ravel()
        
        #Find indices of updateLocations and then the indices of neighboring unMeasuredLocations 
        indices = [squareMeasuredIdxs_list.index(updateLocations_list[index]) for index in range(0, len(updateLocations))]
        indices = np.concatenate([np.argwhere(neighborIndices==index) for index in indices]).flatten()

        #If there are no locations that need updating, then just return
        if len(indices)==0: return
        
        #Extract unMeasuredLocations to be updated and their relevant neighbor information (to avoid recalculation)
        neighborDistances = neighborDistances[:,0][indices]
        unMeasuredLocations = squareUnMeasuredIdxs[indices]
        
    #Calculate the sigma values for chosen c value
    sigmaValues = neighborDistances/cValue
    
    #Determine RDValues, parallelizing if not done so at a higher level when it isn't just an update
    if not bestCFlag and parallelization and not update:
        RDPP_id = ray.put(sample.RDPP)
        sigmaValues_id = ray.put(sigmaValues)
        idxs_id = ray.put(unMeasuredLocations)
        sample.squareRD[tuple(unMeasuredLocations.T)] = np.asarray(list(chain.from_iterable(ray.get([gaussian_parhelper.remote(RDPP_id, idxs_id, sigmaValues_id, indexes) for indexes in np.array_split(np.arange(0, len(unMeasuredLocations)), multiprocessing.cpu_count())]))))
    else:
        sample.squareRD[tuple(unMeasuredLocations.T)] = np.asarray([computeRDValue(sample.RDPP, unMeasuredLocations[index], sigmaValues[index]) for index in range(0, len(unMeasuredLocations))])

    #Ensure RD values at scanned locations are set to zero
    sample.squareRD[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]] = 0
    
    #Update the previous mask, so measurement locations can be isolated in future updates
    sample.prevSquareMask = copy.deepcopy(sample.squareMask)

#Extract features of the reconstruction to use as inputs to SLADS(-Net) models
def computePolyFeatures(sampleData, reconImage, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights, neighborDistances):
    
    #Retreive recon values
    inputValues = reconImage[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]]
    
    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in squareMeasuredIdxs]))
    measuredValues = reconImage[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Find neighbor information
    neighborValues = measuredValues[neighborIndices]
    
    #Create array to hold features
    feature = np.zeros((np.shape(squareUnMeasuredIdxs)[0],6))
    
    #Compute std div features
    diffVect = computeDifference(neighborValues, np.transpose(np.matlib.repmat(inputValues, np.shape(neighborValues)[1],1)))
    feature[:,0] = np.sum(neighborWeights*diffVect, axis=1)
    feature[:,1] = np.sqrt(np.sum(np.power(diffVect,2),axis=1))
    
    #Compute distance/density features
    cutoffDist = np.ceil(np.sqrt((featDistCutoff/100)*(sampleData.area/np.pi)))
    feature[:,2] = neighborDistances[:,0]
    neighborsInCircle = np.sum(neighborDistances<=cutoffDist,axis=1)
    feature[:,3] = (1+(np.pi*(np.square(cutoffDist))))/(1+neighborsInCircle)
    
    #Compute gradient features; assume continuous features
    gradientImageX, gradientImageY = np.gradient(reconImage)
    feature[:,4] = abs(gradientImageY)[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]]
    feature[:,5] = abs(gradientImageX)[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]]
    
    #Fit polynomial features to the determined array
    polyFeatures = PolynomialFeatures(degree=2).fit_transform(feature)
    
    return polyFeatures

#Prepare data for DLADS model input
def prepareInput(sample):

    squareMeasuredIdxs, squareUnMeasuredIdxs = np.transpose(np.where(sample.squareMask==1)), np.transpose(np.where(sample.squareMask==0))
    
    if inputMethod == 'ReconValues':
        inputImage = copy.deepcopy(np.moveaxis(sample.squaremzReconImages, 0, -1))
        inputImage[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1], :] = 0
    elif inputMethod == 'ReconImages':
        inputImage = copy.deepcopy(np.moveaxis(sample.squaremzReconImages, 0, -1))
    elif inputMethod == 'MeasuredValues':
        inputImage = copy.deepcopy(np.moveaxis(sample.squaremzReconImages, 0, -1))
        inputImage[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1], :] = 0
    elif inputMethod == 'ReconAndMeasured':
        measuredValues = copy.deepcopy(np.moveaxis(sample.squaremzReconImages, 0, -1))
        reconValues = copy.deepcopy(np.moveaxis(sample.squaremzReconImages, 0, -1))
        reconValues[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1], :] = 0
        measuredValues[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1], :] = 0
        inputImage = np.dstack((reconValues, measuredValues))
    elif inputMethod == 'AverageReconValues':
        inputImage = copy.deepcopy(sample.squaremzAvgReconImage)
        inputImage[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]] = 0
        inputImage = np.expand_dims(inputImage, 2)
    elif inputMethod == 'AverageMeasuredValues':
        inputImage = copy.deepcopy(sample.squaremzAvgReconImage)
        inputImage[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]] = 0
        inputImage = np.expand_dims(inputImage, 2)
    elif inputMethod == 'AverageReconAndMeasured':
        reconValues = copy.deepcopy(sample.squaremzAvgReconImage)
        reconValues[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]] = 0
        measuredValues = copy.deepcopy(sample.squaremzAvgReconImage)
        measuredValues[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]] = 0
        inputImage = np.dstack((reconValues, measuredValues))
    elif inputMethod == 'features':
        inputImage = copy.deepcopy(np.moveaxis(sample.squaremzReconImages, 0, -1))
        inputImage = (np.ones((inputImage.shape))*np.max(inputImage, axis=(0, 1)))-inputImage
        inputImage[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1], :] = 0
        #inputImage[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1], :] = 0
        #inputImage = np.dstack((measuredValues, inputImage))
        #inputImage[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1], :] = 0
        #inputImage[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1], :] = 0
        #inputImage = (inputImage-np.mean(inputImage))/np.std(inputImage)
        #inputImage = np.dstack(((sample.distanceMap/np.max(sample.distanceMap))*np.max(inputImage), inputImage))
        #inputImage = scipy.ndimage.sobel(copy.deepcopy(np.moveaxis(sample.squaremzReconImages, 0, -1)))
        #inputImage = np.dstack((sample.distanceMap/np.max(sample.distanceMap), inputImage))
    elif inputMethod == 'features2':
        inputImage = copy.deepcopy(np.moveaxis(sample.squaremzReconImages, 0, -1))
        inputImage = (np.ones((inputImage.shape))*np.mean(inputImage, axis=(0, 1)))-inputImage
        inputImage[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1], :] = 0

    #Normalize/scale the data
    if normInputMethod == 'minmax': inputImage = (inputImage-np.min(inputImage))/(np.max(inputImage)-np.min(inputImage))
    elif normInputMethod == 'standardize': inputImage = (inputImage-np.mean(inputImage))/np.std(inputImage)

    return inputImage

#Determine the Expected Reduction in Distortion for uneasured points in a sample
def computeERD(sample, sampleData, model, squareUnMeasuredIdxs, squareMeasuredIdxs):
    
    if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net':
        
        #Compute ERD; those that have already been measured have 0 ERD
        ERD = np.zeros(sampleData.squareDim)
        ERD[squareUnMeasuredIdxs[:, 0], squareUnMeasuredIdxs[:, 1]] = model.predict(sample.polyFeatures)
        
        #Remove values that are less than those already scanned (0 ERD)
        ERD[np.where((ERD < 0))] = 0
    
    elif erdModel == 'DLADS':
        
        #Send input, made compatible, through trained model
        ERD = model(makeCompatible(prepareInput(sample)), training=False)[0,:,:,0].numpy()

        #Remove values at measured locations
        ERD[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]] = 0
        
    return np.nan_to_num(ERD, nan=0, posinf=0, neginf=0)

#Determine which unmeasured points of a sample should be scanned given the current E/RD
def findNewMeasurementIdxs(sample, sampleData, result, model, cValue, percToScan, oracleFlag, bestCFlag):
    
    #Resize ERD for selection
    ERD = resize(sample.ERD, tuple(sampleData.finalDim), order=0)
    
    if sampleData.scanMethod == 'random':
        np.random.shuffle(sample.unMeasuredIdxs)
        newIdxs = sample.unMeasuredIdxs[:sampleData.pointsToScan].astype(int)
    elif sampleData.scanMethod == 'pointwise':
        #If performing a groupwise scan, use reconstruction as the measurement value, until reaching target number of points to scan
        if percToScan != None:
        
            #Create a list to hold the chosen scanning locations
            newIdxs = []
            
            #Until the percToScan has been reached, substitute reconstruction values for actual measurements
            while True:
                
                #If there are no more points with ERD > 0, break from loop
                if np.sum(ERD) <= 0: break
                
                #Find next measurement location and store the chosen scanning location for later, actual measurement
                newIdx = sample.unMeasuredIdxs[np.argmax(ERD[sample.unMeasuredIdxs[:,0], sample.unMeasuredIdxs[:,1]])]
                newIdxs.append(newIdx.tolist())
                
                #Perform the measurement, using values from reconstruction 
                sample.performMeasurements(sampleData, newIdx, model, cValue, bestCFlag, oracleFlag, True)
                                
                #When enough new locations have been determined, break from loop
                if (np.sum(sample.mask)-np.sum(result.lastMask)) >= sampleData.pointsToScan: break
                
                #Reacquire the ERD and resize for selection
                ERD = resize(sample.ERD, tuple(sampleData.finalDim), order=0)
                
            #Convert to array for indexing
            newIdxs = np.asarray(newIdxs)
        else:
            #Identify the unmeasured location with the highest ERD value; return in a list to ensure it is iterable
            newIdxs = np.asarray([sample.unMeasuredIdxs[np.argmax(ERD[sample.unMeasuredIdxs[:,0], sample.unMeasuredIdxs[:,1]])].tolist()])
            
    elif sampleData.scanMethod == 'linewise':
    
        #Choose the line with maximum ERD and extract the actual indices
        if not lineRevist: ERD[np.where(np.sum(sample.mask, axis=1)>0)] = 0
        lineToScanIdx = np.nanargmax(np.nansum(ERD, axis=1))
        #lineToScanIdx = np.nanargmax(np.nanmean(ERD, axis=1))
        #lineToScanIdx = np.nanargmax(np.nanvar(ERD, axis=1))
        
        #Identify the line where the number of pointsToScan with the highest ERD have the largest mean
        #lineToScanIdx = np.argmax([np.nanmean(np.sort(ERD[lineNum])[::-1][:sampleData.pointsToScan]) for lineNum in range(0, ERD.shape[0])])

        #If points on the line should be chosen one-by-one, temporarily using reconstruction values for updating ERD
        if linePointSelection == 'single': 
            
            #Create a list to hold the chosen scanning locations
            newIdxs = []
            
            #Until the stopPerc has been reached, substitute reconstruction values for actual measurements
            while True:
                
                #If there are no points to scan on this line with ERD > 0, break from loop
                if np.sum(ERD[lineToScanIdx]) <= 0: break
                
                #Identify the next scanning location and store it for later, actual measurement
                nextIndex = np.argmax(ERD[lineToScanIdx])
                
                #Store that choice for later actual measurement
                newIdxs.append([lineToScanIdx, nextIndex])
                
                #Perform the measurement using values from reconstruction 
                sample.performMeasurements(sampleData, np.asarray(newIdxs[-1]), model, cValue, bestCFlag, oracleFlag, True)
                
                #When enough new locations have been determined, break from loop
                if len(newIdxs) >= sampleData.pointsToScan: break
                
                #Reacquire the ERD and resize for selection
                ERD = resize(sample.ERD, tuple(sampleData.finalDim), order=0)
                
            #Convert to array for indexing
            newIdxs = np.asarray(newIdxs)
            
            #Sort columns for progressive physical scanning order
            newIdxs[:,1] = np.sort(newIdxs[:,1])
            
        #If points on the line should be selected in one step/group
        elif linePointSelection == 'group':
            indexes = np.sort(np.argsort(ERD[lineToScanIdx])[::-1][:sampleData.pointsToScan])
            newIdxs = np.column_stack([np.ones(len(indexes))*lineToScanIdx, indexes]).astype(int)
        
        #==========================================
        #PARTIAL LINE BY START/END POINTS
        #==========================================
        #Choose segment to scan on line which contains at least stopPerc locations with maximal ERD
        if lineMethod == 'segLine': 
            indexes = np.arange(newIdxs[:,1][0],newIdxs[:,1][-1]+1)
            newIdxs = np.column_stack([np.ones(len(indexes))*lineToScanIdx, indexes]).astype(int)
        elif lineMethod != 'percLine': sys.exit('Error! - Unknown line method specified in configuation: ' + lineMethod)
        #==========================================
        
        #==========================================
        #SELECTION SAFEGUARD
        #==========================================
        #If there are not enough locations selected, just scan the whole remainder of the line with the greatest ERD; ensures model will reach termination
        if len(newIdxs) < int(round(0.01*sample.mask.shape[1])): 
            indexes = np.where(sample.mask[lineToScanIdx]==0)[0]
            newIdxs = np.column_stack([np.ones(len(indexes))*lineToScanIdx, indexes]).astype(int)
        #==========================================
        
    return newIdxs

def findNeighbors(measuredIdxs, unMeasuredIdxs):

    #Calculate knn
    neigh = NearestNeighbors(n_neighbors=numNeighbors).fit(measuredIdxs)
    neighborDistances, neighborIndices = neigh.kneighbors(unMeasuredIdxs)

    #Determine inverse distance weights
    unNormNeighborWeights = 1.0/np.square(neighborDistances)
    neighborWeights = unNormNeighborWeights/(np.sum(unNormNeighborWeights, axis=1))[:, np.newaxis]

    return neighborIndices, neighborWeights, neighborDistances

#Perform the reconstruction
def computeRecon(inputImage, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights):

    #Create a blank image for the reconstruction
    reconImage = np.zeros(inputImage.shape)
    
    #If 3d array, then perform all the reconstructions all at once
    if len(reconImage.shape) == 3:

        #Retreive measured values
        measuredValues = inputImage[:, squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]]
        
        #Compute reconstruction values using IDW (inverse distance weighting)
        reconImage[:, squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]] = np.sum(measuredValues[:, neighborIndices]*neighborWeights, axis=-1)

        #Combine measured values back into the reconstruction image
        reconImage[:, squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]] = measuredValues
    else:
        
        #Retreive measured values
        measuredValues = inputImage[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]]
        
        #Compute reconstruction values using IDW (inverse distance weighting)
        reconImage[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]] = np.sum(measuredValues[neighborIndices]*neighborWeights, axis=1)

        #Combine measured values back into the reconstruction image
        reconImage[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]] = measuredValues

    return reconImage

def unet(numFilters, numChannels, batchSize):
    
    inputs = Input(shape=(None,None,numChannels), batch_size=batchSize)

    conv1 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters, kernel_size=(3,3), padding='same')(inputs))
    conv2 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters, kernel_size=(3,3), padding='same')(conv1))
    down1 = MaxPool2D(pool_size=(2,2))(conv2)

    conv3 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*2, kernel_size=(3,3), padding='same')(down1))
    conv4 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*2, kernel_size=(3,3), padding='same')(conv3))
    down2 = MaxPool2D(pool_size=(2,2))(conv4)

    conv5 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*4, kernel_size=(3,3), padding='same')(down2))
    conv6 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*4, kernel_size=(3,3), padding='same')(conv5))
    down3 = MaxPool2D(pool_size=(2,2))(conv6)

    conv7 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*8, kernel_size=(3,3), padding='same')(down3))
    conv8 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*8, kernel_size=(3,3), padding='same')(conv7))
    down4 = MaxPool2D(pool_size=(2,2))(conv8)

    conv9 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*16, kernel_size=(3,3), padding='same')(down4))
    conv10 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*16, kernel_size=(3,3), padding='same')(conv9))
    
    up1 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*8, kernel_size=(2,2), padding='same')(customResize(conv10, conv8)))
    merge1 = concatenate([conv8, up1])
    conv11 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*8, kernel_size=(3,3), padding='same')(merge1))
    conv12 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*8, kernel_size=(3,3), padding='same')(conv11))

    up2 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*4, kernel_size=(2,2), padding='same')(customResize(conv12, conv6)))
    merge2 = concatenate([conv6, up2])
    conv13 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*4, kernel_size=(3,3), padding='same')(merge2))
    conv14 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*4, kernel_size=(3,3), padding='same')(conv13))

    up3 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*2, kernel_size=(2,2), padding='same')(customResize(conv14, conv4)))
    merge3 = concatenate([conv4, up3])
    conv15 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*2, kernel_size=(3,3), padding='same')(merge3))
    conv16 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters*2, kernel_size=(3,3), padding='same')(conv15))

    up4 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters, kernel_size=(2,2), padding='same')(customResize(conv16, conv2)))
    merge4 = concatenate([conv2, up4])
    conv17 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters, kernel_size=(3,3), padding='same')(merge4))
    conv18 = tf.keras.layers.LeakyReLU(alpha=0.2)(Conv2D(filters=numFilters, kernel_size=(3,3), padding='same')(conv17))

    outputs = Conv2D(filters=1, kernel_size=(1,1), activation='relu', padding='same')(conv18)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

#NEAREST_NEIGHBOR, BILINEAR
def customResize(x, y):
    x = image_ops.resize_images_v2(x, array_ops.shape(y)[1:3], method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
    nshape = tuple(y.shape.as_list())
    x.set_shape((None, nshape[1], nshape[2], None))
    return x

#Convert image into TF model compatible shapes/tensors
def makeCompatible(image):

    #Reshape for tensor transition, as needed by number of channels
    if len(image.shape) > 2: image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    else: image = image.reshape((1,image.shape[0],image.shape[1],1))

    return image

#Interpolate results to a given precision for averaging results
def percResults(results, perc_testingResults, precision):

    percents = np.linspace(min(np.hstack(perc_testingResults)), max(np.hstack(perc_testingResults)), int((max(np.hstack(perc_testingResults)) - min(np.hstack(perc_testingResults))) / precision + 1))
    newResults = [np.interp(percents, perc_testingResults[resultNum], results[resultNum]) for resultNum in range(0, len(results))]
    averageResults = np.average(newResults, axis=0)
    
    return percents, averageResults

#Quick print for titles in UI 
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

#Determine absolute difference between two arrays
def computeDifference(array1, array2):
    return abs(array1-array2)

class DiceLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return 1 - numerator / denominator

class JaccardLoss():
    def call(self, y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + 100) / (sum_ - intersection + 100)
        return (1 - jac) * 100

