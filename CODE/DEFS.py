#==================================================================
#METHOD AND CLASS DEFINITIONS
#==================================================================

#Object for initializing and storing sample metadata
class SampleData:
    def __init__(self, sampleFolder, initialPercToScan, stopPerc, scanMethod, lineRevist, postFlag, simulationFlag):
        
        #Save options as internal variables
        self.scanMethod = scanMethod
        self.initialPercToScan = initialPercToScan
        self.stopPerc = stopPerc
        self.lineRevist = lineRevist
        self.postFlag = postFlag
        self.simulationFlag = simulationFlag
        self.lineExt = None
        self.mask = None
        self.unorderedNames = False
        self.missingLines = np.asarray([])
        self.chanValues = []
        
        #Set global variables to indicate that OOM error states have not yet occurred; limited handle for ERD inferencing limitations
        self.OOM_multipleChannels, self.OOM_singleChannel = False, False
        
        #Store location of MSI data and sample name
        self.sampleFolder = sampleFolder
        self.name = os.path.basename(sampleFolder)
        if impModel: self.name = impSampleName
        
        #Note which files have already been read
        self.readScanFiles = []
        self.readLines = []
        
        #Storage location for matching sequentially generated indexes with physical line numbers
        self.physicalLineNums = {}
        
        #Read in data from sampleInfo.txt, starting with sample type, otherwise assume image data
        lineIndex = 0
        try: 
            sampleInfo = open(sampleFolder+os.path.sep+'sampleInfo.txt').readlines()
            self.sampleType = sampleInfo[lineIndex].rstrip()
            lineIndex += 1
        except: 
            self.sampleType = 'IMAGE'

        if self.sampleType == 'DESI':

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
            
            #Read window tolerance (ppm)
            self.ppm = float(sampleInfo[lineIndex].rstrip())*1e-6
            lineIndex += 1

            #Number of decimals for mz precision
            self.mzResOrder = int(sampleInfo[lineIndex].rstrip())
            lineIndex += 1

            #Check if files are numbered sequentially, or according to physical position
            self.unorderedNames = int(sampleInfo[lineIndex].rstrip()) == 1
            lineIndex += 1
            
        elif self.sampleType == 'MALDI':
        
            #Read the sample width (number of columns)
            self.sampleWidth = float(sampleInfo[lineIndex].rstrip())
            lineIndex += 1

            #Read the sample height (number of rows)
            self.sampleHeight = float(sampleInfo[lineIndex].rstrip())
            lineIndex += 1
            
            #Read window tolerance (ppm)
            self.ppm = float(sampleInfo[lineIndex].rstrip())*1e-6
            lineIndex += 1

            #Number of decimals for mz precision
            self.mzResOrder = int(sampleInfo[lineIndex].rstrip())
            lineIndex += 1

        elif self.sampleType == 'IMAGE':
        
            #Read the sample width (number of columns)
            self.sampleWidth = float(sampleInfo[lineIndex].rstrip())
            lineIndex += 1

            #Read the sample height (number of rows)
            self.sampleHeight = float(sampleInfo[lineIndex].rstrip())
            lineIndex += 1
            
            #Read number of sample channels
            self.numChannels = int(sampleInfo[lineIndex].rstrip())
            lineIndex += 1
            
        else: 
            sys.exit('Error - Unknown sample type: ' + self.sampleType + ' specified in sampleInfo.txt for : ' + sampleFolder)

        #MSI specific
        if self.sampleType == 'MALDI' or self.sampleType == 'DESI':
        
            #Process information as needed
            self.ignoreMissingLines = self.simulationFlag
            self.ppmPos, self.ppmNeg = 1+self.ppm, 1-self.ppm

            #Get mz ranges to use for visualizations
            try: self.chanValues = np.loadtxt(self.sampleFolder+os.path.sep+'channels.csv', delimiter=',')
            except: self.chanValues = np.loadtxt('channels.csv', delimiter=',')
            self.mzRanges = np.round(np.column_stack((self.chanValues*self.ppmNeg, self.chanValues*self.ppmPos)), self.mzResOrder)
            self.numChannels = len(self.chanValues)
            
        #DESI determines dimensionality from scan and acquisition rate, where other types are explicitly specified
        if self.sampleType == 'DESI':

            #If the filenames were sequentially generated, then load location mapping dictionary
            if self.unorderedNames and not impModel:
                for item in np.loadtxt(sampleFolder+os.path.sep+'physicalLineNums.csv', 'int', delimiter=','): self.physicalLineNums[item[0]] = item[1]
                
            #Store final dimensions for physical domain, determining the number of columns for row-alignment interpolations
            self.finalDim = [self.numLines, int(round(((self.sampleWidth*1e3)/self.scanRate)*self.acqRate))]
            
        else:
            self.finalDim = [int(self.sampleHeight), int(self.sampleWidth)]
            self.squareDim = self.finalDim
        
        #If this is a DESI simulation sample, then get filetype extension and check for missing lines if applicable
        if self.sampleType == 'DESI' and self.simulationFlag:
            extensions = list(map(lambda x:x, np.unique([filename.split('.')[-1] for filename in natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'), reverse=False)])))
            if 'd' in extensions: self.lineExt = '.d'
            elif 'D' in extensions: self.lineExt = '.D'
            elif 'raw' in extensions: self.lineExt = '.raw'
            elif 'RAW' in extensions: self.lineExt = '.RAW'
            elif 'png' in extensions: self.lineExt = '.png'
            elif 'jpg' in extensions: self.lineExt = '.jpg'
            elif 'tiff' in extensions: self.lineExt = '.tiff'
            elif 'imzML' in extensions: self.lineExt = '.imzML'
            else: sys.exit('Error! - Either no files are present, or an unknown filetype being used for sample: ' + self.name)
            scanFiles = natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'+self.lineExt), reverse=False)
            if self.sampleType == 'DESI' and self.ignoreMissingLines:
                self.missingLines = np.asarray(list(set(np.arange(1, self.finalDim[0]).tolist()) - set([int(scanFileName.split('line-')[1].split('.')[0].lstrip('0')) for scanFileName in scanFiles])))-1
                self.finalDim[0] -= len(self.missingLines)
        
        #For DESI samples, determine image dimensions that will produce square pixels (consistent vertical/horizontal resolution) and new times to use as a common grid
        if self.sampleType == 'DESI':
            if(self.finalDim[1]/self.sampleWidth) > (self.finalDim[0]/self.sampleHeight): self.squareDim = [int(round((self.finalDim[1]*self.sampleHeight)/self.sampleWidth)), self.finalDim[1]]
            elif (self.finalDim[1]/self.sampleWidth) < (self.finalDim[0]/self.sampleHeight): self.squareDim = [self.finalDim[0], int(round((self.finalDim[0]*self.sampleWidth)/self.sampleHeight))]
            else: self.squareDim = self.finalDim
            self.newTimes = np.linspace(0, ((self.sampleWidth*1e3)/self.scanRate)/60, self.finalDim[1])
            
        #Establish total sample area, setup zeroed channel and sum images for MSI data
        self.area = int(round(self.finalDim[0]*self.finalDim[1]))
        self.chanImages = np.zeros((self.numChannels, self.finalDim[0], self.finalDim[1]))
        self.sumImage = np.zeros((self.finalDim))
    
        #If not just post-processing, setup initial sets
        if not self.postFlag: self.generateInitialSets(self.scanMethod)
        else:
            try: self.mask = np.loadtxt(self.sampleFolder+os.path.sep+'measuredMask.csv', 'int', delimiter=',') 
            except: sys.exit('Error - Unable to load measurement mask for sample: ' + self.name)
            
        #If a simulation or post-processing, then just read all the data outright
        if self.simulationFlag or self.postFlag: self.readScanData()
        
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
            lineIndexes = [int(round((self.finalDim[0]-1)*startLinePosition)) for startLinePosition in startLinePositions]
            
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
    
    def readScanData(self):
        
        #Get the MSI file extension automatically if it isn't already known
        if self.lineExt == None:
            extensions = list(map(lambda x:x, np.unique([filename.split('.')[-1] for filename in natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'), reverse=False)])))
            if 'd' in extensions: self.lineExt = '.d'
            elif 'D' in extensions: self.lineExt = '.D'
            elif 'raw' in extensions: self.lineExt = '.raw'
            elif 'RAW' in extensions: self.lineExt = '.RAW'
            elif 'png' in extensions: self.lineExt = '.png'
            elif 'jpg' in extensions: self.lineExt = '.jpg'
            elif 'tiff' in extensions: self.lineExt = '.tiff'
            elif 'imzML' in extensions: self.lineExt = '.imzML'
            else: sys.exit('Error! - Either no files are present, or an unknown filetype being used for sample: ' + self.name)
        
        #Obtain and sort the available line files pertaining to the current scan
        scanFiles = natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'+self.lineExt), reverse=False)
            
        #If DESI, each scanfile corresponds to a line of data
        if self.sampleType == 'DESI':
            
            #Identify which files have not yet been scanned, if line revisiting is disabled
            if self.lineRevist == False: scanFiles = natsort.natsorted(list(set(scanFiles)-set(self.readScanFiles)), reverse=False)
            
            #Read in specified files
            if parallelization:
                dataResults = ray.get([scanData_DESI_parhelper.remote(self, scanFileName) for scanFileName in scanFiles])
                for result in dataResults: 
                    if result != None:
                        for chanNum in range(0, len(self.chanImages)): self.chanImages[chanNum, result[0], :] = result[1][chanNum]
                        self.sumImage[result[0]] = result[2]
                        self.readLines.append(result[0])
                        self.readScanFiles.append(result[3]) 
            else:
                for scanFileName in scanFiles:

                    #Establish file pointer and line number (1 indexed) for the specific scan; flag indicates 'good'/'bad' data file (primarily checking for files without data)
                    readErrorFlag = False
                    try: data = mzFile(scanFileName)
                    except: readErrorFlag = True
                
                    #If the data file is 'good' then continue processing
                    if not readErrorFlag:
                        
                        #Extract line number from the filename, removing leading zeros, subtract 1 for zero indexing
                        fileNum = int(scanFileName.split('line-')[1].split('.')[0].lstrip('0'))-1
                        
                        #If the file numbers are not the physical row numbers, then obtain correct number from stored LUT
                        if self.unorderedNames: 
                            try: lineNum = self.physicalLineNums[fileNum+1]
                            except: 
                                print('Warning - Attempt to find the physical line number for the file: ' + scanFileName + ' has failed; the file will therefore be ignored this iteration.')
                                readErrorFlag = True
                        else: lineNum = fileNum
                        #if (impModel or postModel) and self.unorderedNames: lineNum, columnNum = self.physicalLineNums[fileNum+1], self.physicalColumnNums[fileNum+1]
                        
                    #If the data file is still 'good' then continue processing
                    if not readErrorFlag:
                    
                        #Add file name to those already scanned
                        self.readScanFiles.append(scanFileName)
                        
                        #Record that the line number specified has been read previously
                        self.readLines.append(lineNum)
                        
                        #If ignoring missing lines, then determine the offset for correct indexing
                        if self.ignoreMissingLines and len(self.missingLines) > 0: lineNum -= int(np.sum(lineNum > self.missingLines))
                        
                        #Obtain the total ion chromatogram and extract original times
                        sumImageData = np.asarray(data.xic(data.time_range()[0], data.time_range()[1]))
                        origTimes, sumImageData = sumImageData[:,0], sumImageData[:,1]
                        
                        #Offset the original measured times, such that the first position's time equals 0
                        origTimes -= np.min(origTimes)
                        
                        #If the data is being sparesly acquired, then the listed times in the file need to be shifted; convert np.float to float for method compatability
                        #For impOffset compatability with percent-linewise or pointwise: np.argwhere(mask[lineNum]==1).min() must be improved...
                        #Where physicalLineNums is updated, add another dictionary of physicalColumnNums mapping number in filename to time offset
                        #Remember to make modifications to corresponding parhelper method
                        #if (impModel or postModel) and impOffset and scanMethod == 'linewise' and lineMethod == 'segLine': origTimes += (columnNum/self.finalDim[1])*(((self.sampleWidth*1e3)/self.scanRate)/60)
                        if (impModel or postModel) and impOffset and scanMethod == 'linewise' and (lineMethod == 'segLine' or lineMethod == 'fullLine'): origTimes += (np.argwhere(self.mask[lineNum]==1).min()/self.finalDim[1])*(((self.sampleWidth*1e3)/self.scanRate)/60)
                        elif (impModel or postModel) and impOffset: sys.exit('Error - Using implementation or post-process modes with an offset but not segmented-linewise operation is not currently a supported configuration.')
                        for chanNum in range(0, len(self.chanValues)): self.chanImages[chanNum, lineNum, :] = np.interp(self.newTimes, origTimes, np.nan_to_num(np.asarray(data.xic(data.time_range()[0], data.time_range()[1], float(self.mzRanges[chanNum][0]), float(self.mzRanges[chanNum][1])))[:,1], nan=0, posinf=0, neginf=0), left=0, right=0)
                        
                        #Interpolate sumImage (TIC) to final new times
                        self.sumImage[lineNum] = np.interp(self.newTimes, origTimes, np.nan_to_num(sumImageData, nan=0, posinf=0, neginf=0), left=0, right=0)
        
            #Resize for square dimensions
            self.squareChanImages = np.moveaxis(resize(np.moveaxis(self.chanImages, 0, -1), tuple(self.squareDim), order=0), -1, 0)
        
        #If MALDI, then there is only a single file with all of the spectral data
        elif self.sampleType == 'MALDI':
        
            #Establish file pointer for the scan
            try: data = ImzMLParser(scanFiles[0])
            except: sys.exit('Error - Unable to read file' + scanFiles[chanNum])
            
            #Read all data available if a simulation; not yet implemented: for actual implementation should just read new positions ref. new idxs
            #No parallel implementation yet
            for i, (x, y, z) in enumerate(data.coordinates):
                mzs, ints = data.getspectrum(i)
                for chanNum in range(0, len(self.chanValues)): 
                    min_i, max_i = pyimzml.ImzMLParser._bisect_spectrum(mzs, self.chanValues[chanNum], self.chanValues[chanNum]*self.ppm)
                    self.chanImages[chanNum, y-1, x-1] = sum(ints[min_i:max_i+1])
                self.sumImage[y-1, x-1] = np.sum(ints)
            
        #If IMAGE, each scanfile corresponds to a channel, read in each and sum of all data, augmenting list of channel labels
        #No parallel implementation yet
        elif self.sampleType == 'IMAGE':
            for chanNum in range(0, len(scanFiles)):
                self.chanValues.append(scanFiles[chanNum].split('chan-')[1].split('.')[0])
                try: self.chanImages[chanNum] = cv2.imread(scanFiles[chanNum], 0)
                except: sys.exit('Error - Unable to read file' + scanFiles[chanNum])
            self.sumImage = np.sum(np.atleast_3d(self.chanImages), axis=0)
        
        #If not DESI, then the square dimensions are equal to the original
        if self.sampleType != 'DESI': self.squareChanImages = self.chanImages
            
        #Find the maximum value in each channel image for easy referencing
        self.chanImagesMax = np.max(self.chanImages, axis=(1,2))

#Relevant sample data at each time step; static information should be held in corresponding SampleData object
class Sample:
    def __init__(self, sampleData):
        
        #Initialize variables that are expected to exist
        self.mask = np.zeros((sampleData.finalDim))
        self.squareMask = np.zeros((sampleData.squareDim))
        self.squareRD = np.zeros((sampleData.squareDim))
        self.squareRDs = np.zeros((sampleData.numChannels, sampleData.squareDim[0], sampleData.squareDim[1]))
        self.squareERD = np.zeros((sampleData.squareDim))
        self.squareERDs = np.zeros((sampleData.numChannels, sampleData.squareDim[0], sampleData.squareDim[1]))
        self.percMeasured = 0
        self.iteration = 0
        
        #If post-processing, link to the sampled mask
        if sampleData.postFlag: self.mask = sampleData.mask
        
    def performMeasurements(self, sampleData, result, newIdxs, model, cValue, bestCFlag, oracleFlag, datagenFlag, fromRecon):

        #Ensure newIdxs are indexible in 2 dimensions and update mask; post-processing will send empty set
        if not sampleData.postFlag:
            newIdxs = np.atleast_2d(newIdxs)
            self.mask[newIdxs[:,0], newIdxs[:,1]] = 1
        
        #Update which positions have not yet been measured
        self.unMeasuredIdxs = np.transpose(np.where(self.mask==0))
        
        #If not taking values from a reconstruction, get from equipment or ground-truth; else get from the reconstruction
        if not fromRecon:
            #If not simulation, then read from equipment; update sampleData mask and mask images by what should have been scanned
            if not sampleData.simulationFlag and not sampleData.postFlag:
                print('Writing UNLOCK')
                with open(dir_ImpDataFinal + 'UNLOCK', 'w') as filehandle: _ = [filehandle.writelines(str(tuple([pos[0]+1, (pos[1]*sampleData.scanRate)/sampleData.acqRate]))+'\n') for pos in newIdxs.tolist()]
                if sampleData.unorderedNames and impModel and scanMethod == 'linewise': sampleData.physicalLineNums[len(sampleData.physicalLineNums.keys())+1] = int(newIdxs[0][0])
                sampleData.mask = self.mask
                equipWait()
                sampleData.readScanData()
            self.chanImages = copy.deepcopy(sampleData.chanImages)*self.mask
            self.sumImage = copy.deepcopy(sampleData.sumImage)*self.mask
        else:
            self.chanImages[:, newIdxs[:,0], newIdxs[:,1]] = self.chanReconImages[:, newIdxs[:,0], newIdxs[:,1]]
            self.sumImage[newIdxs[:,0], newIdxs[:,1]] = self.sumImageReconImage[newIdxs[:,0], newIdxs[:,1]]
        
        #Update percentage pixels measured; only when not fromRecon
        self.percMeasured = (np.sum(self.mask)/sampleData.area)*100
        
        #For DESI, resize the mask to enforce square pixels and extract measured and unmeasured locations, otherwise no resizing is needed
        if sampleData.sampleType == 'DESI': self.squareMask = resize(self.mask, tuple(sampleData.squareDim), order=0)
        else: self.squareMask = self.mask
            
        squareMeasuredIdxs, squareUnMeasuredIdxs = np.transpose(np.where(self.squareMask==1)), np.transpose(np.where(self.squareMask==0))

        #Determine neighbor information for unmeasured locations
        if len(squareUnMeasuredIdxs) > 0: neighborIndices, neighborWeights, neighborDistances = findNeighbors(squareMeasuredIdxs, squareUnMeasuredIdxs)
        else: neighborIndices, neighborWeights, neighborDistances = [], [], []

        #Compute the reconstructions (using square pixels) if new data is acquired
        if not fromRecon:
        
            #Update the iteration counter
            self.iteration += 1
            
            #Compute reconstructions and average for visualization, if DESI then resize to physical dimensions 
            if sampleData.sampleType == 'DESI':
                self.squareSumImageReconImage = computeRecon(resize(self.sumImage, tuple(sampleData.squareDim), order=0), squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
                self.squareChanReconImages = computeRecon(np.moveaxis(resize(np.moveaxis(self.chanImages, 0, -1), tuple(sampleData.squareDim), order=0), -1, 0), squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
                self.sumImageReconImage = resize(self.squareSumImageReconImage, tuple(sampleData.finalDim), order=0)
                self.chanReconImages = np.moveaxis(resize(np.moveaxis(self.squareChanReconImages , 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)
            else:
                self.squareSumImageReconImage = computeRecon(self.sumImage, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
                self.squareChanReconImages = computeRecon(self.chanImages, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
                self.sumImageReconImage = self.squareSumImageReconImage
                self.chanReconImages = self.squareChanReconImages
                
            #Compute feature information for SLADS models; not needed for DLADS
            if (datagenFlag or ((erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net')) and not bestCFlag) and len(squareUnMeasuredIdxs) > 0: 
                t0 = time.time()
                self.polyFeatures = [computePolyFeatures(sampleData, squareChanReconImage, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights, neighborDistances) for squareChanReconImage in self.squareChanReconImages]
                t1 = time.time()
                polyComputeTime = t1-t0
            else: polyComputeTime = 0
        else: polyComputeTime = 0
        
        #Compute RD/ERD; if every location has been scanned all positions are zero
        if len(squareUnMeasuredIdxs) == 0:
            if oracleFlag or bestCFlag: 
                self.RD = np.zeros(sampleData.finalDim)
                self.squareRD = np.zeros(sampleData.squareDim)
                self.squareRDs = np.zeros((sampleData.numChannels, sampleData.squareDim[0], sampleData.squareDim[1]))
                self.squareRDValues = self.squareRDs[:, squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]]
                self.squareERD = self.squareRD
            else: self.squareERD = np.zeros(sampleData.squareDim)
        elif oracleFlag or bestCFlag:

            #If this is a full measurement step, compute the RDPP
            if not fromRecon: self.RDPPs = abs(sampleData.squareChanImages-self.squareChanReconImages)
            
            #Compute the RD and use it in place of an ERD; only save times if they are fully computed, not just being updated
            t0 = time.time()
            computeRD(self, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborDistances, cValue, bestCFlag, datagenFlag, fromRecon, result.liveOutputFlag, result.impModel)
            t1 = time.time()
            if not fromRecon: result.computeRDTimes.append(t1-t0)
            self.squareRDValues = self.squareRDs[:, squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]]
            if sampleData.sampleType == 'DESI': self.RD = resize(self.squareRD, tuple(sampleData.finalDim), order=0)*(1-self.mask)
            else: self.RD = self.squareRD*(1-self.mask)
            self.squareERD = self.RD
        else: 
            t0 = time.time()
            computeERD(self, sampleData, model, squareUnMeasuredIdxs, squareMeasuredIdxs)
            t1 = time.time()
            result.computeERDTimes.append((t1-t0)+polyComputeTime)
            
        #Process ERD for next measurement(s) selection (resize for DESI)
        if sampleData.sampleType == 'DESI':
            self.ERD = resize(self.squareERD, tuple(sampleData.finalDim), order=0)*(1-self.mask)
            self.ERDs = np.moveaxis(resize(np.moveaxis(self.squareERDs , 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)*(1-self.mask)
        else:
            self.ERD = self.squareERD
            self.ERDs = self.squareERDs
        
        #For processed ERD, set measured locations to 0, ensure >= values, rescale for Otsu, and prevent line revisitation as specified)
        self.physicalERD = copy.deepcopy(self.ERD)
        self.physicalERD[self.physicalERD<0] = 0
        if np.max(self.physicalERD) != 0: self.physicalERD = ((self.physicalERD-np.min(self.physicalERD))/(np.max(self.physicalERD)-np.min(self.physicalERD)))*100
        if sampleData.scanMethod == 'linewise' and not sampleData.lineRevist: self.physicalERD[np.where(np.sum(self.mask, axis=1)>0)] = 0

#Sample scanning progress and final results processing
class Result:
    def __init__(self, sampleData, liveOutputFlag, dir_Results, bestCFlag, datagenFlag, cValue, impModel):
        self.startTime = time.time()
        self.finalTime = time.time()
        self.sampleData = sampleData
        self.cValue = cValue
        self.impModel = impModel
        self.bestCFlag = bestCFlag
        self.datagenFlag = datagenFlag
        self.samples = []
        self.cSelectionList = []
        self.lastMask = None
        self.percsMeasured = []
        self.liveOutputFlag = liveOutputFlag
        self.dir_Results = dir_Results
        self.computeRDTimes, self.computeERDTimes = [], []
        
        if dir_Results != None:

            #Setup/clean base sample directory
            self.dir_sampleResults = self.dir_Results + self.sampleData.name + os.path.sep
            if os.path.exists(self.dir_sampleResults): shutil.rmtree(self.dir_sampleResults)
            os.makedirs(self.dir_sampleResults)
            
            #Prepare subdirectories; for frames and videos of channel progressions
            self.dir_chanProgression = self.dir_sampleResults + 'Channels' + os.path.sep
            os.makedirs(self.dir_chanProgression)
            self.dir_chanProgressions = [self.dir_chanProgression + str(self.sampleData.chanValues[chanNum]) + os.path.sep for chanNum in range(0, len(self.sampleData.chanValues))]
            for dir_chanProgressionsub in self.dir_chanProgressions: 
                try: os.makedirs(dir_chanProgressionsub)
                except: print('Folder already exists')
            self.dir_progression = self.dir_sampleResults + 'Progression' + os.path.sep
            os.makedirs(self.dir_progression)
            if not self.sampleData.postFlag:
                self.dir_videos= self.dir_sampleResults + 'Videos' + os.path.sep
                os.makedirs(self.dir_videos)
        
    def update(self, sample):
    
        #Update measurement mask and percentage of FOV measured at this step 
        self.lastMask = copy.deepcopy(sample.mask)
        self.percsMeasured.append(copy.deepcopy(sample.percMeasured))
        
        #If outputs should be produced at every update step, then do so, determining related metrics as needed
        if self.liveOutputFlag: 
            if self.sampleData.simulationFlag: self.extractSimulationData(sample)
            visualize_serial(sample, self.sampleData, self.dir_progression, self.dir_chanProgressions)
        
        #If evaluating a c parameter, then find the PSNR of the current reconstructions, otherwise save a copy of the measurement step for later evaluation
        if self.bestCFlag: 
            self.cSelectionList.append(np.mean([compare_psnr(self.sampleData.chanImages[index], sample.chanReconImages[index], data_range=self.sampleData.chanImagesMax[index]) for index in range(0, len(self.sampleData.chanImages))]))
        else:
            self.samples.append(copy.deepcopy(sample))
            self.finalTime = time.time()-self.startTime
            
        #If there is a results directory
        if self.dir_Results != None: 
        
            #If the filenames were unordered, then save the mapping from filename to physical row
            if self.sampleData.unorderedNames: np.savetxt(self.dir_sampleResults+'physicalLineNums.csv', np.asarray(list(self.sampleData.physicalLineNums.items())), delimiter=',', fmt='%d')
        
            #Save a copy of the final measurement mask
            np.savetxt(self.dir_sampleResults+'measuredMask.csv', self.lastMask, delimiter=',', fmt='%d')
    
    #For a given measurement step find PSNR/SSIM of reconstructions, compute the RD, find PSNR of ERD
    def extractSimulationData(self, sample):
    
        #Extract measured and unmeasured locations for the square pixel mask
        squareMeasuredIdxs, squareUnMeasuredIdxs = np.transpose(np.where(sample.squareMask==1)), np.transpose(np.where(sample.squareMask==0))
        
        #Determine neighbor information for unmeasured locations
        if len(squareUnMeasuredIdxs) > 0: neighborIndices, neighborWeights, neighborDistances = findNeighbors(squareMeasuredIdxs, squareUnMeasuredIdxs)
        else: neighborIndices, neighborWeights, neighborDistances = [], [], []
        
        sample.chanImagesPSNRList = [compare_psnr(self.sampleData.chanImages[index], sample.chanReconImages[index], data_range=self.sampleData.chanImagesMax[index]) for index in range(0, len(self.sampleData.chanImages))]
        sample.sumImagePSNR = compare_psnr(self.sampleData.sumImage, sample.sumImageReconImage, data_range=np.max(self.sampleData.sumImage))
        sample.chanImagesSSIMList = [compare_ssim(self.sampleData.chanImages[index], sample.chanReconImages[index], data_range=self.sampleData.chanImagesMax[index]) for index in range(0, len(self.sampleData.chanImages))]
        sample.sumImageSSIM = compare_ssim(self.sampleData.sumImage, sample.sumImageReconImage, data_range=np.max(self.sampleData.sumImage))

        #Compute RD; if every location has been scanned all positions are zero
        if len(squareUnMeasuredIdxs) == 0: 
            sample.squareRD = np.zeros(self.sampleData.squareDim)
            sample.RD = np.zeros(self.sampleData.finalDim)
        else: 
            sample.RDPPs = abs(self.sampleData.squareChanImages-sample.squareChanReconImages)
            computeRD(sample, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborDistances, self.cValue, self.bestCFlag, self.datagenFlag, False, self.liveOutputFlag, self.impModel)

        #Determine SSIM/PSNR between averaged RD and ERD
        maxRangeValue = np.max([sample.squareRD, sample.squareERD])
        sample.ERDPSNR = compare_psnr(sample.squareRD, sample.squareERD, data_range=maxRangeValue)
        sample.ERDSSIM = compare_ssim(sample.squareRD, sample.squareERD, data_range=maxRangeValue)
        
        #Resize RD(s) for final visualization if DESI
        if self.sampleData.sampleType == 'DESI':
            sample.RD = resize(sample.squareRD, tuple(self.sampleData.finalDim), order=0)*(1-sample.mask)
            sample.RDs = np.moveaxis(resize(np.moveaxis(sample.squareRDs , 0, -1), tuple(self.sampleData.finalDim), order=0), -1, 0)*(1-sample.mask)
        else:
            sample.RD = sample.squareRD
            sample.RDs = sample.squareRDs

    #Generate visualiations/metrics as needed at the end of scanning
    def complete(self):
        
        #Make sure samples is writable
        self.samples = copy.deepcopy(self.samples)
        
        #If the filenames were unordered, then save the mapping from filename to physical row
        if self.sampleData.unorderedNames: np.savetxt(self.dir_sampleResults+'physicalLineNums.csv', np.asarray(list(self.sampleData.physicalLineNums.items())), delimiter=',', fmt='%d')
        
        #Save a copy of the final measurement mask
        np.savetxt(self.dir_sampleResults+'measuredMask.csv', self.samples[-1].mask, delimiter=',', fmt='%d')
        
        #If this is a simulation, then can compare against ground-truth information
        if self.sampleData.simulationFlag:
            
            #If not done during acquisiton, then for each of the measurement steps find PSNR/SSIM of reconstructions, compute the RD, find PSNR of ERD
            if not self.liveOutputFlag: _ = [self.extractSimulationData(sample) for sample in tqdm(self.samples, desc='RD/Metrics Extraction', leave=False, ascii=True)]

            #Summarize scores for testing printout
            self.chanAvgPSNRList = [np.mean(sample.chanImagesPSNRList) for sample in self.samples]
            self.sumImagePSNRList = [sample.sumImagePSNR for sample in self.samples]
            self.ERDPSNRList = [sample.ERDPSNR for sample in self.samples]
            self.chanAvgSSIMList = [np.mean(sample.chanImagesSSIMList) for sample in self.samples]
            self.sumImageSSIMList = [sample.sumImageSSIM for sample in self.samples]
            self.ERDSSIMList = [sample.ERDSSIM for sample in self.samples]
        
        #Generate visualizations if they are not created during operation
        if not self.liveOutputFlag:
            if parallelization:
                samples_id, sampleData_id = ray.put(self.samples), ray.put(self.sampleData)
                _ = ray.get([visualize_parhelper.remote(samples_id, sampleData_id, self.dir_progression, self.dir_chanProgressions, indexes) for indexes in np.array_split(np.arange(0, len(self.samples)), numberCPUS)])
            else: 
                _ = [visualize_serial(sample, self.sampleData, self.dir_progression, self.dir_chanProgressions) for sample in tqdm(self.samples, desc='Steps', leave=False, ascii=True)]
                
        #Combine images into animations
        if not self.sampleData.postFlag:
        
            #Animate each channel
            for chanNum in tqdm(range(0, len(self.sampleData.chanValues)), desc='Channel Videos', leave = False, ascii=True): 
                dataFileNames = natsort.natsorted(glob.glob(self.dir_chanProgressions[chanNum] + 'progression_*.png'))
                height, width, layers = cv2.imread(dataFileNames[0]).shape
                animation = cv2.VideoWriter(self.dir_videos + str(self.sampleData.chanValues[chanNum]) + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (width, height))
                for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                animation.release()
                animation = None
                
            #Animate progression
            dataFileNames = natsort.natsorted(glob.glob(self.dir_progression + 'progression_*.png'))
            height, width, layers = cv2.imread(dataFileNames[0]).shape
            animation = cv2.VideoWriter(self.dir_videos + 'progression.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (width, height))
            for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
            animation.release()
            animation = None

#Visualize single sample progression step
def visualize_serial(sample, sampleData, dir_progression, dir_chanProgressions):

    #Turn percent measured into a string
    percMeasured = "{:.2f}".format(sample.percMeasured)
    
    #Turn metrics into strings
    if sampleData.simulationFlag: 
        sumImagePSNR = "{:.2f}".format(sample.sumImagePSNR)
        sumImageSSIM = "{:.2f}".format(sample.sumImageSSIM)
        erdPSNR = "{:.2f}".format(sample.ERDPSNR)
        erdSSIM = "{:.2f}".format(sample.ERDSSIM)
        chanImageAvgPSNR = "{:.2f}".format(np.mean(sample.chanImagesPSNRList))
        chanImageAvgSSIM = "{:.2f}".format(np.mean(sample.chanImagesSSIMList))

    #For each of the channels, generate visuals
    for chanNum in range(0, sampleData.numChannels):
        
        #Find minimum and maximum channel values for colorbars
        chanMinValue, chanMaxValue = np.min(sampleData.chanImages[chanNum]), np.max(sampleData.chanImages[chanNum])
        
        #Turn metrics into strings
        chanLabel = str(sampleData.chanValues[chanNum])
        if sampleData.simulationFlag: 
            chanImagesPSNR = "{:.2f}".format(sample.chanImagesPSNRList[chanNum])
            chanImagesSSIM = "{:.2f}".format(sample.chanImagesSSIMList[chanNum])
        if sampleData.simulationFlag: f = plt.figure(figsize=(20,10))
        else: f = plt.figure(figsize=(20,5.3865))
        
        if sampleData.simulationFlag: plt.suptitle(r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Channel:\ }$" + chanLabel + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured + '\n' + r"$\bf{PSNR - Average\ Channel\ Recon:\ }$" + chanImageAvgPSNR + r"$\bf{\ \ Channel\ Recon:\ }$" + chanImagesPSNR+ '\n' + r"$\bf{SSIM - Average\ Channel\ Recon:\ }$" + chanImageAvgSSIM + r"$\bf{\ \ Channel\ Recon:\ }$" + chanImagesSSIM)
        else: plt.suptitle(r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Channel:\ }$" + chanLabel + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured)

        if sampleData.simulationFlag: 
            ax = plt.subplot2grid(shape=(2,3), loc=(0,0))
            im = ax.imshow(sampleData.chanImages[chanNum], cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue)
            ax.set_title('Ground-Truth')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

        if sampleData.simulationFlag: ax = plt.subplot2grid((2,3), (0,1))
        else: ax = plt.subplot2grid((1,3), (0,0))
        im = ax.imshow(sample.chanReconImages[chanNum], cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue)
        ax.set_title('Reconstruction')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

        if sampleData.simulationFlag: 
            ax = plt.subplot2grid((2,3), (0,2))
            im = ax.imshow(abs(sampleData.chanImages[chanNum]-sample.chanReconImages[chanNum]), cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue)
            ax.set_title('Absolute Difference')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

        if sampleData.simulationFlag: ax = plt.subplot2grid((2,3), (1,0))
        else: ax = plt.subplot2grid((1,3), (0,1))
        im = ax.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Measurement Mask')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        if sampleData.simulationFlag: ax = plt.subplot2grid((2,3), (1,1))
        else: ax = plt.subplot2grid((1,3), (0,2))
        im = ax.imshow(sample.ERDs[chanNum], cmap='viridis', vmin=0, aspect='auto')
        ax.set_title('ERD')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

        if sampleData.simulationFlag: 
            ax = plt.subplot2grid((2,3), (1,2))
            im = ax.imshow(sample.RDs[chanNum], cmap='viridis', vmin=0, aspect='auto')
            ax.set_title('RD')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        #Save
        f.tight_layout()
        f.subplots_adjust(top = 0.85)
        saveLocation = dir_chanProgressions[chanNum] + 'progression_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) +'.png'
        plt.savefig(saveLocation)
        plt.close()

        #Do borderless saves for each channel image here; mask will be the same as produced in the progression output
        saveLocation = dir_chanProgressions[chanNum] + 'erd_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.ERDs[chanNum], saveLocation, cmap='viridis', vmin=0)
        
        if sampleData.simulationFlag:
            saveLocation = dir_chanProgressions[chanNum] + 'rd_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
            borderlessPlot(sample.RDs[chanNum], saveLocation, cmap='viridis', vmin=0)
            
            saveLocation = dir_chanProgressions[chanNum] + 'groundTruth_channel_' + chanLabel + '.png'
            borderlessPlot(sample.chanImages[chanNum], saveLocation, cmap='hot', vmin=chanMinValue, vmax=chanMaxValue)

        saveLocation = dir_chanProgressions[chanNum] + 'reconstruction_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.chanReconImages[chanNum], saveLocation, cmap='hot', vmin=chanMinValue, vmax=chanMaxValue)
        
        saveLocation = dir_chanProgressions[chanNum] + 'measured_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.chanImages[chanNum], saveLocation, cmap='hot', vmin=chanMinValue, vmax=chanMaxValue)
        
    #For the overall progression , generate visual
    sumImageMinValue, sumImageMaxValue = np.min(sampleData.sumImage), np.max(sampleData.sumImage)
    if sampleData.simulationFlag: f = plt.figure(figsize=(20,10))
    else: f = plt.figure(figsize=(20,5.3865))

    if sampleData.simulationFlag: plt.suptitle(r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured + '\n' + r"$\bf{PSNR\ -\ Sum\ Image\ Recon: }$" + sumImagePSNR + r"$\bf{\ \ Average \ Channel\ Recon:\ }$" + chanImageAvgPSNR + r"$\bf{\ \ ERD:\ }$" + erdPSNR + '\n' + r"$\bf{SSIM\ -\ Sum\ Image\ Recon: }$" + sumImageSSIM + r"$\bf{\ \ Average\ Channel\ Recon:\ }$" + chanImageAvgSSIM + r"$\bf{\ \ ERD:\ }$" + erdSSIM)
    else: plt.suptitle(r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured)
    
    if sampleData.simulationFlag: 
        ax = plt.subplot2grid(shape=(2,3), loc=(0,0))
        im = ax.imshow(sampleData.sumImage, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue)
        ax.set_title('Sum Image Ground-Truth')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if sampleData.simulationFlag: ax = plt.subplot2grid((2,3), (0,1))
    else: ax = plt.subplot2grid((1,3), (0,0))
    im = ax.imshow(sample.sumImageReconImage, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    ax.set_title('Sum Image Reconstruction')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if sampleData.simulationFlag: 
        ax = plt.subplot2grid((2,3), (0,2))
        im = ax.imshow(abs(sampleData.sumImage-sample.sumImageReconImage), cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue)
        ax.set_title('Absolute Difference')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if sampleData.simulationFlag: ax = plt.subplot2grid((2,3), (1,0))
    else: ax = plt.subplot2grid((1,3), (0,1))
    im = ax.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Measurement Mask')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    if sampleData.simulationFlag: ax = plt.subplot2grid((2,3), (1,1))
    else: ax = plt.subplot2grid((1,3), (0,2))
    im = ax.imshow(sample.ERD, cmap='viridis', vmin=0, aspect='auto')
    ax.set_title('ERD')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if sampleData.simulationFlag: 
        ax = plt.subplot2grid((2,3), (1,2))
        im = ax.imshow(sample.RD, cmap='viridis', vmin=0, aspect='auto')
        ax.set_title('RD')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    #Save
    f.tight_layout()
    f.subplots_adjust(top = 0.85)
    saveLocation = dir_progression + 'progression' + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '_avg.png'
    plt.savefig(saveLocation)
    plt.close()

    #Borderless saves
    saveLocation = dir_progression + 'reconstruction_sumImage' + '_iter_' + str(sample.iteration) +  '_perc_' + str(sample.percMeasured) + '.png'
    borderlessPlot(sample.sumImageReconImage, saveLocation, cmap='hot', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    
    saveLocation = dir_progression + 'mask_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
    borderlessPlot(sample.mask, saveLocation, cmap='gray')
    
    saveLocation = dir_progression + 'ERD_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
    borderlessPlot(sample.ERD, saveLocation, cmap='viridis')
    
    saveLocation = dir_progression + 'measured_sumImage_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
    borderlessPlot(sample.sumImage, saveLocation, cmap='hot', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    
    if sampleData.simulationFlag:
        saveLocation = dir_progression + 'RD_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(sample.RD, aspect='auto', vmin=0)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()
        saveLocation = dir_progression + 'groundTruth_sumImage.png'
        borderlessPlot(sampleData.sumImage, saveLocation, cmap='hot', vmin=sumImageMinValue, vmax=sumImageMaxValue)

def runSampling(sampleData, cValue, model, percToScan, percToViz, bestCFlag, oracleFlag, lineVisitAll, liveOutputFlag, dir_Results, datagenFlag, impModel, tqdmHide):

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
    result = Result(sampleData, liveOutputFlag, dir_Results, bestCFlag, datagenFlag, cValue, impModel)
    
    #Scan initial sets
    for initialSet in sampleData.initialSets: sample.performMeasurements(sampleData, result, initialSet, model, cValue, bestCFlag, oracleFlag, datagenFlag, False)
    
    #Check stopping criteria, just in case of a bad input
    if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random' or not lineVisitAll) and (sample.percMeasured >= sampleData.stopPerc): completedRunFlag = True
    elif sampleData.scanMethod == 'linewise' and sampleData.finalDim[0]-np.sum(np.sum(sample.mask, axis=1)>0) == 0: completedRunFlag = True
    if np.sum(sample.physicalERD) == 0: completedRunFlag = True
    
    #Perform the first update for the result
    result.update(sample)
    
    if not lineVisitAll or sampleData.scanMethod != 'linewise': maxProgress = stopPerc
    else: maxProgress = 100

    #Until the stopping criteria has been met
    with tqdm(total = float(maxProgress), desc = '% Sampled', leave=False, ascii=True, disable=tqdmHide) as pbar:

        #Initialize progress bar state according to % measured
        pbar.n = np.clip(round(sample.percMeasured,2), 0, maxProgress)
        pbar.refresh()
        
        #Until the program has completed
        while not completedRunFlag:

            #Find next measurement locations
            newIdxs = findNewMeasurementIdxs(sample, sampleData, result, model, cValue, percToScan, oracleFlag, bestCFlag, datagenFlag)
            
            #Perform measurements, reconstructions and ERD/RD computations
            if len(newIdxs) != 0: sample.performMeasurements(sampleData, result, newIdxs, model, cValue, bestCFlag, oracleFlag, datagenFlag, False)
            else: break
            
            #Check stopping criteria
            if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random' or not lineVisitAll) and (sample.percMeasured >= sampleData.stopPerc): completedRunFlag = True
            elif sampleData.scanMethod == 'linewise' and sampleData.finalDim[0]-np.sum(np.sum(sample.mask, axis=1)>0) == 0: completedRunFlag = True
            if np.sum(sample.physicalERD) == 0: completedRunFlag = True

            #If viz limit, only update when percToViz has been met; otherwise update every iteration
            if ((percToViz != None) and ((sample.percMeasured-result.percsMeasured[-1]) >= percToViz)) or (percToViz == None) or sampleData.scanMethod == 'linewise' or completedRunFlag: result.update(sample)

            #Update the progress bar
            pbar.n = np.clip(round(sample.percMeasured,2), 0, maxProgress)
            pbar.refresh()
            
    return result
    
#Compute approximated Reduction in Distortion (RD) values
def computeRD(sample, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborDistances, cValue, bestCFlag, datagenFlag, update, liveOutputFlag, impModel):
    
    #If a full calculation of RD then use the squareUnMeasured locations, otherwise find those that should be updated
    if not update: 
        squareUnMeasuredLocations = squareUnMeasuredIdxs
        neighborDistances = neighborDistances[:,0]
    else:
        squareUnMeasuredLocations = np.empty((0,2)).astype(int)
        updateLocations = np.argwhere(sample.prevSquareMask-sample.squareMask)
        
        #Prepare variables for indexing
        updateLocations_list = updateLocations.tolist()
        squareMeasuredIdxs_list = squareMeasuredIdxs.tolist()
        neighborIndices = neighborIndices[:,0].ravel()
        
        #Find indices of updateLocations and then the indices of neighboring squareUnMeasuredLocations 
        indices = [squareMeasuredIdxs_list.index(updateLocations_list[index]) for index in range(0, len(updateLocations))]
        indices = np.concatenate([np.argwhere(neighborIndices==index) for index in indices]).flatten()

        #If there are no locations that need updating, then just return
        if len(indices)==0: return
        
        #Extract squareUnMeasuredLocations to be updated and their relevant neighbor information (to avoid recalculation)
        neighborDistances = neighborDistances[:,0][indices]
        squareUnMeasuredLocations = squareUnMeasuredIdxs[indices]
        
    #Calculate the sigma values for chosen c value
    sigmaValues = neighborDistances/cValue
    
    #Determine window sizes, ensuring odd values, and radii for Gaussian generation
    if not staticWindow: windowSizes = np.ceil(2*dynWindowSigMult*sigmaValues).astype(int)+1
    else: windowSizes = (np.ones((len(sigmaValues)))*staticWindowSize).astype(int)
    windowSizes[windowSizes%2==0] += 1
    radii = (windowSizes//2).reshape(-1, 1).astype(int)
    
    #Zero-pad the RDPPs and get offset positions according to maxRadius for window extraction
    maxRadius = np.max(radii)
    paddedRDPPs = np.pad(sample.RDPPs, [(0, 0), (maxRadius, maxRadius), (maxRadius, maxRadius)], mode='constant')
    offsetLocations = squareUnMeasuredLocations + maxRadius
    startOffsetLocations, stopOffsetLocations = offsetLocations-radii, offsetLocations+radii
    
    #Determine gaussian window parameters for each unmeasured location; act as keys to gaussianWindows
    gaussianParams = list(map(tuple, np.vstack((windowSizes, sigmaValues)).T))
    
    #Generate a Gaussian for each unique sigma and window size, storing in a dictionary for ease of reference
    uniqueGaussianParams = np.unique(gaussianParams, axis=0)
    gaussianSignals = [signal.gaussian(windowSize, sigma) for windowSize, sigma in uniqueGaussianParams]
    gaussianWindows = dict([(tuple(uniqueGaussianParams[index]), np.outer(gaussianSignals[index], gaussianSignals[index])) for index in range(0, len(gaussianSignals))])
    
    #Compute RD Values
    sample.squareRDs[:, squareUnMeasuredLocations[:,0], squareUnMeasuredLocations[:,1]] = np.asarray([np.sum(gaussianWindows[gaussianParams[index]]*paddedRDPPs[:, startOffsetLocations[index][0]:stopOffsetLocations[index][0]+1, startOffsetLocations[index][1]:stopOffsetLocations[index][1]+1], axis=(1,2))  for index in range(0, len(offsetLocations))]).T
    
    #Make sure measured locations have 0 RD values
    sample.squareRDs = sample.squareRDs*(1-sample.squareMask)

    #Average the results together to form a single RD, by which to make selections
    sample.squareRD = np.mean(sample.squareRDs, axis=0)

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
    
    #Create array to hold featuresv
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
def prepareInput(sample, numChannel):
    if erdModel == 'DLADS': return np.dstack((sample.squareMask, sample.squareChanReconImages[numChannel]*(1-sample.squareMask), sample.squareChanReconImages[numChannel]*sample.squareMask))
    elif erdModel == 'GLANDS': return np.dstack((sample.squareMask, sample.squareChanReconImages[numChannel]*sample.squareMask))

#Determine the Estimated Reduction in Distortion
def computeERD(sample, sampleData, model, squareUnMeasuredIdxs, squareMeasuredIdxs):

    #Compute the ERD with the prescribed model; if configured to, only use a single channel
    if not chanSingle:
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
            for chanNum in range(0, len(sample.squareERDs)): sample.squareERDs[chanNum, squareUnMeasuredIdxs[:, 0], squareUnMeasuredIdxs[:, 1]] = ray.get(model.remote(sample.polyFeatures[chanNum]))
        elif erdModel == 'DLADS': 
        
            #First try inferencing all m/z channels at the same time 
            if not sampleData.OOM_multipleChannels:
                try: sample.squareERDs = ray.get(model.remote(makeCompatible([prepareInput(sample, chanNum) for chanNum in range(0, len(sample.squareERDs))]))).copy()
                except: 
                    sampleData.OOM_multipleChannels = True
                    if (len(gpus) > 0): print('Warning - Could not inference ERD for all channels of sample '+sampleData.name+' simultaneously on system GPU; will try processing channels iteratively.')
                    if (len(gpus) == 0): print('Warning - Could not inference ERD for all channels of sample '+sampleData.name+' simultaneously on system; will try processing channels iteratively.')
            
            #If multiple channels causes an OOM, then try running each channel through on its own
            if sampleData.OOM_multipleChannels and not sampleData.OOM_singleChannel:
                try: sample.squareERDs = np.asarray([ray.get(model.remote(makeCompatible(prepareInput(sample, chanNum))))[0,:,:].copy() for chanNum in range(0, len(sample.squareERDs))])
                except: sampleData.OOM_singleChannel = True
            
            #If an OOM occured for both mutiple and single channel inferencing, then exit; need to either restart program with no GPUs, or there isn't enough system RAM
            if sampleData.OOM_multipleChannels and sampleData.OOM_singleChannel and (len(gpus) > 0): sys.exit('Error - Sample '+sampleData.name+' dimensions are too high for the ERD to be inferenced on system GPU; please try disabling the GPU in the CONFIG.')
            if sampleData.OOM_multipleChannels and sampleData.OOM_singleChannel and (len(gpus) == 0): sys.exit('Error - Sample '+sampleData.name+' dimensions are too high for the ERD to be inferenced on this system by the loaded model.')
            
    else:
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
            ERDValues = ray.get(model.remote(sample.polyFeatures[0]))
            for chanNum in range(0, len(sample.squareERDs)): sample.squareERDs[chanNum, squareUnMeasuredIdxs[:, 0], squareUnMeasuredIdxs[:, 1]] = ERDValues
        elif erdModel == 'DLADS': 
            sample.squareERDs[0] = ray.get(model.remote(makeCompatible(prepareInput(sample, 0))))[0,:,:].copy()
            for chanNum in range(1, len(sample.squareERDs)): sample.squareERDs[chanNum] = sample.squareERDs[0]
    
    #Remove any negative values, measured locations, nan, or inf values
    sample.squareERDs[sample.squareERDs<0] = 0
    sample.squareERDs = sample.squareERDs*(1-sample.squareMask)
    sample.squareERDs = np.nan_to_num(sample.squareERDs, nan=0, posinf=0, neginf=0)
    sample.squareERD = np.mean(sample.squareERDs, axis=0)

#Determine which unmeasured points of a sample should be scanned given the current E/RD
def findNewMeasurementIdxs(sample, sampleData, result, model, cValue, percToScan, oracleFlag, bestCFlag, datagenFlag):

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
                
                #If there are no more points with physical ERD > 0, break from loop
                if np.sum(sample.physicalERD) <= 0: break
                
                #Find next measurement location and store the chosen scanning location for later, actual measurement
                newIdx = sample.unMeasuredIdxs[np.argmax(sample.physicalERD[sample.unMeasuredIdxs[:,0], sample.unMeasuredIdxs[:,1]])]
                newIdxs.append(newIdx.tolist())
                
                #Perform the measurement, using values from reconstruction 
                sample.performMeasurements(sampleData, result, newIdx, model, cValue, bestCFlag, oracleFlag, datagenFlag, True)
                
                #When enough new locations have been determined, break from loop
                if (np.sum(sample.mask)-np.sum(result.lastMask)) >= sampleData.pointsToScan: break
                
            #Convert to array for indexing
            newIdxs = np.asarray(newIdxs)
        else:
            #Identify the unmeasured location with the highest physicalERD value; return in a list to ensure it is iterable
            newIdxs = np.asarray([sample.unMeasuredIdxs[np.argmax(sample.physicalERD[sample.unMeasuredIdxs[:,0], sample.unMeasuredIdxs[:,1]])].tolist()])
            
    elif sampleData.scanMethod == 'linewise':

        #Create a list to hold the chosen scanning locations
        newIdxs = []

        #Choose the line with maximum physical ERD and extract the actual indices
        lineToScanIdx = np.nanargmax(np.nansum(sample.physicalERD, axis=1))

        #If points on the line should be chosen one-by-one, temporarily using reconstruction values for updating the ERD
        if lineMethod == 'percLine' and linePointSelection == 'single': 
            
            #Until the stopPerc has been reached, substitute reconstruction values for actual measurements
            while True:
                
                #If there are no points to scan on this line with physical ERD > 0, break from loop
                if np.sum(sample.physicalERD[lineToScanIdx]) <= 0: break
                
                #Identify the next scanning location and store it for later, actual measurement
                nextIndex = np.argmax(sample.physicalERD[lineToScanIdx])
                
                #Store that choice for later actual measurement
                newIdxs.append([lineToScanIdx, nextIndex])
                
                #Perform the measurement using values from reconstruction 
                sample.performMeasurements(sampleData, result, np.asarray(newIdxs[-1]), model, cValue, bestCFlag, oracleFlag, datagenFlag, True)
                
                #When enough new locations have been determined, break from loop
                if len(newIdxs) >= sampleData.pointsToScan: break
                
            #Convert to array for indexing
            newIdxs = np.asarray(newIdxs)
            
            #Sort columns for progressive physical scanning order
            newIdxs[:,1] = np.sort(newIdxs[:,1])
            
        #If points on the line should be selected in one step/group
        elif lineMethod == 'percLine' and linePointSelection == 'group':
            indexes = np.sort(np.argsort(sample.physicalERD[lineToScanIdx])[::-1][:sampleData.pointsToScan])
            newIdxs = np.column_stack([np.ones(len(indexes))*lineToScanIdx, indexes]).astype(int)
        
        #If all the points on a chosen line should be scanned
        if lineMethod =='fullLine':
            indexes = np.sort(np.argsort(sample.physicalERD[lineToScanIdx])[::-1])
            newIdxs = np.column_stack([np.ones(len(indexes))*lineToScanIdx, indexes]).astype(int)
        
        #==========================================
        #PARTIAL LINE BY START/END POINTS
        #==========================================
        #Choose segment to scan on line
        if lineMethod == 'segLine': 
            if segLineMethod == 'otsu':
                indexes = np.sort(np.where(sample.physicalERD[lineToScanIdx]>skimage.filters.threshold_otsu(sample.physicalERD, nbins=100))[0])
                if len(indexes)>0: 
                    indexes = np.arange(indexes[0],indexes[-1]+1)
                    newIdxs = np.column_stack([np.ones(len(indexes))*lineToScanIdx, indexes]).astype(int)
            elif segLineMethod == 'minPerc':
                indexes = np.sort(np.argsort(sample.physicalERD[lineToScanIdx])[::-1][:sampleData.pointsToScan])
                if len(indexes)>0: newIdxs = np.column_stack([np.ones(indexes[-1]-indexes[0]+1)*lineToScanIdx, np.arange(indexes[0],indexes[-1]+1)]).astype(int)
        #==========================================
        
        #==========================================
        #SELECTION SAFEGUARD
        #==========================================
        #If there are not enough locations selected, then return no new measurement locations which will terminate scanning
        if len(newIdxs) < int(round(0.01*sample.mask.shape[1])): return []
        #==========================================
        
    return newIdxs

def findNeighbors(measuredIdxs, unMeasuredIdxs):

    #Calculate knn and determine inverse distance weights with sklearn
    neighborDistances, neighborIndices = NearestNeighbors(n_neighbors=numNeighbors).fit(measuredIdxs).kneighbors(unMeasuredIdxs)
    unNormNeighborWeights = 1.0/(neighborDistances**2.0)
    neighborWeights = unNormNeighborWeights/(np.sum(unNormNeighborWeights, axis=1))[:, np.newaxis]

    return neighborIndices, neighborWeights, neighborDistances

#Perform the reconstruction using IDW (inverse distance weighting)
def computeRecon(inputImage, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights):

    #Create a blank image for the reconstruction
    reconImage = np.zeros(inputImage.shape)
    
    #Retrieve measured values, compute reconstruction values, and combine; if 3D do all channels at once
    if len(reconImage.shape) == 3:
        measuredValues = inputImage[:, squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]]
        if len(squareUnMeasuredIdxs) > 0: reconImage[:, squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]] = np.sum(measuredValues[:, neighborIndices]*neighborWeights, axis=-1)
        reconImage[:, squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]] = measuredValues
    else:
        measuredValues = inputImage[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]]
        if len(squareUnMeasuredIdxs) > 0: reconImage[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]] = np.sum(measuredValues[neighborIndices]*neighborWeights, axis=1)
        reconImage[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]] = measuredValues

    return reconImage

def downConv(numFilters, inputs):
    return LeakyReLU(alpha=0.2)(Conv2D(numFilters, 3, padding='same')(LeakyReLU(alpha=0.2)(Conv2D(numFilters, 1, padding='same')(inputs))))

def upConv(numFilters, inputs):
    return Conv2D(numFilters, 3, activation='relu', padding='same')(Conv2D(numFilters, 1, activation='relu', padding='same')(inputs))

def unet(numFilters, numChannels):
    inputs = Input(shape=(None,None,numChannels), batch_size=None)
    conv0 = downConv(numFilters, inputs)
    conv1 = downConv(numFilters*2, MaxPool2D(pool_size=(2,2))(conv0))
    conv2 = downConv(numFilters*4, MaxPool2D(pool_size=(2,2))(conv1))
    conv3 = downConv(numFilters*8, MaxPool2D(pool_size=(2,2))(conv2))
    conv4 = downConv(numFilters*16, MaxPool2D(pool_size=(2,2))(conv3))
    up1 = Conv2D(numFilters*16, 2, activation='relu', padding='same')(customResize(conv4, conv3))
    conv5 = upConv(numFilters*8, concatenate([conv3, up1]))
    up2 = Conv2D(numFilters*8, 2, activation='relu', padding='same')(customResize(conv5, conv2))
    conv6 = upConv(numFilters*4, concatenate([conv2, up2]))
    up3 = Conv2D(numFilters*4, 2, activation='relu', padding='same')(customResize(conv6, conv1))
    conv7 = upConv(numFilters*2, concatenate([conv1, up3]))
    up4 = Conv2D(numFilters*2, 2, activation='relu', padding='same')(customResize(conv7, conv0))
    conv8 = upConv(numFilters, concatenate([conv0, up4]))
    outputs = Conv2D(1, 1, activation='relu', padding='same')(conv8)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

#Rescale spatial dimensions of tensor x to match to those of tensor y
def customResize(x, y):
    x = image_ops.resize_images_v2(x, array_ops.shape(y)[1:3], method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
    nshape = tuple(y.shape.as_list())
    x.set_shape((None, nshape[1], nshape[2], None))
    return x

#Convert image into TF model compatible shapes/tensors
def makeCompatible(image):
    
    #Turn into an array before processings; will error in the event of dimensional incompatability
    image = np.asarray(image)

    #Reshape for tensor transition, as needed by number of channels
    if len(image.shape) > 3: return image
    elif len(image.shape) > 2: return image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    else: return image.reshape((1,image.shape[0],image.shape[1],1))

#Interpolate results to a given precision for averaging results
def percResults(results, perc_testingResults, precision):

    percents = np.linspace(min(np.hstack(perc_testingResults)), max(np.hstack(perc_testingResults)), int((max(np.hstack(perc_testingResults)) - min(np.hstack(perc_testingResults))) / precision + 1))
    newResults = [np.interp(percents, perc_testingResults[resultNum], results[resultNum]) for resultNum in range(0, len(results))]
    averageResults = np.average(newResults, axis=0)
    
    return percents, averageResults

#Convert bytes into human readable format
def sizeFunc(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0: return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

#Determine absolute difference between two arrays
def computeDifference(array1, array2):
    return abs(array1-array2)

def borderlessPlot(image, saveLocation, cmap='viridis', vmin=None, vmax=None):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(image, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(saveLocation, bbox_inches=extent)
    plt.close()

def basicPlot(xData, yData, saveLocation, xLabel='', yLabel=''):
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(xData, yData, color='black')
    ax1.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel)
    plt.savefig(saveLocation)
    plt.close()

#Quick print for titles in UI 
def sectionTitle(title):
    print('\n' + ('#' * int(consoleColumns)))
    print(title)
    print(('#' * int(consoleColumns)) + '\n')

#Construct and print a header for the running configuration
def programTitle(versionNum, configFileName):
    configInfo = os.path.splitext(os.path.basename(configFileName).split('_')[1])[0]
    
    if erdModel == 'SLADS-LS': 
        header = "\
 ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄               ▄            ▄▄▄▄▄▄▄▄▄▄▄ \n\
▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌             ▐░▌          ▐░░░░░░░░░░░▌\n\
▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀              ▐░▌          ▐░█▀▀▀▀▀▀▀▀▀ \n\
▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌                       ▐░▌          ▐░▌          \n\
▐░█▄▄▄▄▄▄▄▄▄ ▐░▌          ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ ▐░▌          ▐░█▄▄▄▄▄▄▄▄▄ \n\
▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌\n\
 ▀▀▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌ ▀▀▀▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀▀▀ ▐░▌           ▀▀▀▀▀▀▀▀▀█░▌\n\
          ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌          ▐░▌             ▐░▌                    ▐░▌\n\
 ▄▄▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌             ▐░█▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄█░▌\n\
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌             ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌\n\
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀               ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀ \n"
    elif erdModel == 'SLADS-Net': 
        header = "\
 ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄               ▄▄        ▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ \n\
▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌             ▐░░▌      ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌\n\
▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀              ▐░▌░▌     ▐░▌▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀█░█▀▀▀▀ \n\
▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌                       ▐░▌▐░▌    ▐░▌▐░▌               ▐░▌     \n\
▐░█▄▄▄▄▄▄▄▄▄ ▐░▌          ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ ▐░▌ ▐░▌   ▐░▌▐░█▄▄▄▄▄▄▄▄▄      ▐░▌     \n\
▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌▐░░░░░░░░░░░▌     ▐░▌     \n\
 ▀▀▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌ ▀▀▀▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀▀▀ ▐░▌   ▐░▌ ▐░▌▐░█▀▀▀▀▀▀▀▀▀      ▐░▌     \n\
          ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌          ▐░▌             ▐░▌    ▐░▌▐░▌▐░▌               ▐░▌     \n\
 ▄▄▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌             ▐░▌     ▐░▐░▌▐░█▄▄▄▄▄▄▄▄▄      ▐░▌     \n\
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌             ▐░▌      ▐░░▌▐░░░░░░░░░░░▌     ▐░▌     \n\
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀               ▀        ▀▀  ▀▀▀▀▀▀▀▀▀▀▀       ▀      \n"
    elif erdModel == 'DLADS': 
        header = "\
 ▄▄▄▄▄▄▄▄▄▄   ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄ \n\
▐░░░░░░░░░░▌ ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌\n\
▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ \n\
▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          \n\
▐░▌       ▐░▌▐░▌          ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄ \n\
▐░▌       ▐░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌\n\
▐░▌       ▐░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌ ▀▀▀▀▀▀▀▀▀█░▌\n\
▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌          ▐░▌\n\
▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌\n\
▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌\n\
 ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀ \n"
    elif erdModel == 'GLANDS': 
        header = "\
 ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄ \n\
▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░░▌      ▐░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌\n\
▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌░▌     ▐░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ \n\
▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌▐░▌    ▐░▌▐░▌       ▐░▌▐░▌          \n\
▐░▌ ▄▄▄▄▄▄▄▄ ▐░▌          ▐░█▄▄▄▄▄▄▄█░▌▐░▌ ▐░▌   ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄ \n\
▐░▌▐░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌\n\
▐░▌ ▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌   ▐░▌ ▐░▌▐░▌       ▐░▌ ▀▀▀▀▀▀▀▀▀█░▌\n\
▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌    ▐░▌▐░▌▐░▌       ▐░▌          ▐░▌\n\
▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌▐░▌     ▐░▐░▌▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌\n\
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░▌      ▐░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌\n\
 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀        ▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀ \n"
    
    header+="\
Author(s):\tDavid Helminiak\t\tEECE Marquette University\n\
Advisor(s):\tDong Hye Ye\t\tEECE Marquette University\n\
License:\tGNU General Public License v3.0\n\
Version:\t"+versionNum+"\n\
Config:\t\t"+configInfo

    sectionTitle(header)
