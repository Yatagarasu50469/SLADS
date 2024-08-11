#==================================================================
#DEFINITIONS
#==================================================================

#Object for initializing and storing sample metadata
class SampleData:
    def __init__(self, sampleFolder, initialPercToScan, stopPerc, scanMethod, lineRevist, simulationFlag, trainFlag, datagenFlag, bestCFlag, liveOutputFlag, impFlag, postFlag, oracleFlag, name='NoSampleName_0'):
        
        #Save options as internal variables
        self.sampleFolder = sampleFolder
        self.initialPercToScan = initialPercToScan
        self.stopPerc = stopPerc
        self.scanMethod = scanMethod
        self.lineRevist = lineRevist
        self.simulationFlag = simulationFlag
        self.trainFlag = trainFlag
        self.liveOutputFlag = liveOutputFlag
        self.impFlag = impFlag
        self.postFlag = postFlag
        self.oracleFlag = oracleFlag
        self.name = name
        self.overwriteAllChanFiles = copy.deepcopy(overwriteAllChanFiles)
        
        #These flags should not be changed or used directly during this __init__ method
        self.datagenFlag = datagenFlag
        self.bestCFlag = bestCFlag
        
        #Setup expected initial variables with default values
        self.allChanEvalFlag = False
        self.imzMLExportFlag = False
        self.lineExt = None
        self.mask = None
        self.unorderedNames = False
        self.missingLines = np.asarray([])
        self.mzRanges = []
        self.chanValues = []
        self.chanIndexes = []
        self.maskFOV = None
        self.squareMaskFOV = None
        self.firstScanDone = False
        self.mzFinalGrid = None
        self.chanFinalGrid = None
        self.newTimes = None
        self.scanRate = None
        self.maxRadius = 0
        self.gaussianWindows = {}
        self.dataMSI = False
        self.readAllMSI = False
        self.mzOriginalIndices = None
        self.mzLowerIndex = None
        self.mzInitialCount = None
        self.mzInitialDist = None
        self.mzFinal = []
        self.mzLowerValues = None
        self.mzUpperValues = None
        self.mzDistIndices = None
        self.mzIndices = None
        self.allImagesDataPath = None
        self.allImagesPath = None
        self.squareAllImagesPath = None
        self.avgTimeFileLoad = None
        self.useAlphaTims = False
        self.squareOpticalImage = None
        
        #If linewise, set the % per line as the original stop percentage, if visiting all lines then set the latter value to 100
        if self.scanMethod == 'linewise':
            self.linePerc = copy.deepcopy(stopPerc)
            if lineVisitAll: self.stopPerc = 100.0
            else: self.stopPerc = stopPerc
        
        #Store location of MSI data and sample name
        if self.name == 'NoSampleName_0': self.name = os.path.basename(sampleFolder)
        
        #Set global variables to indicate that OOM error states have not yet occurred; limited handle for ERD inferencing limitations
        self.OOM_multipleChannels, self.OOM_singleChannel = False, False
        
        #Note which files have already been read
        self.readScanFiles = []
        
        #Storage location for matching sequentially generated indexes with physical line numbers
        self.physicalLineNums = {}
        
        #Read in data from sampleInfo.txt, starting with sample type, otherwise assume image data
        lineIndex = 0
        try: 
            sampleInfo = open(sampleFolder+os.path.sep+'sampleInfo.txt').readlines()
            self.sampleType = sampleInfo[lineIndex].rstrip()
            lineIndex += 1
        except: 
            sys.exit('\nError - sampleInfo.txt was not found in the sample folder')
        
        #Check if alphatims should be used instead of multiplierz
        if self.sampleType == 'DESI-ALPHA':
            self.sampleType = 'DESI'
            self.useAlphaTims = True
        
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
            self.ppm = truncate(float(sampleInfo[lineIndex].rstrip())*1e-6, 7)
            lineIndex += 1

            #Read in lower m/z bound
            self.mzLowerBound = float(sampleInfo[lineIndex].rstrip())
            lineIndex += 1
            
            #Read in upper m/z bound
            self.mzUpperBound = float(sampleInfo[lineIndex].rstrip())
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
            self.ppm = truncate(float(sampleInfo[lineIndex].rstrip())*1e-6, 7)
            lineIndex += 1
            
            #Read in lower m/z bound
            self.mzLowerBound = float(sampleInfo[lineIndex].rstrip())
            lineIndex += 1
            
            #Read in upper m/z bound
            self.mzUpperBound = float(sampleInfo[lineIndex].rstrip())
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
            sys.exit('\nError - Unknown sample type: ' + self.sampleType + ' specified in sampleInfo.txt for : ' + sampleFolder)

        #MSI specific
        if self.sampleType == 'MALDI' or self.sampleType == 'DESI':
            
            #Indicate MSI data in a shorter variable
            self.dataMSI = True
            
            #If all spectrum channel data should be read, then only do so if a non-training simulation or an implementation, or a post-processing run; set internal flags as needed
            if ((self.simulationFlag and not self.trainFlag) or self.impFlag or self.postFlag) and (allChanEval or imzMLExport): 
                self.readAllMSI = True
                if allChanEval: self.allChanEvalFlag = True
                if imzMLExport: self.imzMLExportFlag = True
            
            #Whether lines are to be ignored or not is dependent on whether the run is a simulation 
            self.ignoreMissingLines = self.simulationFlag
            
            #Compute multiplicative variables for determining upper and lower m/z bounds given a central m/z value
            self.ppmPos, self.ppmNeg = 1+self.ppm, 1-self.ppm

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
        
        #If this is an ordered simulation sample, then get filetype extension and check for missing lines in DESI if applicable
        if self.simulationFlag:
            extensions = list(map(lambda x:x, np.unique([filename.split('.')[-1] for filename in natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'), reverse=False)])))
            if self.dataMSI:
                if 'd' in extensions: self.lineExt = '.d'
                elif 'D' in extensions: self.lineExt = '.D'
                elif 'raw' in extensions: self.lineExt = '.raw'
                elif 'RAW' in extensions: self.lineExt = '.RAW'
                elif 'imzML' in extensions: self.lineExt = '.imzML'
                else: sys.exit('\nError - Either no files are present, or an unknown filetype being used for sample: ' + self.name)
            elif self.sampleType == 'IMAGE':
                if 'png' in extensions: self.lineExt = '.png'
                elif 'jpg' in extensions: self.lineExt = '.jpg'
                elif 'tiff' in extensions: self.lineExt = '.tiff'
                else: sys.exit('\nError - Either no files are present, or an unknown filetype being used for sample: ' + self.name)
            scanFileNames = natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'+self.lineExt), reverse=False)
            if self.sampleType == 'DESI' and self.ignoreMissingLines and not self.unorderedNames:
                self.missingLines = np.asarray(list(set(np.arange(1, self.finalDim[0]).tolist()) - set([int(scanFileName.split('line-')[1].split('.')[0].lstrip('0')) for scanFileName in scanFileNames])))-1
                self.finalDim[0] -= len(self.missingLines)
        
        #For DESI samples, determine image dimensions that will produce square pixels (consistent vertical/horizontal resolution) and new times to use as a common grid
        if self.sampleType == 'DESI':
            if(self.finalDim[1]/self.sampleWidth) > (self.finalDim[0]/self.sampleHeight): self.squareDim = [int(round((self.finalDim[1]*self.sampleHeight)/self.sampleWidth)), self.finalDim[1]]
            elif (self.finalDim[1]/self.sampleWidth) < (self.finalDim[0]/self.sampleHeight): self.squareDim = [self.finalDim[0], int(round((self.finalDim[0]*self.sampleWidth)/self.sampleHeight))]
            else: self.squareDim = self.finalDim
            self.newTimes = np.ascontiguousarray(np.linspace(0, ((self.sampleWidth*1e3)/self.scanRate)/60, self.finalDim[1]))
        
        #Determine whether to mask the FOV or not
        if (self.trainFlag and trainMaskFOVDilation != None) or (not self.trainFlag and otherMaskFOVDilation != None): self.useMaskFOV = True
        else: self.useMaskFOV = False
        
        #If masking the FOV measurement space, try loading a mask and dilating it if specificed, disable if no mask found and inform user
        if self.useMaskFOV: 
            try: 
                self.maskFOV = np.loadtxt(self.sampleFolder+os.path.sep+'mask.csv', np.float32, delimiter=',')
                self.squareMaskFOV = resize(self.maskFOV, tuple(self.squareDim), order=0)
                if self.trainFlag and (trainMaskFOVDilation > 0):
                    self.squareMaskFOV = cv2.dilate(self.squareMaskFOV.astype(np.uint8), np.ones((trainMaskFOVDilation,trainMaskFOVDilation), np.uint8), iterations=1).astype(np.float32)
                    self.maskFOV = resize(self.squareMaskFOV, tuple(self.finalDim), order=0)
                elif not self.trainFlag and (otherMaskFOVDilation > 0):
                    self.squareMaskFOV = cv2.dilate(self.squareMaskFOV.astype(np.uint8), np.ones((otherMaskFOVDilation,otherMaskFOVDilation), np.uint8), iterations=1).astype(np.float32)
                    self.maskFOV = resize(self.squareMaskFOV, tuple(self.finalDim), order=0)
            except: 
                print('\nWarning - FOV mask use was enabled, but no mask.csv available for ' + self.name + '. Disabled for this sample, but could cause evaluation issues (particularly if percFOVMask enabled). Consider disabling this in the program configuration file.')
                self.useMaskFOV = False
            
        #If post-processing load the measurement mask
        if self.postFlag:
            try: self.mask = np.loadtxt(self.sampleFolder+os.path.sep+'measuredMask.csv', np.float32, delimiter=',')
            except: sys.exit('\nError - Unable to load measurement mask for sample: ' + self.name)
            try: 
                self.progMap = np.loadtxt(self.sampleFolder+os.path.sep+'progressMap.csv', np.float32, delimiter=',')
                self.progMap[self.progMap==-1]=np.nan
            except: self.progMap = np.empty([])
        
        #Establish sample area to measure; do not apply percFOVMask in training/validation database
        if self.useMaskFOV and percFOVMask and not self.trainFlag: self.area = np.sum(self.maskFOV)
        else: self.area = int(round(self.finalDim[0]*self.finalDim[1]))
        
        #If not just post-processing, setup initial sets
        if not self.postFlag: self.generateInitialSets(self.scanMethod)
        
        #If MSI data, determine the needed precision to prevent overlap when mapping/indexing m/z values to a common spectrum
        #How many decimal places are used in the lowerbound mz range, as first limited by the ppm precision
        #Arbitrary precision rounding is problematic, so go to the lower power of 10 instead
        if self.dataMSI:
            ppmRound = ('%f' % self.ppm).rstrip('0').rstrip('.')[::-1].find('.')
            mzMinDiff = np.round((self.mzLowerBound*self.ppmPos)-(self.mzLowerBound*self.ppmNeg), ppmRound)
            self.mzRound = str(mzMinDiff)[::-1].find('.')
            self.mzPrecision = np.float32(10**-self.mzRound)
        
        #If reading in all MSI Data determine set of non-overlapping bins based on ppm
        if self.dataMSI and self.readAllMSI: 
        
            #Setup .hdf5 file locations
            self.allImagesDataPath = self.sampleFolder+os.path.sep+'allImageData'+os.path.sep
            self.allImagesPath = self.allImagesDataPath + 'allImages.hdf5'
            if self.sampleType=='MALDI': self.squareAllImagesPath = None
            elif self.sampleType=='DESI': self.squareAllImagesPath = self.allImagesDataPath+'squareAllImages.hdf5'
            
            #If not configured to overwrite, but the allImages .hdf5 file doesn't exist, then activate the flag
            if not self.overwriteAllChanFiles and not os.path.exists(self.allImagesPath): self.overwriteAllChanFiles = True
            
            #If overwrite enabled
            if self.overwriteAllChanFiles:
            
                #Erase/create storage location
                if os.path.exists(self.allImagesDataPath): shutil.rmtree(self.allImagesDataPath)
                os.makedirs(self.allImagesDataPath)
                
                #Compute the unshifted lower m/z bound's index position
                self.mzLowerIndex = int(self.mzLowerBound/self.mzPrecision)
                
                #Compute m/z for initial binning
                self.mzInitialCount = int((self.mzUpperBound-self.mzLowerBound)/self.mzPrecision)+1
                self.mzInitialDist = np.round(np.round(np.linspace(self.mzLowerBound, self.mzUpperBound, self.mzInitialCount)/self.mzPrecision)*self.mzPrecision, self.mzRound)
                
                #Determine m/z values by ppm with non-overlapping bins
                self.mzFinal = [self.mzLowerBound/self.ppmNeg]
                while(True):
                    mzNewValue = (self.mzFinal[-1]*self.ppmPos)/self.ppmNeg
                    if mzNewValue*self.ppmNeg <= self.mzUpperBound: self.mzFinal.append(mzNewValue)
                    else: break
                
                #Set final mz as a contiguous array 
                self.mzFinal = np.ascontiguousarray(self.mzFinal)
                
                #Compute lower/upper bin bounds for each of the final m/z ranges
                self.mzLowerValues, self.mzUpperValues = self.mzFinal*self.ppmNeg, self.mzFinal*self.ppmPos
                
                #Find unique index mapping between maximum precision distribution and the non-overlapping ppm bins
                _, self.mzOriginalIndices = np.unique([bisect_left(self.mzUpperValues, mzValue) for mzValue in self.mzInitialDist], return_index=True)
                
                #Save variables for future reuse
                np.save(self.allImagesDataPath + 'mzLowerIndex', self.mzLowerIndex)
                np.save(self.allImagesDataPath + 'mzInitialCount', self.mzInitialCount)
                np.save(self.allImagesDataPath + 'mzInitialDist', self.mzInitialDist)
                np.save(self.allImagesDataPath + 'mzFinal', self.mzFinal)
                np.save(self.allImagesDataPath + 'mzLowerValues', self.mzLowerValues)
                np.save(self.allImagesDataPath + 'mzUpperValues', self.mzUpperValues)
                np.save(self.allImagesDataPath + 'mzOriginalIndices', self.mzOriginalIndices)
                
            else: 
                self.mzLowerIndex = np.load(self.allImagesDataPath + 'mzLowerIndex.npy', allow_pickle=True)
                self.mzInitialCount = np.load(self.allImagesDataPath + 'mzInitialCount.npy', allow_pickle=True)
                self.mzInitialDist = np.load(self.allImagesDataPath + 'mzInitialDist.npy', allow_pickle=True)
                self.mzFinal = np.load(self.allImagesDataPath + 'mzFinal.npy', allow_pickle=True)
                self.mzLowerValues = np.load(self.allImagesDataPath + 'mzLowerValues.npy', allow_pickle=True)
                self.mzUpperValues = np.load(self.allImagesDataPath + 'mzUpperValues.npy', allow_pickle=True)
                self.mzOriginalIndices = np.load(self.allImagesDataPath + 'mzOriginalIndices.npy', allow_pickle=True)
        
        #If MSI data, regardless of if all channel data is to be loaded or not
        if self.dataMSI:
            
            #Load targeted m/z values in sorted order, preferring those in the sample folders if available, and set corresponding ranges 
            #If there is only a single specificed channel, then need to convert format to array with list
            try: 
                self.chanValues = np.loadtxt(self.sampleFolder+os.path.sep+'channels.csv', delimiter=',')
            except: 
                if overrideChannelsFile == None: self.chanValues = np.loadtxt('channels.csv', delimiter=',')
                else: self.chanValues = np.loadtxt(overrideChannelsFile, delimiter=',')
            if self.chanValues.shape == (): self.chanValues = np.array([self.chanValues])
            self.numChannels = len(self.chanValues)
            self.chanValues.sort()
            self.mzRanges = np.column_stack((self.chanValues*self.ppmNeg, self.chanValues*self.ppmPos))
            
            #For DESI MSI samples, create regular grids for interpolating all/targeted m/z data
            if self.sampleType == 'DESI': 
                mzFinalGrid_x, mzFinalGrid_y = np.meshgrid(self.newTimes, self.mzFinal)
                if parallelization: self.mzFinalGrid_id = ray.put((mzFinalGrid_x, mzFinalGrid_y))
                else: self.mzFinalGrid = (mzFinalGrid_x, mzFinalGrid_y)
                chanFinalGrid_x, chanFinalGrid_y = np.meshgrid(self.newTimes, self.chanValues)
                if parallelization: self.chanFinalGrid_id = ray.put((chanFinalGrid_x, chanFinalGrid_y))
                else: self.chanFinalGrid = (chanFinalGrid_x, chanFinalGrid_y)
                _ = gc.collect()
            
            #Setup shared memory actor for operations in parallel, or local memory for serial execution
            if parallelization: 
                self.mzOriginalIndices_id, self.mzRanges_id = ray.put(self.mzOriginalIndices), ray.put(self.mzRanges)
                self.reader_MSI_Actor = Reader_MSI_Actor.remote(self.sampleType, self.readAllMSI, len(self.mzFinal), len(self.chanValues), self.finalDim[0], self.finalDim[1], self.allImagesDataPath, self.allImagesPath, self.squareAllImagesPath, self.overwriteAllChanFiles)
            elif self.readAllMSI and self.overwriteAllChanFiles: 
                self.allImages = np.zeros((len(self.mzFinal), self.finalDim[0], self.finalDim[1]), dtype=np.float32)
        
        #Setup targeted channel and sum images for holding data
        self.chanImages = np.zeros((self.numChannels, self.finalDim[0], self.finalDim[1]), dtype=np.float32)
        self.sumImage = np.zeros((self.finalDim), dtype=np.float32)
        
        #If an MSI sample and an optical image is to be used try loading different extensions
        if self.dataMSI and ('opticalData' in inputChannels):
            opticalImageFound = False
            for extension in ['.png', '.jpg', '.tiff']:
                if os.path.isfile(self.sampleFolder+os.path.sep+'optical'+extension): 
                    opticalImageFound = True
                    break
            if not opticalImageFound: sys.exit('\nError - opticalData was enabled, but no optical image was found for sample: ' + sample.name)
            
            #Load the optical image, rescaling range 0 to 1 and inversing non-zero values (remove background and positively weight structures)
            opticalImage = (cv2.imread(self.sampleFolder+os.path.sep+'optical'+extension, 0)/255)
            opticalMask = opticalImage!=0
            opticalImage[opticalMask] = 1-opticalImage[opticalMask]
            self.squareOpticalImage = resize(opticalImage, tuple(self.squareDim), order=0)
            self.opticalImage = resize(opticalImage, tuple(self.finalDim), order=0)
        
        #If a simulation or post-processing, read all the sample data and save in hdf5 if applicable, optimized for loading whole channel images
        if self.simulationFlag or self.postFlag: 
            self.avgTimeFileLoad = self.readScanData()
            if self.dataMSI: 
                if parallelization: 
                    if self.readAllMSI: self.allImagesMin, self.allImagesMax = copy.deepcopy(ray.get(self.reader_MSI_Actor.writeToDisk.remote(self.squareDim)))
                    del self.reader_MSI_Actor, self.mzOriginalIndices_id, self.mzRanges_id
                    if self.sampleType == 'DESI': del self.mzFinalGrid_id, self.chanFinalGrid_id
                else: 
                    if self.readAllMSI: 
                        self.allImagesMin, self.allImagesMax = np.min(self.allImages, axis=(1,2)), np.max(self.allImages, axis=(1,2))
                        np.save(self.allImagesDataPath + 'allImagesMin', self.allImagesMin)
                        np.save(self.allImagesDataPath + 'allImagesMax', self.allImagesMax)
                        if self.overwriteAllChanFiles:
                            allImagesFile = h5py.File(self.allImagesPath, 'a')
                            _ = allImagesFile.create_dataset(name='allImages', data=self.allImages, chunks=(1, self.finalDim[0], self.finalDim[1]), dtype=np.float32)
                            allImagesFile.close()
                            if self.sampleType=='DESI':
                                squareAllImagesFile = h5py.File(self.squareAllImagesPath, 'a')
                                _ = squareAllImagesFile.create_dataset(name='squareAllImages', data=np.moveaxis(resize(self.allImages, tuple(squareDim), order=0), -1, 0), chunks=(1, squareDim[0], squareDim[1]), dtype=np.float32)
                                squareAllImagesFile.close()
                        del self.allImages
                        _ = gc.collect()

    #Generate initial scanning mask; allows changing the scan method without rescanning all of the data
    def generateInitialSets(self, scanMethod):
    
        #Update the scan method
        self.scanMethod = scanMethod
        
        #List of what points/lines should be initially measured
        self.initialSets = []
        
        #If scanning with line-bounded constraint
        if self.scanMethod == 'linewise':
        
            #If applicable, limit rows/columns to consider according to the FOV mask
            if self.useMaskFOV: 
            
                #Create list of arrays containing points to measure on each line
                validRows = np.where(np.sum(self.maskFOV, axis=1)>0)[0]
                self.linesToScan = [[tuple([rowNum, columnNum]) for columnNum in np.where(self.maskFOV[rowNum]>0)[0]] for rowNum in validRows]
                
                #Set initial lines to scan
                lineIndexes = np.unique([int(np.ceil(len(validRows)*startLinePosition)) for startLinePosition in startLinePositions]).tolist()
                
            else:
            
                #Create list of arrays containing points to measure on each line
                self.linesToScan = np.asarray([[tuple([rowNum, columnNum]) for columnNum in np.arange(0, self.finalDim[1], 1)] for rowNum in np.arange(0, self.finalDim[0], 1)]).tolist()

                #Set initial lines to scan
                lineIndexes = np.unique([int(np.ceil((self.finalDim[0]-1)*startLinePosition)) for startLinePosition in startLinePositions]).tolist()
            
            #Obtain points in the specified lines and add them to the initial scan list
            for lineIndex in lineIndexes:
                
                #If only a percentage should be scanned, then randomly select points, otherwise select all
                if lineMethod == 'percLine':
                    newIdxs = copy.deepcopy(self.linesToScan[lineIndex])
                    np.random.shuffle(newIdxs)
                    newIdxs = newIdxs[:int(np.ceil((self.linePerc/100)*len(newIdxs)))]
                else: 
                    newIdxs = [pt for pt in self.linesToScan[lineIndex]]
                
                #Add positions to initial list
                self.initialSets.append(newIdxs)
                
        elif self.scanMethod == 'pointwise' or self.scanMethod == 'random':
            
            #Randomly select points to initially scan
            if self.useMaskFOV: newIdxs = np.transpose(np.where(self.maskFOV==1))
            else: newIdxs = np.transpose(np.where(np.zeros(self.finalDim, dtype=int)==0))
            np.random.shuffle(newIdxs)
            newIdxs = newIdxs[:int(np.ceil(((self.initialPercToScan/100)*self.area)))]
            
            #Add positions to initial list
            self.initialSets.append(newIdxs)
    
    def readScanData(self, newIdxs=[]):
        
        #Start file loading timer
        t0_readFile = time.perf_counter()
        
        #Get the MSI file extension automatically if it isn't already known
        if self.lineExt == None:
            extensions = list(map(lambda x:x, np.unique([filename.split('.')[-1] for filename in natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'), reverse=False)])))
            if self.dataMSI:
                if 'd' in extensions: self.lineExt = '.d'
                elif 'D' in extensions: self.lineExt = '.D'
                elif 'raw' in extensions: self.lineExt = '.raw'
                elif 'RAW' in extensions: self.lineExt = '.RAW'
                elif 'imzML' in extensions: self.lineExt = '.imzML'
                else: sys.exit('\nError - Either no files are present, or an unknown filetype being used for sample: ' + self.name)
            elif self.sampleType == 'IMAGE':
                if 'png' in extensions: self.lineExt = '.tiff'
                elif 'jpg' in extensions: self.lineExt = '.jpg'
                elif 'tiff' in extensions: self.lineExt = '.tiff'
                else: sys.exit('\nError - Either no files are present, or an unknown filetype being used for sample: ' + self.name)
        
        #TODO: MALDI experimental operation is not yet implemented, where the program should just read new positions referencing the chosen new idxs
        
        #Obtain and sort the available line files pertaining to the current scan
        scanFileNames = natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'+self.lineExt), reverse=False)
        
        #If IMAGE, each file corresponds to a channel, read in each and sum of all data, augmenting list of channel labels
        if self.sampleType == 'IMAGE':
            for chanNum in range(0, len(scanFileNames)):
                self.chanValues.append(scanFileNames[chanNum].split('-chan-')[1].split('.')[0])
                try: self.chanImages[chanNum] = cv2.imread(scanFileNames[chanNum], 0)
                except: sys.exit('\nError - Unable to read file' + scanFileNames[chanNum])
            self.sumImage = np.sum(np.atleast_3d(self.chanImages), axis=0)

        #If MALDI, then there is only a single file with all of the spectral data
        elif self.sampleType == 'MALDI':
        
            #Establish file pointer for the single imzML file and verify it is readable
            try: data = ImzMLParser(scanFileNames[0])
            except: sys.exit('\nError - Unable to read file' + scanFileNames[0])
            
            #Adjust stored coordinates to be zero-based
            coordinates = np.asarray(data.coordinates)-1
            
            #If parallelization is disabled then read in data sequentially, otherwise pass writable coordinates to parallel actor
            if not parallelization:
                for i, (x, y, z) in tqdm(enumerate(coordinates), total = len(coordinates), desc='Reading', leave=False, disable=self.impFlag, ascii=asciiFlag):
                    mzs, ints = data.getspectrum(i)
                    self.sumImage[y, x] = np.sum(ints)
                    if self.overwriteAllChanFiles and self.readAllMSI: 
                        filtIndexLow, filtIndexHigh = bisect_left(mzs, self.mzLowerBound), bisect_right(mzs, self.mzUpperBound)
                        self.allImages[:, y, x] = np.add.reduceat(mzFastIndex(mzs[filtIndexLow:filtIndexHigh], ints[filtIndexLow:filtIndexHigh], self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount), self.mzOriginalIndices).astype(np.float32)
                    self.chanImages[:, y, x] = [np.sum(ints[bisect_left(mzs, mzRange[0]):bisect_right(mzs, mzRange[1])]) for mzRange in self.mzRanges]
            else:
                _ = ray.get(self.reader_MSI_Actor.setCoordinates.remote(coordinates))
                _ = ray.get([msi_parhelper.remote(self.reader_MSI_Actor, self.useAlphaTims, self.readAllMSI, scanFileNames, indexes, self.mzOriginalIndices_id, self.mzRanges_id, self.sampleType, self.mzLowerBound, self.mzUpperBound, self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount, self.mask, self.newTimes, self.finalDim, self.sampleWidth, self.scanRate, self.overwriteAllChanFiles) for indexes in np.array_split(np.arange(0, len(coordinates)), numberCPUS)])
                
            #Close the file
            del data
            _ = gc.collect()
        
        #If DESI, each file corresponds to a full line of data
        elif self.sampleType == 'DESI':
        
            #If line revisiting is disabled, identify which files have not yet been scanned
            if not self.lineRevist: scanFileNames = natsort.natsorted(list(set(scanFileNames)-set(self.readScanFiles)), reverse=False)
            
            #If parallelization is disabled then read in data sequentially
            if not parallelization:
                
                #Extract line number from the filenames, removing leading zeros, subtract 1 for zero indexing, and obtain correct physical row indexes from LUT if applicable
                for scanFileName in tqdm(scanFileNames, total = len(scanFileNames), desc='Reading', leave=False, disable=self.impFlag, ascii=asciiFlag):
                    
                    #Load the line data and flag errors during the process (primarily checking for files without data)
                    errorFlag = False
                    try: 
                        if not self.useAlphaTims: 
                            data = mzFile(scanFileName, numThreads=1)
                        else: 
                            data = alphatims.bruker.TimsTOF(scanFileName, use_hdf_if_available=False)
                            data.format = 'Bruker'
                    except: 
                        errorFlag = True
                        if debugMode: print('\nWarning - Failed to load any data from file: ' + scanFileName + ' This file will be ignored this iteration.')
                    
                    #Extract the file number and if unordered find corresponding line number in LUT, otherwise line number is the file number minus 1
                    if not errorFlag:
                        fileNum = int(scanFileName.split('line-')[1].split('.')[0].lstrip('0'))-1
                        if self.unorderedNames: 
                            try: 
                                lineNum = self.physicalLineNums[fileNum+1]
                            except: 
                                errorFlag = True 
                                if debugMode: print('\nWarning - Failed to find the physical line number for the file: ' + scanFileName + ' This file will be ignored this iteration.')
                        else: lineNum = fileNum
                    
                    #If error still has not occurred 
                    if not errorFlag:
                        
                        #If ignoring missing lines and there are stored missing lines (simulation with ordered filenames only), then determine the offset for correct indexing
                        if self.ignoreMissingLines and len(self.missingLines) > 0: lineNum -= int(np.sum(lineNum > self.missingLines))
                        
                        #Add file name to those that will have been already scanned (when this process finishes)
                        self.readScanFiles.append(scanFileName)
                        
                        #Extract original measurement times
                        if data.format == 'Bruker': 
                            if not self.useAlphaTims: origTimes = np.asarray(data.ms1_frames)[:,1]/60
                            else: origTimes = np.delete(data.rt_values, 0, axis = 0)/60
                        else: 
                            origTimes = np.asarray(data.scan_info())[:,0].astype('float64')
                        
                        #Force original times memory allocation to be contiguous
                        origTimes = np.ascontiguousarray(origTimes)
                        
                        #Offset the original measurement times, such that the first position's time equals 0
                        origTimes -= np.min(origTimes)
                        
                        #If the data is being sparesly acquired in lines, then the listed times in the file need to be shifted
                        if (self.impFlag or self.postFlag) and impOffset and scanMethod == 'linewise' and (lineMethod == 'segLine' or lineMethod == 'fullLine'): origTimes += (np.argwhere(self.mask[lineNum]==1).min()/self.finalDim[1])*(((self.sampleWidth*1e3)/self.scanRate)/60)
                        elif (self.impFlag or self.postFlag) and impOffset: sys.exit('\nError - Using implementation or post-process modes with an offset but not segmented-linewise operation is not currently a supported configuration.')
                        
                        #Setup storage locations for each measured location
                        chanDataLine, mzDataLine = [[] for _ in range(0, len(self.mzRanges))], []
                        
                        #Set positions to be scanned for the line
                        if data.format == 'Bruker': positions = range(1, len(origTimes)+1)
                        else: positions = range(data.scan_range()[0], data.scan_range()[1]+1)
                        
                        #Create list for holding TIC data
                        sumImageLine = []
                        
                        #Read in and process spectrum data for each location; use 'profile' spectrum mode
                        for pos in positions:
                            if data.format == 'Bruker':
                                if not self.useAlphaTims: 
                                    mzs, ints = data.scan(pos, True)
                                else: 
                                    mzs, ints = data[pos]['mz_values'].values, data[pos]['corrected_intensity_values'].values
                                    sortedIndices = np.argsort(mzs)
                                    mzs, ints = mzs[sortedIndices], ints[sortedIndices]
                            elif data.format == 'Agilent': 
                                mzs, ints = data.scan(pos, 'profile')
                            elif data.format == 'raw': 
                                mzs, ints = data.scan(pos, False, True)
                            
                            #Obtain TIC by intensity summation
                            sumImageLine.append(np.sum(ints))
                            
                            if self.overwriteAllChanFiles and self.readAllMSI: 
                                filtIndexLow, filtIndexHigh = bisect_left(mzs, self.mzLowerBound), bisect_right(mzs, self.mzUpperBound)
                                mzDataLine.append(np.add.reduceat(mzFastIndex(mzs[filtIndexLow:filtIndexHigh], ints[filtIndexLow:filtIndexHigh], self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount), self.mzOriginalIndices))
                            for mzRangeNum in range(0, len(self.mzRanges)):
                                mzRange = self.mzRanges[mzRangeNum]
                                chanDataLine[mzRangeNum].append(np.sum(ints[bisect_left(mzs, mzRange[0]):bisect_right(mzs, mzRange[1])]))
                        
                        #Create regular grid interpolators, aligning all/targeted m/z data, and storing results
                        if self.overwriteAllChanFiles and self.readAllMSI: self.allImages[:, lineNum, :] = scipy.interpolate.RegularGridInterpolator((origTimes, self.mzFinal), np.asarray(mzDataLine, dtype='float64'), bounds_error=False, fill_value=0)(self.mzFinalGrid).astype('float32')
                        self.chanImages[:, lineNum, :] = scipy.interpolate.RegularGridInterpolator((origTimes, self.chanValues), np.asarray(chanDataLine, dtype='float64').T, bounds_error=False, fill_value=0)(self.chanFinalGrid).astype('float32')
                        self.sumImage[lineNum, :] = np.interp(self.newTimes, origTimes, np.nan_to_num(sumImageLine, nan=0, posinf=0, neginf=0), left=0, right=0)
                        
                        #Close the file; (Do not use multiplierz function: data.close() - it's really slow and unclear why
                        del data
                        _ = gc.collect()

            #Otherwise read data in parallel and perform remaining interpolations of any remaining m/z data to regular grid in serial (parallel operation is too memory intensive)
            else:
                _ = ray.get([msi_parhelper.remote(self.reader_MSI_Actor, self.useAlphaTims, self.readAllMSI, scanFileNames, indexes, self.mzOriginalIndices_id, self.mzRanges_id, self.sampleType, self.mzLowerBound, self.mzUpperBound, self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount, self.mask, self.newTimes, self.finalDim, self.sampleWidth, self.scanRate, self.overwriteAllChanFiles, self.mzFinal, self.mzFinalGrid_id, self.chanValues, self.chanFinalGrid_id, self.impFlag, self.postFlag, impOffset, scanMethod, lineMethod, self.physicalLineNums, self.ignoreMissingLines, self.missingLines, self.unorderedNames) for indexes in np.array_split(np.arange(0, len(scanFileNames)), numberCPUS)])
                if self.readAllMSI: _ = ray.get(self.reader_MSI_Actor.interpolateDESI.remote(self.mzFinal, self.mzFinalGrid_id))
                for scanFileName in ray.get(self.reader_MSI_Actor.getReadScanFiles.remote()).copy(): self.readScanFiles.append(scanFileName)

        #If parallelization is enabled, and this is a MSI sample, then read MSI data in parallel, retrieve from shared memory, and process data into accessible shape
        if parallelization and self.dataMSI:
            
            #Update local identification of which files have already been imported for DESI samples
            if self.sampleType == 'DESI': self.readScanFiles = ray.get(self.reader_MSI_Actor.getReadScanFiles.remote()).copy()
            
            #If there are were not new specific locations that were to be scanned, retrieve everything, otherwise only pull data for new idxs
            if len(newIdxs) == 0: 
                self.chanImages = np.moveaxis(ray.get(self.reader_MSI_Actor.getChanImages.remote()).copy(), -1, 0)
                self.sumImage = ray.get(self.reader_MSI_Actor.getSumImage.remote()).copy()
            else: 
                self.chanImages[:, newIdxs[:,0], newIdxs[:,1]] = ray.get(self.reader_MSI_Actor.getChanImagesNewIdxs.remote(newIdxs[:,0], newIdxs[:,1])).copy().T
                self.sumImage[newIdxs[:,0], newIdxs[:,1]] = ray.get(self.reader_MSI_Actor.getSumImageNewIdxs.remote(newIdxs[:,0], newIdxs[:,1])).copy()
            
        #If DESI MSI, then need to resize the images to obtain square dimensionality, otherwise the square dimensions are equal to the original
        if self.sampleType == 'DESI': self.squareChanImages = np.moveaxis(resize(np.moveaxis(self.chanImages, 0, -1), tuple(self.squareDim), order=0), -1, 0)
        else: self.squareChanImages = self.chanImages
        
        #Find the minimum/maximum values in each channel image for easy referencing
        self.chanImagesMin, self.chanImagesMax = np.min(self.chanImages, axis=(1,2)), np.max(self.chanImages, axis=(1,2))
        
        #Stop file load timer and return average across number of files scanned
        t1_readFile = time.perf_counter()
        if len(scanFileNames)>0: return (t1_readFile-t0_readFile)/len(scanFileNames)
        else: return np.nan

#Storage object for holding updatable variables needed over the course of scanning (here to prevent unneccessary memory usage)
class TempScanData:
    def __init__(self): 
        self.squareMeasuredIdxs = None
        self.squareUnMeasuredIdxs = None
        self.neighborIndices = None
        self.neighborWeights = None
        self.neighborDistances = None
        self.winStartPos = None
        self.winStopPos = None

#Relevant sample data at each time step; static information should be held in corresponding SampleData object
class Sample:
    def __init__(self, sampleData, tempScanData):
        
        #Initialize measurement masks and other variables that are expected to exist
        self.mask = np.zeros((sampleData.finalDim), dtype=np.float32)
        if sampleData.postFlag:
            self.progMap = sampleData.progMap
        else: 
            self.progMap = np.empty((sampleData.finalDim), dtype=np.float32)
            self.progMap[:] = np.nan
        if sampleData.sampleType == 'DESI': self.squareMask = resize(self.mask, tuple(sampleData.squareDim), order=0)
        else: self.squareMask = self.mask
        self.squareRD = np.zeros((sampleData.squareDim), dtype=np.float32)
        self.squareRDs = np.zeros((sampleData.numChannels, sampleData.squareDim[0], sampleData.squareDim[1]), dtype=np.float32)
        self.squareERD = np.zeros((sampleData.squareDim), dtype=np.float32)
        self.squareERDs = np.zeros((sampleData.numChannels, sampleData.squareDim[0], sampleData.squareDim[1]), dtype=np.float32)
        self.chanImages = np.zeros((sampleData.numChannels, sampleData.finalDim[0], sampleData.finalDim[1]), dtype=np.float32)
        self.sumImage = np.zeros((sampleData.finalDim), dtype=np.float32)
        self.percMeasured = 0
        self.iteration = 0
        
        #If post-processing, link to the final sampled mask
        if sampleData.postFlag: self.mask = sampleData.mask
    
    #Measure selected locations, computing reconstructions and E/RD as applicable for determination of future sampling locations
    def performMeasurements(self, sampleData, tempScanData, result, newIdxs, model, cValue, updateRD):
        
        #Ensure newIdxs are indexible in 2 dimensions and update mask; post-processing will send an empty set
        if not sampleData.postFlag:
            newIdxs = np.atleast_2d(newIdxs)
            self.mask[newIdxs[:,0], newIdxs[:,1]] = 1
        
        #Update which physical positions have not yet been measured for new measurement location(s) selection
        #Should probably be a faster method that can remove newIdxs from the existing list, but nothing off-the-shelf available
        if sampleData.useMaskFOV: self.unMeasuredIdxs = np.transpose(np.where((self.mask==0) & (sampleData.maskFOV==1)))
        else: self.unMeasuredIdxs = np.transpose(np.where(self.mask==0))
        
        #If updating using recon data, then back up the prior square mask in advanced
        if updateRD: prevSquareMask = copy.deepcopy(self.squareMask)
        
        #For DESI, resize the mask to enforce square pixels, otherwise the square mask is the same as the mask
        if sampleData.sampleType == 'DESI': self.squareMask = resize(self.mask, tuple(sampleData.squareDim), order=0)
        else: self.squareMask = copy.deepcopy(self.mask)
        
        #If updating using recon data identify new update locations in square dimensions, otherwise set to None
        if updateRD: updateLocations = np.argwhere(self.squareMask-prevSquareMask)
        else: updateLocations = []
        
        #If not just updating the RD, either get measurement values from equipment or the stored ground-truth
        if not updateRD:
        
            #Update the actual measurement iteration counter and percent measured
            self.iteration += 1
            self.percMeasured = (np.sum(self.mask)/sampleData.area)*100
            
            #If not simulation or post-processing, then read measurements into sampleData from equipment
            if not sampleData.simulationFlag and not sampleData.postFlag:
                if sampleData.scanMethod == 'linewise' and self.iteration<=len(startLinePositions): print('Writing UNLOCK')
                elif self.iteration==1: print('Writing UNLOCK')
                else: print('\nWriting UNLOCK')
                with open(dir_ImpDataFinal + 'UNLOCK', 'w') as filehandle: _ = [filehandle.writelines(str(tuple([pos[0]+1, (pos[1]*sampleData.scanRate)/sampleData.acqRate]))+'\n') for pos in newIdxs.tolist()]
                if sampleData.unorderedNames and impModel and scanMethod == 'linewise': sampleData.physicalLineNums[len(sampleData.physicalLineNums.keys())+1] = int(newIdxs[0][0])
                sampleData.mask = self.mask
                equipWait()
                avgTimeFileLoad = sampleData.readScanData(newIdxs)
                result.avgTimesFileLoad.append(avgTimeFileLoad)
            if not sampleData.postFlag:
                self.chanImages[:, newIdxs[:,0], newIdxs[:,1]] = sampleData.chanImages[:, newIdxs[:,0], newIdxs[:,1]]
                self.sumImage[newIdxs[:,0], newIdxs[:,1]] = sampleData.sumImage[newIdxs[:,0], newIdxs[:,1]]
                self.progMap[newIdxs[:,0], newIdxs[:,1]] = self.iteration
            else:
                self.chanImages = copy.deepcopy(sampleData.chanImages)*self.mask
                self.sumImage = copy.deepcopy(sampleData.sumImage)*self.mask
        
        #Otherwise, (if not an oracle or c value optimization run) set the reconstruction data as having been 'measured'
        else:
            if not sampleData.oracleFlag and not sampleData.bestCFlag:
                self.chanImages[:, newIdxs[:,0], newIdxs[:,1]] = self.chanReconImages[:, newIdxs[:,0], newIdxs[:,1]]
                self.sumImage[newIdxs[:,0], newIdxs[:,1]] = self.sumReconImage[newIdxs[:,0], newIdxs[:,1]]
        
        #If not just updating the RD, or using a SLADS model to update the ERD
        if not updateRD or (('SLADS' in erdModel) and not sampleData.bestCFlag and not sampleData.oracleFlag):
        
            #Extract measured and unmeasured locations, considering FOV mask if applicable
            tempScanData.squareMeasuredIdxs = np.transpose(np.where(self.squareMask==1))
            if sampleData.useMaskFOV: tempScanData.squareUnMeasuredIdxs = np.transpose(np.where((self.squareMask==0) & (sampleData.squareMaskFOV==1)))
            else: tempScanData.squareUnMeasuredIdxs = np.transpose(np.where(self.squareMask==0))
            
            #Determine neighbor information for unmeasured locations
            if len(tempScanData.squareUnMeasuredIdxs) > 0: findNeighbors(tempScanData)
            else: tempScanData.neighborIndices, tempScanData.neighborWeights, tempScanData.neighborDistances = [], [], []
        
        #If not just updating the RD
        if not updateRD:
            t0_compute = time.perf_counter()
            
            #Resize DESI data to square dimensions for processing
            if sampleData.sampleType == 'DESI':
                self.squareSumReconImage = resize(self.sumImage, tuple(sampleData.squareDim), order=0)
                self.squareChanReconImages = np.moveaxis(resize(np.moveaxis(self.chanImages, 0, -1), tuple(sampleData.squareDim), order=0), -1, 0)
            else: 
                self.squareSumReconImage = self.sumImage
                self.squareChanReconImages = self.chanImages
            
            #GLANDS computes both ERD and reconstruction results
            if erdModel == 'GLANDS':
                
                sys.exit('\nError - ERD and reconstruction computation not yet implemented.')
                
                #Prepare data for processing
                sumImageStack = prepareInput(self.squareSumReconImage, self.squareMask, sampleData.squareOpticalImage)
                inputStack = prepareInput(self.squareChanReconImages, self.squareMask, sampleData.squareOpticalImage)
                
                #Inference reconstruction for the sum image; if this fails, then the channels are definitely not going to process through the network
                #try: self.squareSumReconImage = ray.get(model.generate.remote(makeCompatible(sumImageStack), True)).copy()
                #except: sampleData.OOM_multipleChannels, sampleData.OOM_singleChannel = True, True
                
                #Try inferencing model results for all target channels
                if not sampleData.OOM_multipleChannels:
                    try: 
                        #self.squareChanReconImages, self.squareERDs = ray.get(model.generate.remote(makeCompatible(inputStack))).copy()
                        self.squareChanReconImages, self.squareERDs = self.squareChanReconImages.copy(), self.squareERDs.copy()
                    except: 
                        sampleData.OOM_multipleChannels = True
                        if (len(gpus) > 0): print('\nWarning - Could not inference GLANDS for all channels of sample '+sampleData.name+' simultaneously on system GPU; will try processing channels iteratively.')
                        if (len(gpus) == 0): print('\nWarning - Could not inference GLANDS for all channels of sample '+sampleData.name+' simultaneously on system; will try processing channels iteratively.')
                
                #If multiple channels causes an OOM, then try running each channel's data through on its own
                if sampleData.OOM_multipleChannels and not sampleData.OOM_singleChannel: 
                    try: 
                        for chanNum in range(0, len(self.squareERDs)):
                            #self.squareChanReconImages[chanNum], self.squareERDs[chanNum] = ray.get(model.generate.remote(makeCompatible(inputStack[chanNum]))).copy()
                            self.squareChanReconImages[chanNum], self.squareERDs[chanNum] = self.squareChanReconImages[chanNum].copy(), self.squareERDs[chanNum].copy()
                    except: sampleData.OOM_singleChannel = True
                
                #If an OOM occured for both mutiple and single channel inferencing, then exit; need to either restart program with no GPUs, or there isn't enough system RAM
                if sampleData.OOM_multipleChannels and sampleData.OOM_singleChannel and (len(gpus) > 0): sys.exit('\nError - Sample '+sampleData.name+' dimensions are too high for GLANDS inferencing on system GPU; please try disabling the GPU in the CONFIG.')
                if sampleData.OOM_multipleChannels and sampleData.OOM_singleChannel and (len(gpus) == 0): sys.exit('\nError - Sample '+sampleData.name+' dimensions are too high for GLANDS inferencing on this system by the loaded model.')
            
            #SLADS/DLADS compute reconstruction data
            else:
                self.squareSumReconImage = computeReconIDW(self.squareSumReconImage, tempScanData)
                self.squareChanReconImages = computeReconIDW(self.squareChanReconImages, tempScanData)
            
            #Resize DESI reconstruction results back to physical dimensions and copy back original measured values to reconstructions
            if sampleData.sampleType == 'DESI':
                self.sumReconImage = resize(self.squareSumReconImage, tuple(sampleData.finalDim), order=0)
                self.chanReconImages = np.moveaxis(resize(np.moveaxis(self.squareChanReconImages , 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)
                self.measuredIdxs = np.transpose(np.where(self.mask==1))
                self.chanReconImages[:, self.measuredIdxs[:,0], self.measuredIdxs[:,1]] = self.chanImages[:, self.measuredIdxs[:,0], self.measuredIdxs[:,1]]
                self.sumReconImage[self.measuredIdxs[:,0], self.measuredIdxs[:,1]] = self.sumImage[self.measuredIdxs[:,0], self.measuredIdxs[:,1]]
            else:
                self.sumReconImage = self.squareSumReconImage
                self.chanReconImages = self.squareChanReconImages
            
            t1_compute = time.perf_counter()
            if erdModel == 'GLANDS': result.avgTimesComputeIter.append(t1_compute-t0_compute)
            else: result.avgTimesComputeRecon.append(t1_compute-t0_compute)
        
        #Compute feature information for for training/utilizing SLADS models
        if (sampleData.datagenFlag or ('SLADS' in erdModel) and not sampleData.bestCFlag and not sampleData.oracleFlag) and len(tempScanData.squareUnMeasuredIdxs) > 0: 
            t0_computePoly = time.perf_counter()
            self.polyFeatures = [computePolyFeatures(sampleData, tempScanData, squareChanReconImage) for squareChanReconImage in self.squareChanReconImages]
            t1_computePoly = time.perf_counter()
            polyComputeTime = t1_computePoly-t0_computePoly
        else: polyComputeTime = 0
        
        #If every location has been scanned all E/RD values are zero
        if len(tempScanData.squareUnMeasuredIdxs) == 0:
            if sampleData.oracleFlag or sampleData.bestCFlag: 
                self.RD = np.zeros(sampleData.finalDim, dtype=np.float32)
                self.squareRD = np.zeros(sampleData.squareDim, dtype=np.float32)
                self.squareRDs = np.zeros((sampleData.numChannels, sampleData.squareDim[0], sampleData.squareDim[1]), dtype=np.float32)
                self.squareRDValues = self.squareRDs[:, tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]]
                self.squareERD = self.squareRD
            else: self.squareERD = np.zeros(sampleData.squareDim, dtype=np.float32)
        
        #If the ground-truth data is known for training or oracle runs then compute the RDPPs and resulting RD
        elif sampleData.oracleFlag or sampleData.bestCFlag:
        
            #If this is a full measurement step, (i.e. whenever the reconstruction(s) are updated) compute the new RDPP
            if not updateRD: self.RDPPs = computeRDPPs(sampleData.squareChanImages, self.squareChanReconImages)
            
            #Compute/Update the RD and use it in place of an ERD
            t0_computeRD = time.perf_counter()
            computeRD(self, sampleData, tempScanData, cValue, updateLocations)
            t1_computeRD = time.perf_counter()
            if not updateRD: result.avgTimesComputeRD.append(t1_computeRD-t0_computeRD)
            self.squareRDValues = self.squareRDs[:, tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]]
            self.squareERD, self.squareERDs = self.squareRD, self.squareRDs
            self.ERD, self.ERDs = self.RD, self.RDs
        
        #If using SLADS/DLADS and there are unmeasured locations left and the ground-truth data isn't known, compute and process the ERD
        elif erdModel != 'GLANDS': 
            t0_computeERD = time.perf_counter()
            
            #Prepare model inputs
            if 'DLADS' in erdModel: inputStack = prepareInput(self.squareChanReconImages, self.squareMask, sampleData.squareOpticalImage)
            
            #Compute the ERD with the prescribed model; if configured to, only use a single channel
            if not chanSingle:
            
                if 'SLADS' in erdModel: 
                    for chanNum in range(0, len(self.squareERDs)): self.squareERDs[chanNum, tempScanData.squareUnMeasuredIdxs[:, 0], tempScanData.squareUnMeasuredIdxs[:, 1]] = ray.get(model.generate.remote(self.polyFeatures[chanNum])).copy()
                elif 'DLADS' in erdModel: 
                    
                    #First try inferencing all target channels at the same time 
                    if not sampleData.OOM_multipleChannels:
                        try: 
                            self.squareERDs = ray.get(model.generate.remote(inputStack)).copy()
                        except: 
                            sampleData.OOM_multipleChannels = True
                            print('\nWarning - Could not inference ERD for all channels of sample: '+sampleData.name+' simultaneously; will try processing channels iteratively.')
                    
                    #If multiple channels causes an OOM, then try running each channel through on its own
                    if sampleData.OOM_multipleChannels and not sampleData.OOM_singleChannel:                         
                        try: self.squareERDs = np.concatenate([ray.get(model.generate.remote(inputStack[chanNum])).copy() for chanNum in range(0, len(self.squareERDs))])
                        except: sampleData.OOM_singleChannel = True
                    
                    #If an OOM occured for both mutiple and single channel inferencing, then exit; need to either restart program with no GPUs, or there isn't enough system RAM
                    if sampleData.OOM_multipleChannels and sampleData.OOM_singleChannel:
                        if len(gpus) > 0: sys.exit('\nError - Dimensions of sample: '+sampleData.name+' are too high for the ERD to be inferenced on system GPU; please try disabling the GPU in the CONFIG.')
                        else: sys.exit('\nError - Sample '+sampleData.name+' dimensions are too high for the ERD to be inferenced on this system by the loaded model.')
            
            else:
                if 'SLADS' in erdModel: 
                    ERDValues = ray.get(model.generate.remote(self.polyFeatures[0])).copy()
                    self.squareERDs[:, tempScanData.squareUnMeasuredIdxs[:, 0], tempScanData.squareUnMeasuredIdxs[:, 1]] = np.repeat(np.expand_dims(ERDValues, 0), len(self.squareERDs), 0)
                elif 'DLADS' in erdModel: 
                    self.squareERDs = np.repeat(ray.get(model.generate.remote(inputStack[0])).copy(), len(self.squareERDs), 0)
            
            t1_computeERD = time.perf_counter()
            result.avgTimesComputeERD.append((t1_computeERD-t0_computeERD)+polyComputeTime)
    
        #In square dimensionality, set any invalid values to 0 and average results to form a single ERD
        self.squareERDs = np.nan_to_num(self.squareERDs, nan=0, posinf=0, neginf=0)
        self.squareERD = np.mean(self.squareERDs, axis=0)
        
        #Resize/Average ERDs as needed according to data type
        if sampleData.sampleType == 'DESI':
            self.ERDs = np.moveaxis(resize(np.moveaxis(self.squareERDs , 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)
            self.ERD = np.mean(self.ERDs, axis=0)
        else:
            self.ERD = self.squareERD
            self.ERDs = self.squareERDs
        
        #Duplicate the per-channel ERDs for processing, masking to just the unmeasured locations
        self.processedERDs = copy.deepcopy(self.ERDs)*(1-self.mask)
        
        #Mask the ERDs by the FOV foreground if applicable
        if sampleData.useMaskFOV: self.processedERDs *= sampleData.maskFOV
        
        #Mask out visited lines if linewise and line revisiting is disabled
        if sampleData.scanMethod == 'linewise' and not sampleData.lineRevist: self.processedERDs[:, np.where(np.sum(self.mask, axis=1)>0)[0], :] = 0
        
        #Average across all channels for measurement location selection
        self.processedERD = np.mean(self.processedERDs, axis=0)
        
        #Rescale to force min/max values
        minValue, maxValue = np.min(self.processedERDs, axis=(1,2)), np.max(self.processedERDs, axis=(1,2))
        self.processedERDs = np.moveaxis(divideSafe((np.moveaxis(self.processedERDs, 0, -1)-minValue), (maxValue-minValue)), -1, 0)
        minValue, maxValue = np.min(self.processedERD), np.max(self.processedERD)
        self.processedERD = divideSafe((self.processedERD-minValue), (maxValue-minValue))

#Sample scanning progress and final results processing
class Result:
    def __init__(self, sampleDataNum, sampleData, dir_Results, cValue):
        self.startTime = time.perf_counter()
        self.finalTime = time.perf_counter()
        self.sampleDataNum = sampleDataNum
        self.dir_Results = dir_Results
        self.cValue = cValue
        self.samples = []
        self.cSelectionList = []
        self.percsMeasured = []
        self.avgTimesComputeRD = []
        self.avgTimesComputeERD = []
        self.avgTimesComputeRecon = []
        self.avgTimesComputeIter = []
        self.avgTimesFileLoad = []
        self.lastMask = None
        self.avgTimeComputeRD = 0
        self.avgTimeComputeERD = 0
        self.avgTimeComputeRecon = 0
        self.avgTimeFileLoad = 0
        self.avgTimeComputeIter = 0
        
        #If there is to be a results directory, then ensure it is setup
        if self.dir_Results != None:

            #Setup/clean base sample directory
            self.dir_sampleResults = self.dir_Results + sampleData.name + os.path.sep
            if os.path.exists(self.dir_sampleResults): shutil.rmtree(self.dir_sampleResults)
            os.makedirs(self.dir_sampleResults)
            
            #Prepare subdirectories; for frames and videos of channel progressions; #Note: cannot recall why videos/animations was disabled for post, so re-enabled...
            self.dir_chanProgression = self.dir_sampleResults + 'Channels' + os.path.sep
            os.makedirs(self.dir_chanProgression)
            self.dir_chanProgressions = [self.dir_chanProgression + str(sampleData.chanValues[chanNum]) + os.path.sep for chanNum in range(0, len(sampleData.chanValues))]
            for dir_chanProgressionsub in self.dir_chanProgressions: 
                try: os.makedirs(dir_chanProgressionsub)
                except: print('Folder already exists')
            self.dir_progression = self.dir_sampleResults + 'Progression' + os.path.sep
            os.makedirs(self.dir_progression)
            #if not sampleData.postFlag:
            self.dir_videos = self.dir_sampleResults + 'Videos' + os.path.sep
            os.makedirs(self.dir_videos)
            
        #MSI Specific; if this is a simulation with live output enabled, then load all images data
        if sampleData.dataMSI and sampleData.simulationFlag and sampleData.liveOutputFlag:
            
            #If operating in parallel, create actors for reconstruction and load portions of the data into each, otherwise load data into main memory
            if parallelization and erdModel != 'GLANDS':
                self.recon_Actors = [Recon_Actor.remote(indexes, sampleData.sampleType, sampleData.squareDim, sampleData.finalDim, sampleData.allImagesMin, sampleData.allImagesMax, sampleData.allImagesPath, sampleData.squareAllImagesPath, sampleData.squareOpticalImage, erdModel) for indexes in np.array_split(np.arange(0, len(sampleData.mzFinal)), numberCPUS)]
            else:
                sampleData.allImagesFile = h5py.File(sampleData.allImagesPath, 'a')
                sampleData.allImages = sampleData.allImagesFile['allImages'][:]
        
    def update(self, sample, sampleData, completedRunFlag):
    
        #Update copies of the measurement mask, progression map, and percent measured (needed for reference in future iterations and termination criteria)
        self.lastMask = copy.deepcopy(sample.mask)
        self.lastProgMap = copy.deepcopy(sample.progMap)
        self.lastPercMeasured = copy.deepcopy(sample.percMeasured)
        
        #If optimizing c, then don't store sample data from the first iteration, since it was not used to determine initial measurement locations
        if sampleData.bestCFlag and sample.iteration == 1: return
        
        #Update the percentage of FOV measured
        self.percsMeasured.append(copy.deepcopy(sample.percMeasured))
        
        #If outputs should be produced at every update step, then do so, determining related metrics as needed
        if sampleData.liveOutputFlag: 
            if sampleData.simulationFlag: self.extractSimulationData(sample, sampleData)
            visualizeStep(sample, sampleData, self.dir_progression, self.dir_chanProgressions)
        
        #Save a copy of the measurement step for later evaluation
        self.samples.append(copy.deepcopy(sample))
        
        #When applicable, save the physicalLineNums.csv, measuredMask.csv, and progressMap.csv to the same folder as the scanned MSI files and the results folder; otherwise just save to results
        if (completedRunFlag or saveIterationFlag) and self.dir_Results != None: 
            if sampleData.impFlag: #not sampleData.simulationFlag and not sampleData.postFlag:
                if sampleData.unorderedNames: np.savetxt(dir_ImpDataFinal+'physicalLineNums.csv', np.asarray(list(sampleData.physicalLineNums.items())), delimiter=',', fmt='%d')
                np.savetxt(dir_ImpDataFinal+'measuredMask.csv', self.lastMask, delimiter=',', fmt='%d')
                if len(self.lastProgMap.shape)>0: np.savetxt(dir_ImpDataFinal+'progressMap.csv', np.nan_to_num(self.lastProgMap, nan=-1), delimiter=',', fmt='%d')
            if sampleData.unorderedNames: np.savetxt(self.dir_sampleResults+'physicalLineNums.csv', np.asarray(list(sampleData.physicalLineNums.items())), delimiter=',', fmt='%d')
            np.savetxt(self.dir_sampleResults+'measuredMask.csv', self.lastMask, delimiter=',', fmt='%d')
            if len(self.lastProgMap.shape)>0: np.savetxt(self.dir_sampleResults+'progressMap.csv', np.nan_to_num(self.lastProgMap, nan=-1), delimiter=',', fmt='%d')
        
        #Store the final scan time if run has completed
        if completedRunFlag: self.finalTime = time.perf_counter()-self.startTime
    
    #For a given measurement step find NRMSE/SSIM/PSNR of reconstructions; if applicable: compute the RD and/or find metrics for ERD
    def extractSimulationData(self, sample, sampleData, lastReconOnly=False):
        
        #Create a segmented storage object for variables that must be referenced
        tempScanData = TempScanData()
        
        #Extract measured and unmeasured locations for the measured mask
        tempScanData.squareMeasuredIdxs = np.transpose(np.where(sample.squareMask==1))
        if sampleData.useMaskFOV: tempScanData.squareUnMeasuredIdxs = np.transpose(np.where((sample.squareMask==0) & (sampleData.squareMaskFOV==1)))
        else: tempScanData.squareUnMeasuredIdxs = np.transpose(np.where(sample.squareMask==0))
        
        #Determine neighbor information for unmeasured locations
        if len(tempScanData.squareUnMeasuredIdxs) > 0: findNeighbors(tempScanData)
        else: tempScanData.neighborIndices, tempScanData.neighborWeights, tempScanData.neighborDistances = [], [], []
        
        #Find NRMSE/SSIM/PSNR scores for channel reconstructions
        if not lastReconOnly:
            sample.chanImagesNRMSEList, sample.chanImagesSSIMList, sample.chanImagesPSNRList = [], [], []
            for index in range(0, len(sampleData.chanImages)): 
                score_NRMSE, score_SSIM, score_PSNR = compareImages(sampleData.chanImages[index], sample.chanReconImages[index], sampleData.chanImagesMin[index], sampleData.chanImagesMax[index])
                sample.chanImagesNRMSEList.append(score_NRMSE)
                sample.chanImagesSSIMList.append(score_SSIM)
                sample.chanImagesPSNRList.append(score_PSNR)
            sample.sumImageNRMSE, sample.sumImageSSIM, sample.sumImagePSNR = compareImages(sampleData.sumImage, sample.sumReconImage, np.min(sampleData.sumImage), np.max(sampleData.sumImage))
        
        #MSI specific; if enabled then perform and evaluate reconstructions over the whole spectrum for the data known at the given measurement step
        if (sampleData.allChanEvalFlag or lastReconOnly) and sampleData.dataMSI:
            
            #If operating in parallel, utilize actors created in the complete() method, or at initialization for live simulation; for serial, could vectorize, but extremely RAM intensive and usable batch size is system specific
            #Consider adding a batch option for GLANDS as parallelized reconsturciton is not an option
            if parallelization and erdModel != 'GLANDS':
                tempScanData_id, squareMask_id, mask_id = ray.put(tempScanData), ray.put(sample.squareMask), ray.put(sample.mask)
                _ = ray.get([recon_Actor.computeRecon.remote(tempScanData_id, squareMask_id, mask_id) for recon_Actor in self.recon_Actors])
                _ = ray.get([recon_Actor.computeMetrics.remote() for recon_Actor in self.recon_Actors])
                if not lastReconOnly:
                    sample.allImagesNRMSEList = np.concatenate([ray.get(recon_Actor.getNRMSE.remote()).copy() for recon_Actor in self.recon_Actors])
                    sample.allImagesSSIMList = np.concatenate([ray.get(recon_Actor.getSSIM.remote()).copy() for recon_Actor in self.recon_Actors])
                    sample.allImagesPSNRList = np.concatenate([ray.get(recon_Actor.getPSNR.remote()).copy() for recon_Actor in self.recon_Actors])
                del tempScanData_id, squareMask_id, mask_id
                _ = gc.collect()
            else:
                #Extract the sparsely measured images in square dimensionality
                if sampleData.sampleType == 'DESI': self.reconImages = sampleData.squareAllImages*sample.squareMask
                else: self.reconImages = sampleData.allImages*sample.squareMask
                
                #Compute reconstructions
                if erdModel == 'GLANDS':
                    sys.exit('\nError - Reconstruction for GLANDS has not yet been implemented.')
                    #inputStack = prepareInput(self.reconImages, samplesquareMask, sampleData.squareOpticalImage)
                    #self.reconImages = np.array([ray.get(model.generate.remote(makeCompatible(self.inputStack[index]), True)).copy() for index in range(0, len(self.reconImages))], dtype=np.float32)
                else:
                    for index in range(0, len(self.reconImages)): self.reconImages[index] = computeReconIDW(self.reconImages[index], tempScanData)
                
                #Resize DESI data back to physical dimensions and copy back the original measured values to reconstructions; creating new holding array for resized results and looping is needed for memory efficiency
                if self.sampleType == 'DESI': 
                    resizedReconImages = np.zeros((len(self.reconImages), sampleData.finalDim[0], sampleData.finalDim[1]))
                    for index in range(0, len(self.reconImages)): resizedReconImages[index] = resize(self.reconImages[index], tuple(sampleData.finalDim), order=0)
                    self.reconImages = resizedReconImages
                    del resizedReconImages
                    self.reconImages = (self.reconImages*(1-sample.mask)) + (sampleData.allImages*sample.mask)
                
                #Quantify reconstruction quality
                if not lastReconOnly:
                    sample.allImagesNRMSEList, sample.allImagesSSIMList, sample.allImagesPSNRList = [], [], []
                    for index in range(0, len(sampleData.allImages)):
                        score_NRMSE, score_SSIM, score_PSNR = compareImages(sampleData.allImages[index], self.reconImages[index], sampleData.allImagesMin[index], sampleData.allImagesMax[index])
                        sample.allImagesNRMSEList.append(score_NRMSE)
                        sample.allImagesSSIMList.append(score_SSIM)
                        sample.allImagesPSNRList.append(score_PSNR)
        
        #Otherwise assume all images results are the same as for targeted channels; i.e. all channels were targeted
        elif not lastReconOnly:
            sample.allImagesNRMSEList = sample.chanImagesNRMSEList
            sample.allImagesSSIMList = sample.chanImagesSSIMList
            sample.allImagesPSNRList = sample.chanImagesPSNRList
        
        #GLANDS does not use a ground-truth RD
        if erdModel != 'GLANDS':
        
            #Prior to and for model training there is RD, but no ERD
            if sampleData.simulationFlag and not sampleData.trainFlag and not lastReconOnly:
                
                #Compute RD; if every location has been scanned all positions are zero
                if len(tempScanData.squareUnMeasuredIdxs) == 0: 
                    sample.squareRD = np.zeros(sampleData.squareDim, dtype=np.float32)
                    sample.RD = np.zeros(sampleData.finalDim, dtype=np.float32)
                else: 
                    sample.RDPPs = computeRDPPs(sampleData.squareChanImages, sample.squareChanReconImages)
                    computeRD(sample, sampleData, tempScanData, self.cValue, [])
                
                #Determine NRMSE/SSIM/PSNR between averaged RD and ERD
                sample.ERD_NRMSE, sample.ERD_SSIM, sample.ERD_PSNR = compareImages(sample.squareRD, sample.squareERD, np.min(sample.squareRD), np.max(sample.squareRD))
                
                #Resize RD(s) for final visualizations; has to be done here for live output case, but in complete() method otherwise
                if sampleData.liveOutputFlag: self.resizeRD(sample, sampleData)
        
    #Resize RD(s) for final visualization if DESI, otherwise set variable name 
    def resizeRD(self, sample, sampleData):
        if sampleData.sampleType == 'DESI':
            sample.RD = resize(sample.squareRD, tuple(sampleData.finalDim), order=0)*(1-sample.mask)
            sample.RDs = np.moveaxis(resize(np.moveaxis(sample.squareRDs , 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)*(1-sample.mask)
        else:
            sample.RD = sample.squareRD
            sample.RDs = sample.squareRDs
            
    #Generate visualizations/metrics as needed at the end of scanning
    def complete(self, sampleData):
        
        #If the data was loaded during initialization of the sample data, pull the stored avg. file read time, otherwise compute value
        if sampleData.postFlag or sampleData.simulationFlag: self.avgTimeFileLoad = sampleData.avgTimeFileLoad
        elif len(self.avgTimesFileLoad) > 0: self.avgTimeFileLoad = np.nanmean(self.avgTimesFileLoad)
        
        #If applicable, compute average computation times
        if len(self.avgTimesComputeRecon) > 0: self.avgTimeComputeRecon = np.mean(self.avgTimesComputeRecon)
        if len(self.avgTimesComputeERD) > 0: self.avgTimeComputeERD = np.mean(self.avgTimesComputeERD)
        if len(self.avgTimesComputeRD) > 0: self.avgTimeComputeRD = np.mean(self.avgTimesComputeRD)
        if erdModel != 'GLANDS' and (len(self.avgTimesComputeRecon) > 0) and (len(self.avgTimesComputeERD) > 0): self.avgTimesComputeIter = self.avgTimesComputeRecon + self.avgTimesComputeERD
        if len(self.avgTimesComputeIter) > 0: self.avgTimeComputeIter = np.mean(self.avgTimesComputeIter)
        
        #If performing a benchmark where processing is not needed, return before processing
        if benchmarkNoProcessing: return
        
        #Make sure samples is writable
        self.samples = copy.deepcopy(self.samples)
        
        #If this is an MSI sample
        if sampleData.dataMSI:
        
            #If all channel reconstructions are needed, then setup actors if in parallel, or load data into main memory
            #For GLANDS will have to do reconstructions and evaluations outside of the actor...
            #Tracer()
            if sampleData.allChanEvalFlag or (sampleData.imzMLExportFlag and not sampleData.trainFlag):
                if parallelization and erdModel != 'GLANDS':
                    self.recon_Actors = [Recon_Actor.remote(indexes, sampleData.sampleType, sampleData.squareDim, sampleData.finalDim, sampleData.allImagesMin, sampleData.allImagesMax, sampleData.allImagesPath, sampleData.squareAllImagesPath, sampleData.squareOpticalImage, erdModel) for indexes in np.array_split(np.arange(0, len(sampleData.mzFinal)), numberCPUS)]
                else:
                    sampleData.allImagesFile = h5py.File(sampleData.allImagesPath, 'r')
                    sampleData.allImages = sampleData.allImagesFile['allImages']
                    if sampleData.sampleType == 'DESI':
                        sampleData.squareAllImagesFile = h5py.File(sampleData.squareAllImagesPath, 'r')
                        sampleData.squareAllImages = sampleData.squareAllImagesFile['squareAllImages']
        
        #Extract metrics for samples if a simulation, and neither already done live, nor creating samples for training
        if (sampleData.simulationFlag and not sampleData.liveOutputFlag and not sampleData.datagenFlag):
            for sample in tqdm(self.samples, desc='RD/Metrics Extraction', leave=False, ascii=asciiFlag): self.extractSimulationData(sample, sampleData)
        
        #If not evaluating all channels, but the final reconstructions for all channels are still to be generated
        if sampleData.imzMLExportFlag and not sampleData.allChanEvalFlag: self.extractSimulationData(self.samples[-1], sampleData, imzMLExport)
        
        #If this is an MSI sample
        if sampleData.dataMSI:
            
            #If exporting final reconstruction data to .imzML
            if sampleData.imzMLExportFlag:
                
                #Set the coordinates to save values for
                coordinates = list(map(tuple, list(np.ndindex(tuple(sampleData.finalDim)))))
                
                #Export all measured, reconstructed data in .imzML format
                if parallelization and erdModel != 'GLANDS': self.reconImages = np.concatenate([ray.get(recon_Actor.getReconImages.remote()).copy() for recon_Actor in self.recon_Actors])
                writer = ImzMLWriter(self.dir_sampleResults+sampleData.name+'_reconstructed', intensity_dtype=np.float32, mz_dtype=np.float32, spec_type='profile', mode='processed')
                _  = [writer.addSpectrum(sampleData.mzFinal, self.reconImages[:, coord[0], coord[1]], (coord[1]+1, coord[0]+1)) for coord in coordinates]
                writer.close()
                del self.reconImages
                _ = gc.collect()
                
                #Export the equivalent ground-truth measured data here to .imzML format if needed
                #allImages = np.concatenate([ray.get(recon_Actor.getAllImages.remote()).copy() for recon_Actor in self.recon_Actors])
                #writer = ImzMLWriter(self.dir_sampleResults+sampleData.name+'_groundTruth', intensity_dtype=np.float32, mz_dtype=np.float32, spec_type='profile', mode='processed')
                #for coord in coordinates: writer.addSpectrum(sampleData.mzFinal, allImages[:, coord[0], coord[1]], (coord[1]+1, coord[0]+1))
                #writer.close()
                #del allImages
                #_ = gc.collect()
            
            #If all channel evaluation or imzMLExportFlag, close all images file reference(s)
            if sampleData.allChanEvalFlag or sampleData.imzMLExportFlag:
                if parallelization and erdModel != 'GLANDS':
                    _ = [ray.get(recon_Actor.closeAllImages.remote()) for recon_Actor in self.recon_Actors]
                    self.recon_Actors.clear()
                    del self.recon_Actors
                    resetRay(numberCPUS)
                else:
                    sampleData.allImagesFile.close()
                    del sampleData.allImages, sampleData.allImagesFile
                    if sampleData.sampleType == 'DESI':
                        sampleData.squareAllImagesFile.close()
                        del sampleData.squareAllImages, sampleData.squareAllImagesFile
                    _ = gc.collect()
        
        #If this is a simulation, not for training database generation, then summarize NRMSE/SSIM/PSNR scores across all measurement steps
        if sampleData.simulationFlag and not sampleData.datagenFlag:
            self.chanAvgNRMSEList = [np.nanmean(sample.chanImagesNRMSEList) for sample in self.samples]
            self.chanAvgSSIMList = [np.nanmean(sample.chanImagesSSIMList) for sample in self.samples]
            self.chanAvgPSNRList = [np.nanmean(sample.chanImagesPSNRList) for sample in self.samples]
            self.sumImageNRMSEList = [sample.sumImageNRMSE for sample in self.samples]
            self.sumImageSSIMList = [sample.sumImageSSIM for sample in self.samples]
            self.sumImagePSNRList = [sample.sumImagePSNR for sample in self.samples]
            
            #Compute all channel results if applicable
            if sampleData.allChanEvalFlag and sampleData.dataMSI:
                self.allAvgNRMSEList = [np.nanmean(sample.allImagesNRMSEList) for sample in self.samples]
                self.allAvgSSIMList = [np.nanmean(sample.allImagesSSIMList) for sample in self.samples]
                self.allAvgPSNRList = [np.nanmean(sample.allImagesPSNRList) for sample in self.samples]
        
        #If ERD and RD were computed (i.e., when not a training run, nor GLANDS) summarize ERD NRMSE/SSIM/PSNR scores
        if erdModel != 'GLANDS' and sampleData.simulationFlag and not sampleData.trainFlag: 
            self.ERD_NRMSEList = [sample.ERD_NRMSE for sample in self.samples]
            self.ERD_SSIMList = [sample.ERD_SSIM for sample in self.samples]
            self.ERD_PSNRList = [sample.ERD_PSNR for sample in self.samples]
        
        #Do not generate visuals for c value optimization or in training if visualizeTrainingData disabled
        if not sampleData.bestCFlag and ((sampleData.datagenFlag and visualizeTrainingData) or not sampleData.datagenFlag): 
            
            #generate visualizations if they were not created during operation
            if not sampleData.liveOutputFlag:
                
                #For non-GLANDS models, resize RDs
                if erdModel != 'GLANDS':
                    for sample in self.samples: self.resizeRD(sample, sampleData)
                
                if parallelization: 
                    
                    #Setup an actor to hold global sampling progress across multiple processes
                    samplingProgress_Actor = SamplingProgress_Actor.remote()
                    
                    #Setup visualization jobs and determine total amount of work that is going to be done
                    samples_id = ray.put(self.samples)
                    futures = [visualizeStep_parhelper.remote(samples_id, indexes, sampleData, self.dir_progression, self.dir_chanProgressions, samplingProgress_Actor) for indexes in np.array_split(np.arange(0, len(self.samples)), numberCPUS)]
                    maxProgress = len(self.samples)
                    
                    #Initialize a global progress bar and start parallel visualization operations
                    pbar = tqdm(total=maxProgress, desc = 'Visualizing', leave=False, ascii=asciiFlag)
                    pbar.n = 0
                    pbar.refresh()
                                        
                    while len(futures):
                        _, futures = ray.wait(futures)
                        pbar.n = np.clip(round(copy.deepcopy(ray.get(samplingProgress_Actor.getCurrent.remote())),0), 0, maxProgress)
                        pbar.refresh()
                        time.sleep(0.1)
                    pbar.n = maxProgress
                    pbar.refresh()
                    pbar.close()
                    del samples_id, samplingProgress_Actor, futures
                    resetRay(numberCPUS)
                else: 
                    _ = [visualizeStep(sample, sampleData, self.dir_progression, self.dir_chanProgressions) for sample in tqdm(self.samples, desc='Visualizing', leave=False, ascii=asciiFlag)]

            #Combine total progression and individual channel images into animations
            dataFileNames = natsort.natsorted(glob.glob(self.dir_progression + 'progression_*.tiff'))
            height, width, layers = cv2.imread(dataFileNames[0]).shape
            animation = cv2.VideoWriter(self.dir_videos + 'progression.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (width, height))
            for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
            animation.release()
            animation = None
            for chanNum in tqdm(range(0, len(sampleData.chanValues)), desc='Channel Videos', leave = False, ascii=asciiFlag): 
                dataFileNames = natsort.natsorted(glob.glob(self.dir_chanProgressions[chanNum] + 'progression_*.tiff'))
                height, width, layers = cv2.imread(dataFileNames[0]).shape
                animation = cv2.VideoWriter(self.dir_videos + str(sampleData.chanValues[chanNum]) + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (width, height))
                for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                animation.release()
                animation = None

#Visualize single sample progression step
def visualizeStep(sample, sampleData, dir_progression, dir_chanProgressions):

    #Turn percent measured into a string
    percMeasured = "{:.2f}".format(sample.percMeasured)
    
    #Determine known information to be visualized
    if sampleData.impFlag or sampleData.postFlag: knownGT, knownRD, knownERD = False, False, True
    elif sampleData.trainFlag: knownGT, knownRD, knownERD = True, True, False
    elif sampleData.simulationFlag: knownGT, knownRD, knownERD = True, True, True
    if erdModel == 'GLANDS': knownRD = False
    
    #Turn ground-truth metrics into strings and set flag to indicate availability
    if knownGT:
        if not sampleData.datagenFlag:
            sumImageNRMSE = "{:.6f}".format(round(sample.sumImageNRMSE, 6))
            sumImageSSIM = "{:.6f}".format(round(sample.sumImageSSIM, 6))
            sumImagePSNR = "{:.6f}".format(round(sample.sumImagePSNR, 6))
            chanImageAvgNRMSE = "{:.6f}".format(round(np.nanmean(sample.chanImagesNRMSEList), 6))
            chanImageAvgSSIM = "{:.6f}".format(round(np.nanmean(sample.chanImagesSSIMList), 6))
            chanImageAvgPSNR = "{:.6f}".format(round(np.nanmean(sample.chanImagesPSNRList), 6))
        else:
            sumImageNRMSE = "N/A"
            sumImageSSIM = "N/A"
            sumImagePSNR = "N/A"
            chanImageAvgNRMSE = "N/A"
            chanImageAvgSSIM = "N/A"
            chanImageAvgPSNR = "N/A"
        if sampleData.allChanEvalFlag and not sampleData.datagenFlag and sampleData.dataMSI: 
            allImageAvgNRMSE = "{:.6f}".format(round(np.nanmean(sample.allImagesNRMSEList), 6))
            allImageAvgSSIM = "{:.6f}".format(round(np.nanmean(sample.allImagesSSIMList), 6))
            allImageAvgPSNR = "{:.6f}".format(round(np.nanmean(sample.allImagesPSNRList), 6))
        else: 
            allImageAvgNRMSE = "N/A"
            allImageAvgSSIM = "N/A"
            allImageAvgPSNR = "N/A"
    
    #Turn RD metrics into strings and set flag to indicate availability
    if knownRD and knownERD:
        ERD_NRMSE = "{:.6f}".format(round(sample.ERD_NRMSE, 6))
        ERD_SSIM = "{:.6f}".format(round(sample.ERD_SSIM, 6))
        ERD_PSNR = "{:.6f}".format(round(sample.ERD_PSNR, 6))
    
    #Setup measurement progression image variables if needed
    if len(sample.progMap.shape)>0:
        progMapValues = np.unique(sample.progMap[~np.isnan(sample.progMap)]).astype(int)
        cmapProgMap = plt.get_cmap('autumn', len(progMapValues))
        cmapProgMap.set_bad(color='black')
        boundValuesProgMap = np.linspace(1, progMapValues.max()+1, len(progMapValues)+1, dtype=int)
        normProgMap = matplotlib.colors.BoundaryNorm(boundValuesProgMap, cmapProgMap.N)
        progMap = sample.progMap+0.5
    
    #For each of the channels, generate visuals
    for chanNum in range(0, sampleData.numChannels):
        
        #Find minimum and maximum channel values for colorbars
        chanMinValue, chanMaxValue = np.min(sampleData.chanImages[chanNum]), np.max(sampleData.chanImages[chanNum])
        
        #Turn metrics into strings
        chanLabel = str(sampleData.chanValues[chanNum])
        if knownGT:
            if not sampleData.datagenFlag:
                chanImagesNRMSE = "{:.6f}".format(round(sample.chanImagesNRMSEList[chanNum], 6))
                chanImagesSSIM = "{:.6f}".format(round(sample.chanImagesSSIMList[chanNum], 6))
                chanImagesPSNR = "{:.6f}".format(round(sample.chanImagesPSNRList[chanNum], 6))
            else: 
                chanImagesNRMSE = "N/A"
                chanImagesSSIM = "N/A"
                chanImagesPSNR = "N/A"
        
        #Create a new figure
        if sampleData.impFlag or sampleData.postFlag: f = plt.figure(figsize=(30,5))
        else: f = plt.figure(figsize=(25,10))
        
        #Generate and apply a plot title, with metrics if applicable
        plotTitle = r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Channel:\ }$" + chanLabel + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured
        if knownGT:
            plotTitle += '\n' + r"$\bf{NRMSE\ -\ All\ Channel\ Avg:\ }$" + allImageAvgNRMSE + r"$\bf{\ \ Targeted\ Channel\ Avg:\ }$" + chanImageAvgNRMSE + r"$\bf{\ \ Targeted\ Channel:\ }$" + chanImagesNRMSE
            plotTitle += '\n' + r"$\bf{\ \ SSIM\ -\ All\ Channel\ Avg:\ }$" + allImageAvgSSIM + r"$\bf{\ \ Targeted\ Channel\ Avg:\ }$" + chanImageAvgSSIM + r"$\bf{\ \ Targeted\ Channel:\ }$" + chanImagesSSIM
            plotTitle += '\n' + r"$\bf{\ \ PSNR\ -\ All\ Channel\ Avg:\ }$" + allImageAvgPSNR + r"$\bf{\ \ Targeted\ Channel\ Avg:\ }$" + chanImageAvgPSNR + r"$\bf{\ \ Targeted\ Channel:\ }$" + chanImagesPSNR
        plt.suptitle(plotTitle)
        
        if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,3))
        else: ax = plt.subplot2grid((2,4), (0,0))
        im = ax.imshow(sampleData.chanImages[chanNum]*sample.mask, cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue, interpolation='none')
        ax.set_title('Measured')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.formatter.set_powerlimits((0, 0))
        
        if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,4))
        else: ax = plt.subplot2grid((2,4), (0,1))
        im = ax.imshow(sample.chanReconImages[chanNum], cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue, interpolation='none')
        ax.set_title('Reconstruction')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.formatter.set_powerlimits((0, 0))
        
        if knownGT: 
            ax = plt.subplot2grid((2,4), (0,2))
            im = ax.imshow(sampleData.chanImages[chanNum], cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue, interpolation='none')
            ax.set_title('Ground-Truth')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            cbar.formatter.set_powerlimits((0, 0))
            
            ax = plt.subplot2grid((2,4), (0,3))
            im = ax.imshow(abs(sampleData.chanImages[chanNum]-sample.chanReconImages[chanNum]), cmap='hot', aspect='auto', interpolation='none')
            ax.set_title('Absolute Difference')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            cbar.formatter.set_powerlimits((0, 0))
        
        if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,2))
        elif sampleData.trainFlag: ax = plt.subplot2grid((2,4), (1,1))
        elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,0))
        if len(sample.progMap.shape)>0:
            im = ax.imshow(progMap, cmap=cmapProgMap, norm=normProgMap, aspect='auto', interpolation='none')
            ax.set_title('Measurement Progression')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            cbar.ax.minorticks_off()
            tickValues = np.linspace(1, progMapValues.max(), len(cbar.get_ticks()), dtype=int)
            cbar.set_ticks(tickValues+0.5)
            cbar.ax.set_yticklabels(list(map(("{:0"+str(int(np.ceil(np.log10(sampleData.area))))+"d}").format, tickValues)))
        else:
            im = ax.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1, interpolation='none')
            ax.set_title('Measurement Mask')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            cbar.formatter.set_powerlimits((0, 0))
        
        if knownRD:
            if sampleData.trainFlag: ax = plt.subplot2grid((2,4), (1,2))
            elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,1))
            im = ax.imshow(sample.RDs[chanNum], cmap='viridis', aspect='auto', interpolation='none')
            ax.set_title('RD')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            cbar.formatter.set_powerlimits((0, 0))
        
        if knownERD:
            if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,0))
            elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,2))
            im = ax.imshow(sample.ERDs[chanNum], cmap='viridis', aspect='auto', interpolation='none')
            ax.set_title('ERD')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            cbar.formatter.set_powerlimits((0, 0))
            
            if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,1))
            elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,3))
            im = ax.imshow(np.nan_to_num(sample.processedERDs[chanNum], nan=0, posinf=0, neginf=0), cmap='viridis', aspect='auto', interpolation='none')
            ax.set_title('Processed ERD')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            cbar.formatter.set_powerlimits((0, 0))
        
        #Save
        f.tight_layout()
        if not(sampleData.impFlag or sampleData.postFlag): f.subplots_adjust(top = 0.85)
        saveLocation = dir_chanProgressions[chanNum] + 'progression_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) +'.tiff'
        plt.savefig(saveLocation)
        #plt.close(f)

        #Do borderless saves for each channel image here; skip mask/progMap as they will be produced in the progression output
        if knownERD:
            saveLocation = dir_chanProgressions[chanNum] + 'erd_original_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
            visualizeBorderless(sample.ERDs[chanNum], saveLocation, cmap='viridis')
            
            saveLocation = dir_chanProgressions[chanNum] + 'erd_processed_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
            visualizeBorderless(np.nan_to_num(sample.processedERDs[chanNum], nan=0, posinf=0, neginf=0), saveLocation, cmap='viridis')
        
        if knownRD:
            saveLocation = dir_chanProgressions[chanNum] + 'rd_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
            visualizeBorderless(sample.RDs[chanNum], saveLocation, cmap='viridis')
        
        if knownGT:
            saveLocation = dir_chanProgressions[chanNum] + 'groundTruth_channel_' + chanLabel + '.tiff'
            visualizeBorderless(sampleData.chanImages[chanNum], saveLocation, cmap='hot', vmin=chanMinValue, vmax=chanMaxValue)

        saveLocation = dir_chanProgressions[chanNum] + 'reconstruction_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
        visualizeBorderless(sample.chanReconImages[chanNum], saveLocation, cmap='hot', vmin=chanMinValue, vmax=chanMaxValue)
        
        saveLocation = dir_chanProgressions[chanNum] + 'measured_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
        visualizeBorderless(sample.chanImages[chanNum]*sample.mask, saveLocation, cmap='hot', vmin=chanMinValue, vmax=chanMaxValue)
        
    #For the overall progression, get min/max of the ground-truth sum image for visualization
    sumImageMinValue, sumImageMaxValue = np.min(sampleData.sumImage), np.max(sampleData.sumImage)
    
    #Create a new figure
    if sampleData.impFlag or sampleData.postFlag: f = plt.figure(figsize=(30,5))
    else: f = plt.figure(figsize=(25,10))

    #Generate and apply a plot title, with metrics if applicable
    plotTitle = r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured
    if knownGT:
        plotTitle += '\n' + r"$\bf{NRMSE\ -\ All\ Channel\ Avg:\ }$" + allImageAvgNRMSE + r"$\bf{\ \ Targeted\ Channel\ Avg:\ }$" + chanImageAvgNRMSE + r"$\bf{\ \ Sum\ Image: }$" + sumImageNRMSE
        if knownRD and knownERD: plotTitle += r"$\bf{\ \ Avg\ ERD:\ }$" + ERD_NRMSE 
        plotTitle += '\n' + r"$\bf{\ \ SSIM\ -\ All\ Channel\ Avg:\ }$" + allImageAvgSSIM + r"$\bf{\ \ Targeted\ Channel\ Avg:\ }$" + chanImageAvgSSIM + r"$\bf{\ \ Sum\ Image: }$" + sumImageSSIM
        if knownRD and knownERD: plotTitle += r"$\bf{\ \ Avg\ ERD:\ }$" + ERD_SSIM
        plotTitle += '\n' + r"$\bf{\ \ PSNR\ -\ All\ Channel\ Avg:\ }$" + allImageAvgPSNR + r"$\bf{\ \ Targeted\ Channel\ Avg:\ }$" + chanImageAvgPSNR + r"$\bf{\ \ Sum\ Image: }$" + sumImagePSNR
        if knownRD and knownERD: plotTitle += r"$\bf{\ \ Avg\ ERD:\ }$" + ERD_PSNR
    plt.suptitle(plotTitle)
    
    if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,3))
    else: ax = plt.subplot2grid((2,4), (0,0))
    im = ax.imshow(sampleData.sumImage*sample.mask, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue, interpolation='none')
    ax.set_title('Measured')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    cbar.formatter.set_powerlimits((0, 0))
    
    if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,4))
    else: ax = plt.subplot2grid((2,4), (0,1))
    im = ax.imshow(sample.sumReconImage, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue, interpolation='none')
    ax.set_title('Sum Image Reconstruction')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    cbar.formatter.set_powerlimits((0, 0))
    
    if knownGT: 
        ax = plt.subplot2grid((2,4), (0,2))
        im = ax.imshow(sampleData.sumImage, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue, interpolation='none')
        ax.set_title('Sum Image Ground-Truth')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.formatter.set_powerlimits((0, 0))
        
        ax = plt.subplot2grid((2,4), (0,3))
        im = ax.imshow(abs(sampleData.sumImage-sample.sumReconImage), cmap='hot', aspect='auto', interpolation='none')
        ax.set_title('Absolute Difference')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.formatter.set_powerlimits((0, 0))
    
    if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,2))
    elif sampleData.trainFlag: ax = plt.subplot2grid((2,4), (1,1))
    elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,0))
    if len(sample.progMap.shape)>0:
        im = ax.imshow(progMap, cmap=cmapProgMap, norm=normProgMap, aspect='auto', interpolation='none')
        ax.set_title('Measurement Progression')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.ax.minorticks_off()
        tickValues = np.linspace(1, progMapValues.max(), len(cbar.get_ticks()), dtype=int)
        cbar.set_ticks(tickValues+0.5)
        cbar.ax.set_yticklabels(list(map(("{:0"+str(int(np.ceil(np.log10(sampleData.area))))+"d}").format, tickValues)))
    else:
        im = ax.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1, interpolation='none')
        ax.set_title('Measurement Mask')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.formatter.set_powerlimits((0, 0))

    if knownRD:
        if sampleData.trainFlag: ax = plt.subplot2grid((2,4), (1,2))
        elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,1))
        im = ax.imshow(sample.RD, cmap='viridis', aspect='auto', interpolation='none')
        ax.set_title('RD')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.formatter.set_powerlimits((0, 0))
    
    if knownERD:
        if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,0))
        elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,2))
        im = ax.imshow(sample.ERD, cmap='viridis', aspect='auto', interpolation='none')
        ax.set_title('ERD')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.formatter.set_powerlimits((0, 0))
        
        if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,1))
        elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,3))
        im = ax.imshow(np.nan_to_num(sample.processedERD, nan=0, posinf=0, neginf=0), cmap='viridis', aspect='auto', interpolation='none')
        ax.set_title('Processed ERD')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.formatter.set_powerlimits((0, 0))
    
    #Save
    f.tight_layout()
    f.subplots_adjust(top = 0.85)
    saveLocation = dir_progression + 'progression' + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '_avg.tiff'
    plt.savefig(saveLocation)
    #plt.close(f)

    #Borderless saves
    if knownERD:
        saveLocation = dir_progression + 'ERD_original_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
        visualizeBorderless(sample.ERD, saveLocation, cmap='viridis')
        
        saveLocation = dir_progression + 'ERD_processed_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
        visualizeBorderless(np.nan_to_num(sample.processedERD, nan=0, posinf=0, neginf=0), saveLocation, cmap='viridis')
    
    if knownRD:
        saveLocation = dir_progression + 'RD_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
        visualizeBorderless(sample.RD, saveLocation, cmap='viridis')
        
    if knownGT:
        saveLocation = dir_progression + 'groundTruth_sumImage_' + chanLabel + '.tiff'
        visualizeBorderless(sampleData.sumImage, saveLocation, cmap='hot', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    
    saveLocation = dir_progression + 'reconstruction_sumImage' + '_iter_' + str(sample.iteration) +  '_perc_' + str(sample.percMeasured) + '.tiff'
    visualizeBorderless(sample.sumReconImage, saveLocation, cmap='hot', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    
    saveLocation = dir_progression + 'measured_sumImage_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
    visualizeBorderless(sample.sumImage*sample.mask, saveLocation, cmap='hot', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    
    saveLocation = dir_progression + 'mask_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
    visualizeBorderless(sample.mask, saveLocation, cmap='gray', vmin=0, vmax=1)
    
    #Note: quite incompatible with visualizeBorderless and spent too much time here; leave it alone
    if len(sample.progMap.shape)>0:
        saveLocation = dir_progression + 'progressionMap_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.tiff'
        image = divideSafe(progMap, np.nanmax(progMap))
        image[np.isnan(progMap)] = np.nan
        Image.fromarray(np.uint8(cmapProgMap(image)*255)).save(saveLocation)

#Perform sampling
def runSampling(sampleDataNum, sampleData, cValue, model, percToScan, percToViz, lineVisitAll, dir_Results, tqdmHide, samplingProgress_Actor=None, percProgUpdate=None):
    
    #If in parallel, ignore warnings
    if parallelization: setupLogging()
    
    #Make sure random selection is consistent
    resetRandom()
    
    #If groupwise is active, specify how many points should be scanned each step
    if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random') and percToScan != None: sampleData.pointsToScan = int(np.ceil(((sampleData.stopPerc/100)*sampleData.area)/(sampleData.stopPerc/percToScan)))
    elif (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random') and percToScan == None: sampleData.pointsToScan = 1
    elif sampleData.scanMethod == 'linewise' and sampleData.useMaskFOV: sampleData.pointsToScan = [int(np.ceil((sampleData.linePerc/100)*np.sum(sampleData.maskFOV[lineIndex]))) for lineIndex in range(0, sampleData.finalDim[0])]
    elif sampleData.scanMethod == 'linewise': sampleData.pointsToScan = [int(np.ceil((sampleData.linePerc/100)*sampleData.finalDim[1])) for _ in range(0, sampleData.finalDim[0])]
    else: sys.exit('\nError - The number of points to scan could not be determined. Please confirm that the configuration file options specified were valid.') 
    
    #Create a segmented storage object for variables that must be referenced over the length of scanning, yet (for better memory overhead) are not desired to be retained in the result object
    tempScanData = TempScanData()
    
    #Create a new sample object to hold current information
    sample = Sample(sampleData, tempScanData)
    
    #Indicate that the stopping condition has not yet been met
    completedRunFlag = False
    
    #Create a new result object to hold scanning progression
    result = Result(sampleDataNum, sampleData, dir_Results, cValue)
    
    #Scan initial sets
    for initialSet in sampleData.initialSets: sample.performMeasurements(sampleData, tempScanData, result, initialSet, model, cValue, False)
    
    #Check stopping criteria, just in case of a bad input
    if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random' or not lineVisitAll) and (sample.percMeasured >= sampleData.stopPerc): sys.exit('\nError - All points were scanned or the stopping criteria have been met after the initial acquisition for sample: ' + sample.name)
    elif sampleData.scanMethod == 'linewise' and len(sampleData.linesToScan)-np.sum(np.sum(sample.mask, axis=1)>0) == 0: sys.exit('\nError - All lines were scanned after the inital acquisition for sample: ' + sample.name)
    if not sampleData.datagenFlag and np.sum(sample.processedERD) == 0: sys.exit('\nError - Initial ERD indicates there are no places to scan for sample: ' + sampleData.name + ' This probably means something went wrong during the sample read process, please check that the file formats are compatible.')
    
    #Perform the first update for the result
    result.update(sample, sampleData, completedRunFlag)
    
    #Until the stopping criteria has been met
    if not tqdmHide: pbar = tqdm(total = round(float(sampleData.stopPerc),2), desc = '% Sampled', leave=False, ascii=asciiFlag, disable=tqdmHide)
    
    #Initialize progress bar state according to % measured
    if not tqdmHide:
        pbar.n = np.clip(round(sample.percMeasured,2), 0, sampleData.stopPerc)
        pbar.refresh()
    if samplingProgress_Actor != None and tqdmHide: 
        _ = ray.get(samplingProgress_Actor.update.remote(sample.percMeasured))
        lastPercMeasured = copy.deepcopy(sample.percMeasured)
    
    #Until the program has completed
    while not completedRunFlag:
        
        #Find next measurement locations
        newIdxs = findNewMeasurementIdxs(sample, sampleData, tempScanData, result, model, cValue, percToScan)
        
        #Perform measurements, reconstructions and ERD/RD computations
        if len(newIdxs) != 0: sample.performMeasurements(sampleData, tempScanData, result, newIdxs, model, cValue, False)
        else: break
        
        #Check stopping criteria
        if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random' or not lineVisitAll) and (sample.percMeasured >= sampleData.stopPerc): completedRunFlag = True
        elif sampleData.scanMethod == 'linewise' and len(sampleData.linesToScan)-np.sum(np.sum(sample.mask, axis=1)>0) == 0: completedRunFlag = True
        if not sampleData.datagenFlag and np.sum(sample.processedERD) == 0: completedRunFlag = True
        
        #If viz limit, only update when percToViz has been met; otherwise update every iteration
        if ((percToViz != None) and ((sample.percMeasured-result.lastPercMeasured) >= percToViz)) or (percToViz == None) or (sampleData.scanMethod == 'linewise') or completedRunFlag: result.update(sample, sampleData, completedRunFlag)
        
        #If using a global progress bar and percProgUpdate has been reached, then update the global sampling progress actor
        if samplingProgress_Actor != None and tqdmHide and (sample.percMeasured-lastPercMeasured >= percProgUpdate): 
            _ = ray.get(samplingProgress_Actor.update.remote(sample.percMeasured-lastPercMeasured))
            lastPercMeasured = copy.deepcopy(sample.percMeasured)
        
        #Update the progress bar
        if not tqdmHide:
            pbar.n = np.clip(round(sample.percMeasured,2), 0, sampleData.stopPerc)
            pbar.refresh()
    
    #If multiple samples are being sampled and then processed (as done in simulation), then pickle result to reduce RAM overhead
    #if pickleTestResult: 
    #    resultLocation = dir_TestingResults + sampleData.name + '_result.p'
    #    with open(resultLocation, mode='wb') as file: cloudpickle.dump(result, file)
    #
    #    #resultLocation = dir_TestingResults + sampleData.name + '_result.p'
    #    #pickle.dump(result, open(resultLocation, 'wb'))
    #    return resultLocation
    return result

#Compute approximated Reduction in Distortion (RD) values
def computeRD(sample, sampleData, tempScanData, cValue, updateLocations):
    
    #If updating RD, replace affected locations with 0, otherwise compute RD values in square dimensionality
    if len(updateLocations) > 0:
        
        #Note: The only time that the RD may be updated, rather than fully computed, is in the case of an oracle run or c value optimization with percToScan enabled
        #Therein, for SLADS and DLADS, the reconstructions and RDPPs are not being updated by ground-truth data, as that would require updating neighbor information
        #Since this is trying to be avoided with percToScan, actually computing updated RD values would have no benefit over percToViz with the IDW reconstruction method
        
        #Identify unique unmeasured locations where the RD would be altered based on data at update locations (i.e., which unmeasured locations will have a new nearest measured neighbor), perform no update if there are none (Can happen with DESI)
        indices = np.unique(np.concatenate([np.argwhere(np.sum(updateLocation >= tempScanData.winStartPos, axis=1)+np.sum(updateLocation <= tempScanData.winStopPos, axis=1)==4) for updateLocation in updateLocations]).flatten())
        if len(indices) == 0: return
        squareUnMeasuredLocations = tempScanData.squareUnMeasuredIdxs[indices]
        
        #Set RD Values at affected unmeasured locations to 0
        sample.squareRDs[:, squareUnMeasuredLocations[:,0], squareUnMeasuredLocations[:,1]] = 0
        
    else: 
        
        #Compute sigma values for all unmeasured locations and set locations to be updated
        sigmaValues = tempScanData.neighborDistances[:,0]/cValue
        squareUnMeasuredLocations = tempScanData.squareUnMeasuredIdxs
        
        #Compute window sizes and positions for unmeasured locations, updating a persistant reference for updating RD
        if not staticWindow: windowSizes = 2*np.ceil(dynWindowSigMult*sigmaValues).astype(int)+1
        else: windowSizes = (np.ones((len(sigmaValues)), dtype=np.float32)*staticWindowSize)
        windowSizes[windowSizes%2==0] += 1
        radii = (windowSizes//2).reshape(-1, 1).astype(int)
        winStartPos, winStopPos = squareUnMeasuredLocations-radii, squareUnMeasuredLocations+radii
        if len(updateLocations) == 0: tempScanData.winStartPos, tempScanData.winStopPos = winStartPos, winStopPos
        else: tempScanData.winStartPos[indices], tempScanData.winStopPos[indices] = winStartPos, winStopPos
        
        #Extract unique/new sigma values and their respective window sizes
        newSigmaValues, newIndexes = np.unique(sigmaValues, return_index=True)
        newWindows = windowSizes[newIndexes]
        existingSigma = sampleData.gaussianWindows.keys()
        newIndexes = [index for index in range(0, len(newSigmaValues)) if newSigmaValues[index] not in existingSigma]
        newSigmaValues, newWindows = newSigmaValues[newIndexes], newWindows[newIndexes]
        
        #Store new Gaussian computations
        if len(newSigmaValues) > 0:
            gaussianSignals = [signal.windows.gaussian(newWindows[index], newSigmaValues[index]) for index in range(0, len(newSigmaValues))]
            sampleData.gaussianWindows.update(zip(newSigmaValues, [np.outer(gaussianSignals[index], gaussianSignals[index]) for index in range(0, len(gaussianSignals))]))
        
        #Zero-pad the RDPPs by the maximum radius and offset window positions accordingly
        maxRadius = np.max(radii)
        paddedRDPPs = np.pad(sample.RDPPs, [(0, 0), (maxRadius, maxRadius), (maxRadius, maxRadius)], mode='constant')
        offsetWinStartPos, offsetWinStopPos = winStartPos+maxRadius, winStopPos+maxRadius
        
        #Compute RD Values
        sample.squareRDs[:, squareUnMeasuredLocations[:,0], squareUnMeasuredLocations[:,1]] = np.asarray([np.sum(sampleData.gaussianWindows[sigmaValues[index]]*paddedRDPPs[:, offsetWinStartPos[index][0]:offsetWinStopPos[index][0]+1, offsetWinStartPos[index][1]:offsetWinStopPos[index][1]+1], axis=(1,2)) for index in range(0, len(squareUnMeasuredLocations))]).T
    
    #In square dimensionality, set any invalid values to 0
    sample.squareRDs = np.nan_to_num(sample.squareRDs, nan=0, posinf=0, neginf=0)
    
    #Resize RDs as needed according to sample type, ensure values at measured locations are zero, and average results to form a single RD
    if sampleData.sampleType == 'DESI': 
        sample.RDs = np.moveaxis(resize(np.moveaxis(sample.squareRDs, 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)*(1-sample.mask)
        sample.RD = np.mean(sample.RDs, axis=0)
    
    #In square dimensionality, ensure values at measured locations are zero and average results to form a single RD
    sample.squareRDs = sample.squareRDs*(1-sample.squareMask)
    sample.squareRD = np.mean(sample.squareRDs, axis=0)
    
    #If sample type did not need resizing, set RD in regular dimensionality to that of the square
    if sampleData.sampleType != 'DESI': 
        sample.RDs = sample.squareRDs
        sample.RD = sample.squareRD
    
#Determine which unmeasured points of a sample should be scanned given the current E/RD
def findNewMeasurementIdxs(sample, sampleData, tempScanData, result, model, cValue, percToScan):
    
    if sampleData.scanMethod == 'random':
        np.random.shuffle(sample.unMeasuredIdxs)
        newIdxs = sample.unMeasuredIdxs[:sampleData.pointsToScan].astype(int)
        
    elif sampleData.scanMethod == 'pointwise':
    
        #If performing a groupwise scan
        if percToScan != None:
        
            #SLADS/DLADS use reconstruction data for temporary measurement values, until specified number of locations have been found
            if erdModel != 'GLANDS':
                newIdxs = []
                while True:
                    newIdx = sample.unMeasuredIdxs[np.argmax(sample.processedERD[sample.unMeasuredIdxs[:,0], sample.unMeasuredIdxs[:,1]])]
                    newIdxs.append(newIdx.tolist())
                    sample.performMeasurements(sampleData, tempScanData, result, newIdx, model, cValue, True)
                    if ((np.sum(sample.mask)-np.sum(result.lastMask)) >= sampleData.pointsToScan) or (sample.percMeasured >= sampleData.stopPerc) or (np.sum(sample.processedERD) == 0): break
                newIdxs = np.asarray(newIdxs)
                
            #GLANDS chooses all the locations in a single step
            else:
                newIdxs = sample.unMeasuredIdxs[np.argsort(sample.processedERD[sample.unMeasuredIdxs[:,0], sample.unMeasuredIdxs[:,1]])[::-1]][:sampleData.pointsToScan].astype(int)
            
        #Else, for scanning only a single location, identify the unmeasured location with the highest processedERD value
        else:
            newIdxs = np.asarray([sample.unMeasuredIdxs[np.argmax(sample.processedERD[sample.unMeasuredIdxs[:,0], sample.unMeasuredIdxs[:,1]])].tolist()])
        
    elif sampleData.scanMethod == 'linewise':
        
        #If all locations on a chosen line should be scanned, select the line with maximum sum physical ERD
        if lineMethod =='fullLine':
            lineToScanIdx = np.nanargmax(np.nansum(sample.processedERD, axis=1))
            indexes = np.sort(np.argsort(sample.processedERD[lineToScanIdx])[::-1])
            newIdxs = np.column_stack([np.ones(len(indexes), dtype=np.float32)*lineToScanIdx, indexes]).astype(int)
        
        #If a group of locations should be chosen on a single line
        elif lineMethod == 'percLine': 
            
            #Select line with maximum sum physical ERD
            lineToScanIdx = np.nanargmax(np.nansum(sample.processedERD, axis=1))
            
            #SLADS/DLADS use reconstruction data for temporary measurement values, until specified number of locations have been found
            if erdModel != 'GLANDS':
                newIdxs = []
                while True:
                    if (np.sum(sample.processedERD[lineToScanIdx]) <= 0) or (len(newIdxs) >= sampleData.pointsToScan[lineToScanIdx]): break
                    newIdxs.append([lineToScanIdx, np.argmax(sample.processedERD[lineToScanIdx])])
                    sample.performMeasurements(sampleData, tempScanData, result, np.asarray(newIdxs[-1]), model, cValue, True)
                
                #Convert to array for indexing, sorting columns according to physical scanning order
                newIdxs = np.asarray(newIdxs)
                newIdxs[:,1] = np.sort(newIdxs[:,1])
                
            #GLANDS chooses all the locations in a single step
            else:
                indexes = np.sort(np.argsort(sample.processedERD[lineToScanIdx])[::-1][:sampleData.pointsToScan[lineToScanIdx]])
                newIdxs = np.column_stack([np.ones(len(indexes), dtype=np.float32)*lineToScanIdx, indexes]).astype(int)
            
        #If a segment of a line should be scanned using a given percentage, choose the line with maximum sum physical ERD
        elif lineMethod == 'segLine' and segLineMethod == 'minPerc': 
           lineToScanIdx = np.nanargmax(np.nansum(sample.processedERD, axis=1))
           if np.nansum(sample.processedERD[lineToScanIdx]) == 0: startPos = ((sampleData.finalDim[1]-sampleData.pointsToScan[lineToScanIdx])//2)-1
           else: startPos = np.argmax(np.nansum(np.lib.stride_tricks.sliding_window_view(sample.processedERD[lineToScanIdx], sampleData.pointsToScan[lineToScanIdx]), axis=1))
           newIdxs = np.column_stack([np.ones(sampleData.pointsToScan[lineToScanIdx], dtype=np.float32)*lineToScanIdx, np.arange(startPos,startPos+sampleData.pointsToScan[lineToScanIdx])]).astype(int)
            
        #If a segment of a line should be scanned using Otsu, choose the line with the most scannable positions
        elif lineMethod == 'segLine' and segLineMethod == 'otsu': 
            otsuMask = sample.processedERD>=skimage.filters.threshold_otsu(sample.processedERD, nbins=100)
            lineToScanIdx = np.nanargmax(np.nansum(otsuMask, axis=1))
            indexes = np.sort(np.where(otsuMask[lineToScanIdx])[0])
            if len(indexes)>0: 
                indexes = np.arange(indexes[0],indexes[-1]+1)
                newIdxs = np.column_stack([np.ones(len(indexes), dtype=np.float32)*lineToScanIdx, indexes]).astype(int)
        
        #If there are not enough locations selected, then return no new measurement locations which will terminate scanning
        if len(newIdxs) < int(round(0.01*sample.mask.shape[1])): return []
        
    return newIdxs

#Re-index a set of m/z values to a common grid, all data should be or converted to float32 for numba acceleration to work correctly
@jit(nopython=True, nogil=True)
def mzFastIndex(mz, values, mzLowerIndex, mzPrecision, mzRound, mzInitialCount):
    indices = np.empty(len(mz), dtype=np.float32)
    mzValues = np.zeros(mzInitialCount, dtype=np.float32)
    np.round(np.floor(mz/mzPrecision)*mzPrecision, mzRound, indices)
    mzValues[(indices/mzPrecision).astype(np.int32)-mzLowerIndex] = values
    return mzValues

#Calculate k-nn and determine inverse distance weights
def findNeighbors(tempScanData):
    tempScanData.neighborDistances, tempScanData.neighborIndices = NearestNeighbors(n_neighbors=numNeighbors).fit(tempScanData.squareMeasuredIdxs).kneighbors(tempScanData.squareUnMeasuredIdxs)
    unNormNeighborWeights = 1.0/(tempScanData.neighborDistances**2.0)
    tempScanData.neighborWeights = unNormNeighborWeights/(np.sum(unNormNeighborWeights, axis=1))[:, np.newaxis]

#Perform the reconstruction using IDW (inverse distance weighting); if 3D compute all channels simultaneously
def computeReconIDW(inputImage, tempScanData):
    reconImage = copy.deepcopy(inputImage)
    if len(tempScanData.squareUnMeasuredIdxs) > 0:
        if len(inputImage.shape) == 3: reconImage[:, tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]] = np.sum(inputImage[:, tempScanData.squareMeasuredIdxs[:,0], tempScanData.squareMeasuredIdxs[:,1]][:, tempScanData.neighborIndices]*tempScanData.neighborWeights, axis=-1)
        else: reconImage[tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]] = np.sum(inputImage[tempScanData.squareMeasuredIdxs[:,0], tempScanData.squareMeasuredIdxs[:,1]][tempScanData.neighborIndices]*tempScanData.neighborWeights, axis=1)
    return reconImage

#Define process for converting images into input samples
def processImages(baseFolder, filenames, label):

    #If there are files to be processed, then convert each
    if len(filenames) > 0:
        for filename in tqdm(filenames, desc='Converting '+ label+ ' Images', leave=True, ascii=asciiFlag):
            
            #Extract name and verify the image has a supported extension and number of channels
            basename, extension = os.path.splitext(filename)
            basename, numChannels = os.path.basename(basename).split('-numChan-')
            numChannels = int(numChannels)
            if extension not in ['.png', '.jpg', '.tiff']: 
                print('\nError - Skipping file: '+filename+' as image extenstion is not currently compatible.')
                break
            image = cv2.imread(filename)
            if len(image.shape) > 3:
                print('\nError - Skipping file: '+filename+' as image contains more than 3 channels, which is not currently supported.')
                break
            
            #Setup a destination folder for the new smaple, overwriting existing matches
            destinationFolder = baseFolder+basename+os.path.sep
            if os.path.exists(destinationFolder): shutil.rmtree(destinationFolder)
            os.makedirs(destinationFolder)
            
            #Save a sampleInfo.txt for the sample
            sampleInfo = ['IMAGE', image.shape[1], image.shape[0], numChannels]
            np.savetxt(destinationFolder+'sampleInfo.txt', sampleInfo, delimiter=',', fmt='%s')
            
            #Save each channel as a separate image
            if numChannels > 1:
                for chanNum in range(0, numChannels):
                    outputLocation = destinationFolder+basename+'-chan-'+str(chanNum)+extension
                    _ = cv2.imwrite(outputLocation, image[:,:,chanNum])
            else:
                outputLocation = destinationFolder+basename+'-chan-0'+extension
                _ = cv2.imwrite(outputLocation, image)
            
            #Delete the original file to prevent repeat processing in the future
            os.remove(filename)

#Manually define customized NRMSE metric to use specified data range in min-max normalization
#Derived from: https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/metrics/simple_metrics.py#L51-L109
def compare_NRMSE(image_true, image_test, data_range):
    skimage._shared.utils.check_shape_equality(image_true, image_test)
    image_true, image_test = skimage.metrics.simple_metrics._as_floats(image_true, image_test)
    return np.sqrt(compare_MSE(image_true, image_test))/data_range
    
#Compare label and recon images after min-max rescaling
def compareImages(label, recon, minValue, maxValue):
    data_range=maxValue-minValue
    score_NRMSE = compare_NRMSE(label, recon, data_range)
    score_SSIM = compare_SSIM(label, recon, data_range=data_range)
    score_PSNR = compare_PSNR(label, recon, data_range=data_range)
    return score_NRMSE, score_SSIM, score_PSNR

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

#Perform array division with consideration for divide-by-zero
def divideSafe(nominator, denominator):
    #result = np.nan_to_num(np.divide(nominator, denominator, out=np.zeros_like(nominator), where=denominator!=0), nan=0, posinf=0, neginf=0)
    with np.errstate(divide='ignore', invalid='ignore'): result = np.nan_to_num(np.divide(nominator, denominator), nan=0, posinf=0, neginf=0)
    return result
    
#Truncate a value to a given precision 
def truncate(value, decimalPlaces=0): return np.trunc(value*10**decimalPlaces)/(10**decimalPlaces)

#Visualize/save an image without borders/axes
def visualizeBorderless(image, saveLocation, cmap='viridis', vmin=None, vmax=None):
    if type(cmap) == str: cmap = plt.get_cmap(cmap)
    if vmin==None: vmin=np.nanmin(image)
    if vmax==None: vmax=np.nanmax(image)
    Image.fromarray(np.uint8(cmap(divideSafe((image-vmin), (vmax-vmin)))*255)).save(saveLocation)

#Visualize/save a simple data plot
def basicPlot(xData, yData, saveLocation, xLabel='', yLabel=''):
    font = {'size' : 18}
    plt.rc('font', **font)
    f = plt.figure(figsize=(20,8))
    ax1 = f.add_subplot(1,1,1)    
    ax1.plot(xData, yData, color='black')
    ax1.set_xlabel(xLabel)
    ax1.set_ylabel(yLabel)
    plt.savefig(saveLocation)
    plt.close(f)

#Define how to reset the random seed for deterministic repeatable RNG
def resetRandom():
    if manualSeedValue != -1:
        if erdModel == 'DLADS-TF': tf.random.set_seed(manualSeedValue)
        else: torch.manual_seed(manualSeedValue)        
        np.random.seed(manualSeedValue)
        random.seed(manualSeedValue)