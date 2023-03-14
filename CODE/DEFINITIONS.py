#==================================================================
#GLOBAL METHOD AND CLASS DEFINITIONS
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
        
        #These flags should not be changed or used directly during this __init__ metthod
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
        self.allImagesPath = None
        self.squareAllImagesPath = None
        self.avgTimeFileLoad = None
        self.useAlphaTims = False
        
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
            self.sampleType = 'IMAGE'
            
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
        
        #If MSI data, determine the needed precision to prevent overlap when mapping m/z values to a common spectrum
        #How many decimal places are used in the lowerbound mz range, as first limited by the ppm precision
        #Arbitrary precision rounding is problematic, so go to the lower power of 10 instead
        if self.dataMSI:
            ppmRound = ('%f' % self.ppm).rstrip('0').rstrip('.')[::-1].find('.')
            mzMinDiff = np.round((self.mzLowerBound*self.ppmPos)-(self.mzLowerBound*self.ppmNeg), ppmRound)
            self.mzRound =str(mzMinDiff)[::-1].find('.')
            self.mzPrecision = np.float32(10**-self.mzRound)
        
        #If reading in all MSI Data determine set of non-overlapping bins based on ppm
        if self.dataMSI and self.readAllMSI:
            
            #Compute the unshifted lower m/z bound's index position
            self.mzLowerIndex = int(self.mzLowerBound/self.mzPrecision)
            
            #Compute m/z for initial binning
            self.mzInitialCount = int((self.mzUpperBound-self.mzLowerBound)/self.mzPrecision)+1
            self.mzInitialDist = np.round(np.round(np.linspace(self.mzLowerBound, self.mzUpperBound, self.mzInitialCount)/self.mzPrecision)*self.mzPrecision, self.mzRound)
            
            #Determine m/z values with final non-overlapping ppm bins to enable fast loading/indexing
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
            
            #Now that all of the final dimensions have been determined, setup .hdf5 file locations, and either a shared memory actor or array, for storing results 
            self.allImagesPath = self.sampleFolder+os.path.sep+'allImages.hdf5'
            if os.path.exists(self.allImagesPath): os.remove(self.allImagesPath)
            if self.sampleType=='MALDI': 
                self.squareAllImagesPath = None
            elif self.sampleType=='DESI':
                self.squareAllImagesPath = self.sampleFolder+os.path.sep+'squareAllImages.hdf5'
                if os.path.exists(self.squareAllImagesPath): os.remove(self.squareAllImagesPath)
            
        #If MSI data, regardless of if all channel data is to be loaded or not
        if self.dataMSI:
            
            #Load targeted m/z values in sorted order, preferring those in the sample folders if available, and set corresponding ranges 
            #If there is only a single specificed channel, then need to convert format to array with list
            try: self.chanValues = np.loadtxt(self.sampleFolder+os.path.sep+'channels.csv', delimiter=',')
            except: 
                if overrideChannelsFile == None: self.chanValues = np.loadtxt('channels.csv', delimiter=',')
                else: self.chanValues = np.loadtxt(overrideChannelsFile, delimiter=',')
            if self.chanValues.shape == (): self.chanValues = np.array([self.chanValues])
            self.numChannels = len(self.chanValues)
            self.chanValues.sort()
            #self.mzRanges = np.round(np.column_stack((self.chanValues*self.ppmNeg, self.chanValues*self.ppmPos)), self.mzRound)
            self.mzRanges = np.column_stack((self.chanValues*self.ppmNeg, self.chanValues*self.ppmPos))
            
            #For DESI MSI samples, create regular grids for interpolating all/targeted m/z data
            if self.sampleType == 'DESI': 
                mzFinalGrid_x, mzFinalGrid_y = np.meshgrid(self.newTimes, self.mzFinal)
                if parallelization: self.mzFinalGrid_id = ray.put((mzFinalGrid_x, mzFinalGrid_y))
                else: self.mzFinalGrid = (mzFinalGrid_x, mzFinalGrid_y)
                chanFinalGrid_x, chanFinalGrid_y = np.meshgrid(self.newTimes, self.chanValues)
                if parallelization: self.chanFinalGrid_id = ray.put((chanFinalGrid_x, chanFinalGrid_y))
                else: self.chanFinalGrid = (chanFinalGrid_x, chanFinalGrid_y)
                del mzFinalGrid_x, mzFinalGrid_y, chanFinalGrid_x, chanFinalGrid_y
        
            #Setup shared memory actor for operations in parallel, or local memory for serial execution
            if parallelization: 
                self.mzOriginalIndices_id, self.mzRanges_id = ray.put(self.mzOriginalIndices), ray.put(self.mzRanges)
                self.reader_MSI_Actor = Reader_MSI_Actor.remote(self.sampleType, self.readAllMSI, len(self.mzFinal), len(self.chanValues), self.finalDim[0], self.finalDim[1], self.allImagesPath, self.squareAllImagesPath)
            else: 
                self.allImages = np.zeros((len(self.mzFinal), self.finalDim[0], self.finalDim[1]), dtype=np.float32)
        
        #Setup targeted channel and sum images for holding data
        self.chanImages = np.zeros((self.numChannels, self.finalDim[0], self.finalDim[1]), dtype=np.float32)
        self.sumImage = np.zeros((self.finalDim), dtype=np.float32)
        
        #If an MSI sample and an optical image is to be used try loading different extensions
        if self.dataMSI and (applyOptical != None or 'opticalData' in inputChannels):
            opticalImageFound = False
            for extension in ['.png', '.jpg', '.tiff']:
                if os.path.isfile(self.sampleFolder+os.path.sep+'optical'+extension): 
                    opticalImageFound = True
                    break
            if not opticalImageFound: sys.exit('\nError - applyOptical was enabled, but no optical image was found for sample: ' + sample.name)
            
            #Load the optical image, rescaling range 0 to 1 and inversing non-zero values (remove background and positively weight structures)
            opticalImage = (cv2.imread(self.sampleFolder+os.path.sep+'optical'+extension, 0)/255)
            opticalMask = opticalImage!=0
            opticalImage[opticalMask] = 1-opticalImage[opticalMask]
            if applyOptical == 'secDerivBias': 
                opticalImageSecDeriv = abs(cv2.Laplacian(opticalImage, cv2.CV_64F))
                self.squareOpticalImageSecDeriv = resize(opticalImageSecDeriv, tuple(self.squareDim), order=0)
                self.opticalImageSecDeriv = resize(opticalImageSecDeriv, tuple(self.finalDim), order=0)
            self.squareOpticalImage = resize(opticalImage, tuple(self.squareDim), order=0)
            self.opticalImage = resize(opticalImage, tuple(self.finalDim), order=0)
        
        #If a simulation or post-processing, read all the sample data and save in hdf5 if applicable, optimized for loading whole channel images, force clear the actor memory
        if self.simulationFlag or self.postFlag: 
            self.avgTimeFileLoad = self.readScanData()
            if self.dataMSI: 
                if parallelization: 
                    if self.readAllMSI: self.allImagesMax = ray.get(self.reader_MSI_Actor.writeToDisk.remote(self.squareDim))
                    del self.reader_MSI_Actor, self.mzOriginalIndices_id, self.mzRanges_id
                    if self.sampleType == 'DESI': del self.mzFinalGrid_id, self.chanFinalGrid_id
                    gc.collect()
                else:
                    if self.readAllMSI: 
                        self.allImagesMax = np.max(self.allImages, axis=(1,2))
                        allImagesFile = h5py.File(self.allImagesPath, 'a')
                        _ = allImagesFile.create_dataset(name='allImages', data=self.allImages, chunks=(1, self.finalDim[0], self.finalDim[1]), dtype=np.float32)
                        allImagesFile.close()
                        if self.sampleType=='DESI':
                            self.allImages = np.moveaxis(resize(np.moveaxis(self.allImages, 0, -1), tuple(self.squareDim), order=0), -1, 0)
                            squareAllImagesFile = h5py.File(self.squareAllImagesPath, 'a')
                            _ = squareAllImagesFile.create_dataset(name='squareAllImages', data=self.allImages, chunks=(1, self.squareDim[0], self.squareDim[1]), dtype=np.float32)
                            squareAllImagesFile.close()
                        del self.allImages
                del self.mzFinalGrid, self.chanFinalGrid
    
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
        t0_readFile = time.time()
        
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
                if 'png' in extensions: self.lineExt = '.png'
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
                    if self.readAllMSI: 
                        filtIndexLow, filtIndexHigh = bisect_left(mzs, self.mzLowerBound), bisect_right(mzs, self.mzUpperBound)
                        self.allImages[:, y, x] = np.add.reduceat(mzFastIndex(mzs[filtIndexLow:filtIndexHigh], ints[filtIndexLow:filtIndexHigh], self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount), self.mzOriginalIndices).astype(np.float32)
                    self.chanImages[:, y, x] = [np.sum(ints[bisect_left(mzs, mzRange[0]):bisect_right(mzs, mzRange[1])]) for mzRange in self.mzRanges]
            else:
                _ = ray.get(self.reader_MSI_Actor.setCoordinates.remote(coordinates))
                _ = ray.get([msi_parhelper.remote(self.reader_MSI_Actor, self.useAlphaTims, self.readAllMSI, scanFileNames, indexes, self.mzOriginalIndices_id, self.mzRanges_id, self.sampleType, self.mzLowerBound, self.mzUpperBound, self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount, self.mask, self.newTimes, self.finalDim, self.sampleWidth, self.scanRate) for indexes in np.array_split(np.arange(0, len(coordinates)), numberCPUS)])
            
            #Close the MSI file
            del data
        
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
                        if not debugMode: print('\nWarning - Failed to load any data from file: ' + scanFileName + ' This file will be ignored this iteration.')
                    
                    #Extract the file number and if unordered find corresponding line number in LUT, otherwise line number is the file number minus 1
                    if not errorFlag:
                        fileNum = int(scanFileName.split('line-')[1].split('.')[0].lstrip('0'))-1
                        if self.unorderedNames: 
                            try: 
                                lineNum = self.physicalLineNums[fileNum+1]
                            except: 
                                errorFlag = True 
                                if not debugMode: print('\nWarning - Failed to find the physical line number for the file: ' + scanFileName + ' This file will be ignored this iteration.')
                        else: lineNum = fileNum
                    
                    #If error still has not occurred 
                    if not errorFlag:
                        
                        #If ignoring missing lines and there are stored missing lines (simulation with ordered filenames only), then determine the offset for correct indexing
                        if self.ignoreMissingLines and len(self.missingLines) > 0: lineNum -= int(np.sum(lineNum > self.missingLines))
                        
                        #Add file name to those that will have been already scanned (when this process finishes)
                        self.readScanFiles.append(scanFileName)
                        
                        #Extract original measurement times and setup/read TIC data as applicable
                        if data.format == 'Bruker': 
                            sumImageLine = []
                            if not self.useAlphaTims: origTimes = np.asarray(data.ms1_frames)[:,1]/60
                            else: origTimes = np.delete(data.rt_values, 0, axis = 0)/60
                        else: 
                            imageData = np.asarray(data.xic(data.time_range()[0], data.time_range()[1]))
                            origTimes, sumImageLine = imageData[:,0], imageData[:,1]
                        
                        #Force original times memory allocation to be contigous
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
                        
                        #Read in and process spectrum data for each location
                        for pos in positions:
                            if data.format == 'Bruker':
                                if not self.useAlphaTims: 
                                    mzs, ints = data.scan(pos, True)
                                else: 
                                    mzs, ints = data[pos]['mz_values'].values, data[pos]['corrected_intensity_values'].values
                                    sortedIndices = np.argsort(mzs)
                                    mzs, ints = mzs[sortedIndices], ints[sortedIndices]
                                sumImageLine.append(np.sum(ints))
                            else:
                                mzs, ints = data.scan(pos, False, True)
                            if self.readAllMSI: 
                                filtIndexLow, filtIndexHigh = bisect_left(mzs, self.mzLowerBound), bisect_right(mzs, self.mzUpperBound)
                                mzDataLine.append(np.add.reduceat(mzFastIndex(mzs[filtIndexLow:filtIndexHigh], ints[filtIndexLow:filtIndexHigh], self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount), self.mzOriginalIndices))
                            for mzRangeNum in range(0, len(self.mzRanges)):
                                mzRange = self.mzRanges[mzRangeNum]
                                chanDataLine[mzRangeNum].append(np.sum(ints[bisect_left(mzs, mzRange[0]):bisect_right(mzs, mzRange[1])]))
                        
                        #Create regular grid interpolators, aligning all/targeted m/z data, and storing results
                        if self.readAllMSI: self.allImages[:, lineNum, :] = scipy.interpolate.RegularGridInterpolator((origTimes, self.mzFinal), np.asarray(mzDataLine, dtype='float64'), bounds_error=False, fill_value=0)(self.mzFinalGrid).astype('float32')
                        self.chanImages[:, lineNum, :] = scipy.interpolate.RegularGridInterpolator((origTimes, self.chanValues), np.asarray(chanDataLine, dtype='float64').T, bounds_error=False, fill_value=0)(self.chanFinalGrid).astype('float32')
                        self.sumImage[lineNum, :] = np.interp(self.newTimes, origTimes, np.nan_to_num(sumImageLine, nan=0, posinf=0, neginf=0), left=0, right=0)
                        
                        #Close the file
                        if not self.useAlphaTims: data.close()
                        else: del data
            
            #Otherwise read data in parallel and perform remaining interpolations of any remaining m/z data to regular grid in serial (parallel operation is too memory intensive)
            else:
                _ = ray.get([msi_parhelper.remote(self.reader_MSI_Actor, self.useAlphaTims, self.readAllMSI, scanFileNames, indexes, self.mzOriginalIndices_id, self.mzRanges_id, self.sampleType, self.mzLowerBound, self.mzUpperBound, self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount, self.mask, self.newTimes, self.finalDim, self.sampleWidth, self.scanRate, self.mzFinal, self.mzFinalGrid_id, self.chanValues, self.chanFinalGrid_id, self.impFlag, self.postFlag, impOffset, scanMethod, lineMethod, self.physicalLineNums, self.ignoreMissingLines, self.missingLines, self.unorderedNames) for indexes in np.array_split(np.arange(0, len(scanFileNames)), numberCPUS)])
                if self.readAllMSI: _ = ray.get(self.reader_MSI_Actor.interpolateDESI.remote(self.mzFinal, self.mzFinalGrid))
                for scanFileName in ray.get(self.reader_MSI_Actor.getReadScanFiles.remote()): self.readScanFiles.append(scanFileName)

        #If parallelization is enabled, and this is a MSI sample, then read MSI data in parallel, retrieve from shared memory, and process data into accessible shape
        if parallelization and self.dataMSI:
            
            #Update local identification of which files have already been imported for DESI samples
            if self.sampleType == 'DESI': self.readScanFiles = ray.get(self.reader_MSI_Actor.getReadScanFiles.remote())
            
            #If there are were not new specific locations that were to be scanned, retrieve everything, otherwise only pull data for new idxs
            if len(newIdxs) == 0: 
                self.chanImages = np.moveaxis(ray.get(self.reader_MSI_Actor.getChanImages.remote()), -1, 0)
                self.sumImage = ray.get(self.reader_MSI_Actor.getSumImage.remote())
            else: 
                self.chanImages[:, newIdxs[:,0], newIdxs[:,1]] = ray.get(self.reader_MSI_Actor.getChanImagesNewIdxs.remote(newIdxs[:,0], newIdxs[:,1])).T
                self.sumImage[newIdxs[:,0], newIdxs[:,1]] = ray.get(self.reader_MSI_Actor.getSumImageNewIdxs.remote(newIdxs[:,0], newIdxs[:,1]))
            
        #If DESI MSI, then need to resize the images to obtain square dimensionality, otherwise the square dimensions are equal to the original
        if self.sampleType == 'DESI': self.squareChanImages = np.moveaxis(resize(np.moveaxis(self.chanImages, 0, -1), tuple(self.squareDim), order=0), -1, 0)
        else: self.squareChanImages = self.chanImages
            
        #Find the maximum value in each channel image for easy referencing
        self.chanImagesMax = np.max(self.chanImages, axis=(1,2))
        
        #Stop file load timer and return average across number of files scanned
        t1_readFile = time.time()
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
        
        #Ensure newIdxs are indexible in 2 dimensions and update mask; post-processing will send empty set
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
        if not updateRD or (((erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net')) and not sampleData.bestCFlag and not sampleData.oracleFlag):
        
            #Extract measured and unmeasured locations, considering FOV mask if applicable
            tempScanData.squareMeasuredIdxs = np.transpose(np.where(self.squareMask==1))
            if sampleData.useMaskFOV: tempScanData.squareUnMeasuredIdxs = np.transpose(np.where((self.squareMask==0) & (sampleData.squareMaskFOV==1)))
            else: tempScanData.squareUnMeasuredIdxs = np.transpose(np.where(self.squareMask==0))
            
            #Determine neighbor information for unmeasured locations
            if len(tempScanData.squareUnMeasuredIdxs) > 0: findNeighbors(tempScanData)
            else: tempScanData.neighborIndices, tempScanData.neighborWeights, tempScanData.neighborDistances = [], [], []
        
        #If not just updating the RD, then compute reconstructions (using square dimensionality), resizing to physical dimensions for DESI
        if not updateRD:
            t0_computeRecon = time.time()
            if sampleData.sampleType == 'DESI':
                self.squareSumReconImage = computeReconIDW(resize(self.sumImage, tuple(sampleData.squareDim), order=0), tempScanData)
                self.squareChanReconImages = computeReconIDW(np.moveaxis(resize(np.moveaxis(self.chanImages, 0, -1), tuple(sampleData.squareDim), order=0), -1, 0), tempScanData)
                self.sumReconImage = resize(self.squareSumReconImage, tuple(sampleData.finalDim), order=0)
                self.chanReconImages = np.moveaxis(resize(np.moveaxis(self.squareChanReconImages , 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)
            else:
                self.squareSumReconImage = computeReconIDW(self.sumImage, tempScanData)
                self.squareChanReconImages = computeReconIDW(self.chanImages, tempScanData)
                self.sumReconImage = self.squareSumReconImage
                self.chanReconImages = self.squareChanReconImages
                
            #Copy back the original measured values to the reconstructions (only needed for DESI)
            if sampleData.sampleType == 'DESI':
                self.measuredIdxs = np.transpose(np.where(self.mask==1))
                self.chanReconImages[:, self.measuredIdxs[:,0], self.measuredIdxs[:,1]] = self.chanImages[:, self.measuredIdxs[:,0], self.measuredIdxs[:,1]]
                self.sumReconImage[self.measuredIdxs[:,0], self.measuredIdxs[:,1]] = self.sumImage[self.measuredIdxs[:,0], self.measuredIdxs[:,1]]
            
            t1_computeRecon = time.time()
            result.avgTimesComputeRecon.append(t1_computeRecon-t0_computeRecon)
            
        #Compute feature information for for training/utilizing SLADS models
        if (sampleData.datagenFlag or ((erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net')) and not sampleData.bestCFlag and not sampleData.oracleFlag) and len(tempScanData.squareUnMeasuredIdxs) > 0: 
            t0_computePoly = time.time()
            self.polyFeatures = [computePolyFeatures(sampleData, tempScanData, squareChanReconImage) for squareChanReconImage in self.squareChanReconImages]
            t1_computePoly = time.time()
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
            if not updateRD: 
            
                #If dataAdjust is enabled, and using DLADS or GLANDS, then can optionally rescale RDPP computation inputs to between 0 and 1 or standardize them
                if dataAdjust == 'rescale' and (erdModel=='DLADS' or erdModel=='GLANDS'):
                    minValues, maxValues = np.min(sampleData.squareChanImages, axis=(1,2)), np.max(sampleData.squareChanImages, axis=(1,2))
                    minMaxDiffs = (maxValues-minValues)
                    tempA = (np.moveaxis(sampleData.squareChanImages, 0, -1)-minValues)
                    tempB = (np.moveaxis(self.squareChanReconImages, 0, -1)-minValues)
                    self.RDPPs = np.moveaxis(abs(np.divide(tempA, minMaxDiffs, out=np.zeros_like(tempA), where=minMaxDiffs!=0)-np.divide(tempB, minMaxDiffs, out=np.zeros_like(tempB), where=minMaxDiffs!=0)), -1, 0) 
                elif dataAdjust == 'standardize' and (erdModel=='DLADS' or erdModel=='GLANDS'):
                    meanValues, stdValues =  np.mean(sampleData.squareChanImages, axis=(1,2)), np.std(sampleData.squareChanImages, axis=(1,2))
                    tempA = (np.moveaxis(sampleData.squareChanImages, 0, -1)-meanValues)
                    tempB = (np.moveaxis(self.squareChanReconImages, 0, -1)-meanValues)
                    self.RDPPs = np.moveaxis(abs(np.divide(tempA, stdValues, out=np.zeros_like(tempA), where=stdValues!=0)-np.divide(tempB, stdValues, out=np.zeros_like(tempB), where=stdValues!=0)), -1, 0) 
                else: self.RDPPs = abs(sampleData.squareChanImages-self.squareChanReconImages)
        
            #Compute/Update the RD and use it in place of an ERD
            t0_computeRD = time.time()
            computeRD(self, sampleData, tempScanData, cValue, updateLocations)
            t1_computeRD = time.time()
            if not updateRD: result.avgTimesComputeRD.append(t1_computeRD-t0_computeRD)
            self.squareRDValues = self.squareRDs[:, tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]]
            self.squareERD, self.squareERDs = self.squareRD, self.squareRDs
            self.ERD, self.ERDs = self.RD, self.RDs
        
        #If there are unmeasured locations left and the ground-truth data isn't known, compute and process the ERD
        else: 
            t0_computeERD = time.time()
            computeERD(self, sampleData, tempScanData, model)
            t1_computeERD = time.time()
            result.avgTimesComputeERD.append((t1_computeERD-t0_computeERD)+polyComputeTime)
        
        #Duplicate the per-channel E/RDs for processing, masking to just the unmeasured locations
        self.processedERDs = copy.deepcopy(self.ERDs)*(1-self.mask)
        
        #Mask the E/RDs to the FOV foreground mask if applicable
        if sampleData.useMaskFOV: self.processedERDs *= sampleData.maskFOV
        
        #Mask out visited lines if linewise and line revisiting is disabled
        if sampleData.scanMethod == 'linewise' and not sampleData.lineRevist: self.processedERDs[:, np.where(np.sum(self.mask, axis=1)>0)[0], :] = 0
        
        #Bias E/RDs by an optical image if applicable
        if sampleData.dataMSI and (applyOptical == 'directBias' or applyOptical == 'secDerivBias'): 
            if applyOptical == 'secDerivBias': self.processedERDs *= sampleData.opticalImageSecDeriv
            else: self.processedERDs *= sampleData.opticalImage
        
        #Remove any negative/nan/inf values
        self.processedERDs[self.processedERDs<0] = 0
        self.processedERDs = np.nan_to_num(self.processedERDs, nan=0, posinf=0, neginf=0)
        
        #Rescale each channel to between 0 and 1
        #minValues = np.min(self.processedERDs, axis=(1,2))
        #diffValues = np.max(self.processedERDs, axis=(1,2))-minValues
        #tempA = (np.moveaxis(self.processedERDs, 0, -1)-minValues)
        #self.processedERDs = np.moveaxis(np.divide(tempA, diffValues, out=np.zeros_like(tempA), where=diffValues!=0), -1, 0)
        
        #Average across all channels
        self.processedERD = np.mean(self.processedERDs, axis=0)
        
        #Rescale processed final ERD for potential Otsu selection
        if np.max(self.processedERD) != 0: self.processedERD = ((self.processedERD-np.min(self.processedERD))/(np.max(self.processedERD)-np.min(self.processedERD)))
        
#Sample scanning progress and final results processing
class Result:
    def __init__(self, sampleData, dir_Results, cValue):
        self.startTime = time.time()
        self.finalTime = time.time()
        self.sampleData = sampleData
        self.dir_Results = dir_Results
        self.cValue = cValue
        self.samples = []
        self.cSelectionList = []
        self.percsMeasured = []
        self.avgTimesComputeRD = []
        self.avgTimesComputeERD = []
        self.avgTimesComputeRecon = []
        self.avgTimesFileLoad = []
        self.lastMask = None
        self.avgTimeComputeRD = 0
        self.avgTimeComputeERD = 0
        self.avgTimeComputeRecon = 0
        self.avgTimeFileLoad = 0
        
        #If there is to be a results directory, then ensure it is setup
        if self.dir_Results != None:

            #Setup/clean base sample directory
            self.dir_sampleResults = self.dir_Results + self.sampleData.name + os.path.sep
            if os.path.exists(self.dir_sampleResults): shutil.rmtree(self.dir_sampleResults)
            os.makedirs(self.dir_sampleResults)
            
            #Prepare subdirectories; for frames and videos of channel progressions; #Note: cannot recall why videos/animations was disabled for post, so re-enabled...
            self.dir_chanProgression = self.dir_sampleResults + 'Channels' + os.path.sep
            os.makedirs(self.dir_chanProgression)
            self.dir_chanProgressions = [self.dir_chanProgression + str(self.sampleData.chanValues[chanNum]) + os.path.sep for chanNum in range(0, len(self.sampleData.chanValues))]
            for dir_chanProgressionsub in self.dir_chanProgressions: 
                try: os.makedirs(dir_chanProgressionsub)
                except: print('Folder already exists')
            self.dir_progression = self.dir_sampleResults + 'Progression' + os.path.sep
            os.makedirs(self.dir_progression)
            #if not self.sampleData.postFlag:
            self.dir_videos = self.dir_sampleResults + 'Videos' + os.path.sep
            os.makedirs(self.dir_videos)
            
        #MSI Specific; if this is a simulation with live output enabled, then load all images data
        if self.sampleData.dataMSI and self.sampleData.simulationFlag and self.sampleData.liveOutputFlag:
            
            #If operating in parallel, create actors for reconstruction and load portions of the data into each, otherwise load data into main memory
            if parallelization:
                self.recon_Actors = [Recon_Actor.remote(indexes, self.sampleData.sampleType, self.sampleData.squareDim, self.sampleData.finalDim, self.sampleData.allImagesMax) for indexes in np.array_split(np.arange(0, len(self.sampleData.mzFinal)), numberCPUS)]
                #If performing a non-training simulation, then potentially could pass a ray object id rather than reading in from hdf5...
                #_ = [ray.get(reconIDW_Actor.setupFromShared.remote(self.allImages_id)) for reconIDW_Actor in reconIDW_Actors]
            else:
                self.sampleData.allImagesFile = h5py.File(self.sampleData.allImagesPath, 'a')
                self.sampleData.allImages = self.sampleData.allImagesFile['allImages'][:]
        
    def update(self, sample, completedRunFlag):
    
        #Update measurement mask and progression map
        self.lastMask = copy.deepcopy(sample.mask)
        self.lastProgMap = copy.deepcopy(sample.progMap)
        
        #If optimizing c, then don't store sample data from the first iteration (except for the last mask), since it was not used in the determination of initial set locations
        if self.sampleData.bestCFlag and sample.iteration == 1: return
        
        #Update the percentage of FOV measured
        self.percsMeasured.append(copy.deepcopy(sample.percMeasured))
        
        #If outputs should be produced at every update step, then do so, determining related metrics as needed
        if self.sampleData.liveOutputFlag: 
            if self.sampleData.simulationFlag: self.extractSimulationData(sample, self.sampleData)
            visualizeStep(sample, self.sampleData, self.dir_progression, self.dir_chanProgressions)
        
        #Save a copy of the measurement step for later evaluation
        self.samples.append(copy.deepcopy(sample))
        
        #When applicable, save the physicalLineNums.csv, measuredMask.csv, and progressMap.csv to the same folder as the scanned MSI files and the results folder; otherwise just save to results
        if (completedRunFlag or saveIterationFlag) and self.dir_Results != None: 
            if self.sampleData.impFlag: #not self.sampleData.simulationFlag and not self.sampleData.postFlag:
                if self.sampleData.unorderedNames: np.savetxt(dir_ImpDataFinal+'physicalLineNums.csv', np.asarray(list(self.sampleData.physicalLineNums.items())), delimiter=',', fmt='%d')
                np.savetxt(dir_ImpDataFinal+'measuredMask.csv', self.lastMask, delimiter=',', fmt='%d')
                if len(self.lastProgMap.shape)>0: np.savetxt(dir_ImpDataFinal+'progressMap.csv', np.nan_to_num(self.lastProgMap, nan=-1), delimiter=',', fmt='%d')
            if self.sampleData.unorderedNames: np.savetxt(self.dir_sampleResults+'physicalLineNums.csv', np.asarray(list(self.sampleData.physicalLineNums.items())), delimiter=',', fmt='%d')
            np.savetxt(self.dir_sampleResults+'measuredMask.csv', self.lastMask, delimiter=',', fmt='%d')
            if len(self.lastProgMap.shape)>0: np.savetxt(self.dir_sampleResults+'progressMap.csv', np.nan_to_num(self.lastProgMap, nan=-1), delimiter=',', fmt='%d')
        
        #Store the final scan time if run has completed
        if completedRunFlag: self.finalTime = time.time()-self.startTime
    
    #For a given measurement step find PSNR/SSIM of reconstructions, compute the RD, find PSNR of ERD
    def extractSimulationData(self, sample, lastReconOnly=False):
        
        #Create a segmented storage object for variables that must be referenced
        tempScanData = TempScanData()
        
        #Extract measured and unmeasured locations for the measured mask
        tempScanData.squareMeasuredIdxs = np.transpose(np.where(sample.squareMask==1))
        if self.sampleData.useMaskFOV: tempScanData.squareUnMeasuredIdxs = np.transpose(np.where((sample.squareMask==0) & (self.sampleData.squareMaskFOV==1)))
        else: tempScanData.squareUnMeasuredIdxs = np.transpose(np.where(sample.squareMask==0))
                
        #Determine neighbor information for unmeasured locations
        if len(tempScanData.squareUnMeasuredIdxs) > 0: findNeighbors(tempScanData)
        else: tempScanData.neighborIndices, tempScanData.neighborWeights, tempScanData.neighborDistances = [], [], []
        
        #Find PSNR/SSIM scores for all channel reconstructions
        if not lastReconOnly:
            sample.chanImagesPSNRList = [compare_psnr(self.sampleData.chanImages[index], sample.chanReconImages[index], data_range=self.sampleData.chanImagesMax[index]) for index in range(0, len(self.sampleData.chanImages))]
            sample.sumImagePSNR = compare_psnr(self.sampleData.sumImage, sample.sumReconImage, data_range=np.max(self.sampleData.sumImage))
            sample.chanImagesSSIMList = [compare_ssim(self.sampleData.chanImages[index], sample.chanReconImages[index], data_range=self.sampleData.chanImagesMax[index]) for index in range(0, len(self.sampleData.chanImages))]
            sample.sumImageSSIM = compare_ssim(self.sampleData.sumImage, sample.sumReconImage, data_range=np.max(self.sampleData.sumImage))
        
        #MSI Specific; if enabled then perform and evaluate reconstructions over the whole spectrum for the data known at the given measurement step
        if (self.sampleData.allChanEvalFlag or lastReconOnly) and self.sampleData.dataMSI:
            
            #If operating in parallel, utilize actors created in the complete() method, or at initialization for live simulation; for serial, could vectorize, but extremely RAM intensive and usable batch size is unpredictable per system
            if parallelization:
                tempScanData_id, squareMask_id, mask_id = ray.put(tempScanData), ray.put(sample.squareMask), ray.put(sample.mask)
                _ = ray.get([recon_Actor.applyMask.remote(squareMask_id) for recon_Actor in self.recon_Actors])
                _ = ray.get([recon_Actor.computeRecon.remote(tempScanData_id, mask_id) for recon_Actor in self.recon_Actors])
                _ = ray.get([recon_Actor.computeMetrics.remote() for recon_Actor in self.recon_Actors])
                if not lastReconOnly:
                    sample.allImagesPSNRList = np.concatenate([ray.get(recon_Actor.getPSNR.remote()) for recon_Actor in self.recon_Actors])
                    sample.allImagesSSIMList = np.concatenate([ray.get(recon_Actor.getSSIM.remote()) for recon_Actor in self.recon_Actors])
                del tempScanData_id, squareMask_id
            else:
                if self.sampleData.sampleType == 'DESI': self.reconImages = self.sampleData.squareAllImages*sample.squareMask
                else: self.reconImages = self.sampleData.allImages*sample.squareMask
                self.reconImages = np.array([computeReconIDW(self.reconImages[index], tempScanData) for index in range(0, len(self.reconImages))], dtype=np.float32)
                if self.sampleData.sampleType == 'DESI': self.reconImages = np.moveaxis(resize(np.moveaxis(self.reconImages, 0, -1), tuple(self.sampleData.finalDim), order=0), -1, 0)
                
                #Copy back the original measured values to the reconstructions (only needed for DESI)
                if self.sampleData.sampleType == 'DESI': 
                    measuredIdxs = np.transpose(np.where(sample.mask==1))
                    self.reconImages[:, measuredIdxs[:,0], measuredIdxs[:,1]] = self.sampleData.allImages[:, measuredIdxs[:,0], measuredIdxs[:,1]]
                
                if not lastReconOnly:
                    sample.allImagesPSNRList = [compare_psnr(self.sampleData.allImages[index], self.reconImages[index], data_range=self.sampleData.allImagesMax[index]) for index in range(0, len(self.sampleData.allImages))]
                    sample.allImagesSSIMList = [compare_ssim(self.sampleData.allImages[index], self.reconImages[index], data_range=self.sampleData.allImagesMax[index]) for index in range(0, len(self.sampleData.allImages))]
        
        #Otherwise assume all images results are the same as for targeted channels; i.e. all channels were targeted
        elif not lastReconOnly:
            sample.allImagesPSNRList = sample.chanImagesPSNRList
            sample.allImagesSSIMList = sample.chanImagesSSIMList
            
        #Prior to and for model training there is RD, but no ERD
        if self.sampleData.simulationFlag and not self.sampleData.trainFlag and not lastReconOnly:
            
            #Compute RD; if every location has been scanned all positions are zero
            if len(tempScanData.squareUnMeasuredIdxs) == 0: 
                sample.squareRD = np.zeros(self.sampleData.squareDim, dtype=np.float32)
                sample.RD = np.zeros(self.sampleData.finalDim, dtype=np.float32)
            else: 
                #If dataAdjust is enabled, and using either DLADS or GLANDS, then can optionally rescale RDPP computation inputs to between 0 and 1 or standardize them
                if dataAdjust == 'rescale' and (erdModel=='DLADS' or erdModel=='GLANDS'):
                    minValues, maxValues = np.min(self.sampleData.squareChanImages, axis=(1,2)), np.max(self.sampleData.squareChanImages, axis=(1,2))
                    minMaxDiffs = (maxValues-minValues)
                    tempA = (np.moveaxis(self.sampleData.squareChanImages, 0, -1)-minValues)
                    tempB = (np.moveaxis(sample.squareChanReconImages, 0, -1)-minValues)
                    sample.RDPPs = np.moveaxis(abs(np.divide(tempA, minMaxDiffs, out=np.zeros_like(tempA), where=minMaxDiffs!=0)-np.divide(tempB, minMaxDiffs, out=np.zeros_like(tempB), where=minMaxDiffs!=0)), -1, 0) 
                elif dataAdjust == 'standardize' and (erdModel=='DLADS' or erdModel=='GLANDS'):
                    meanValues, stdValues =  np.mean(self.sampleData.squareChanImages, axis=(1,2)), np.std(self.sampleData.squareChanImages, axis=(1,2))
                    tempA = (np.moveaxis(self.sampleData.squareChanImages, 0, -1)-meanValues)
                    tempB = (np.moveaxis(sample.squareChanReconImages, 0, -1)-meanValues)
                    sample.RDPPs = np.moveaxis(abs(np.divide(tempA, stdValues, out=np.zeros_like(tempA), where=stdValues!=0)-np.divide(tempB, stdValues, out=np.zeros_like(tempB), where=stdValues!=0)), -1, 0) 
                else: sample.RDPPs = abs(self.sampleData.squareChanImages-sample.squareChanReconImages)
                
                computeRD(sample, self.sampleData, tempScanData, self.cValue, [])

            #Determine SSIM/PSNR between averaged RD and ERD
            maxRangeValue = np.max([sample.squareRD, sample.squareERD])
            sample.ERDPSNR = compare_psnr(sample.squareRD, sample.squareERD, data_range=maxRangeValue)
            sample.ERDSSIM = compare_ssim(sample.squareRD, sample.squareERD, data_range=maxRangeValue)
        
            #Resize RD(s) for final visualizations; has to be done here for live output case, but in complete() method otherwise
            if self.sampleData.liveOutputFlag: self.resizeRD(sample)

    #Resize RD(s) for final visualization if DESI, otherwise set variable name 
    def resizeRD(self, sample):
        if self.sampleData.sampleType == 'DESI':
            sample.RD = resize(sample.squareRD, tuple(self.sampleData.finalDim), order=0)*(1-sample.mask)
            sample.RDs = np.moveaxis(resize(np.moveaxis(sample.squareRDs , 0, -1), tuple(self.sampleData.finalDim), order=0), -1, 0)*(1-sample.mask)
        else:
            sample.RD = sample.squareRD
            sample.RDs = sample.squareRDs
            
    #Generate visualiations/metrics as needed at the end of scanning
    def complete(self):
        
        #If the data was loaded during initialization of the sample data, pull the stored avg. file read time, otherwise compute value
        if self.sampleData.postFlag or self.sampleData.simulationFlag: self.avgTimeFileLoad = self.sampleData.avgTimeFileLoad
        elif len(self.avgTimesFileLoad) > 0: self.avgTimeFileLoad = np.nanmean(self.avgTimesFileLoad)
        
        #If applicable, compute average computation times
        if len(self.avgTimesComputeRecon) > 0: self.avgTimeComputeRecon = np.mean(self.avgTimesComputeRecon)
        if len(self.avgTimesComputeERD) > 0: self.avgTimeComputeERD = np.mean(self.avgTimesComputeERD)
        if len(self.avgTimesComputeRD) > 0: self.avgTimeComputeRD = np.mean(self.avgTimesComputeRD)
        
        #If performing a benchmark where processing is not needed, return before processing
        if benchmarkNoProcessing: return
        
        #Make sure samples is writable
        self.samples = copy.deepcopy(self.samples)
        
        #If this is an MSI sample
        if self.sampleData.dataMSI:
        
            #If all channel reconstructions are needed, then setup actors if in parallel, or load data into main memory
            if self.sampleData.allChanEvalFlag or (self.sampleData.imzMLExportFlag and not self.sampleData.trainFlag):
                if parallelization:
                    self.recon_Actors = [Recon_Actor.remote(indexes, self.sampleData.sampleType, self.sampleData.squareDim, self.sampleData.finalDim, self.sampleData.allImagesMax) for indexes in np.array_split(np.arange(0, len(self.sampleData.mzFinal)), numberCPUS)]
                    _ = ray.get([recon_Actor.setup.remote(self.sampleData.allImagesPath, self.sampleData.squareAllImagesPath) for recon_Actor in self.recon_Actors])
                    #_ = [ray.get(recon_Actor.setup.remote(self.sampleData.allImagesPath, self.sampleData.squareAllImagesPath)) for recon_Actor in self.recon_Actors]
                    #If performing a non-training simulation, then potentially could pass a ray object id rather than reading in from hdf5...
                    #_ = [ray.get(reconIDW_Actor.setupFromShared.remote(self.allImages_id)) for reconIDW_Actor in reconIDW_Actors]
                else:
                    self.sampleData.allImagesFile = h5py.File(self.sampleData.allImagesPath, 'r')
                    self.sampleData.allImages = self.sampleData.allImagesFile['allImages']
                    if self.sampleData.sampleType == 'DESI':
                        self.sampleData.squareAllImagesFile = h5py.File(self.sampleData.squareAllImagesPath, 'r')
                        self.sampleData.squareAllImages = self.sampleData.squareAllImagesFile['squareAllImages']
        
        #Extract metrics for samples if a simulation, and neither already done live, nor creating samples for training
        if (self.sampleData.simulationFlag and not self.sampleData.liveOutputFlag and not self.sampleData.datagenFlag):
            for sample in tqdm(self.samples, desc='RD/Metrics Extraction', leave=False, ascii=asciiFlag): self.extractSimulationData(sample)
        
        #If not evaluating all channels, but the final reconstructions for all channels are still to be generated
        if self.sampleData.imzMLExportFlag and not self.sampleData.allChanEvalFlag: self.extractSimulationData(self.samples[-1], imzMLExport)
        
        #If this is an MSI sample
        if self.sampleData.dataMSI:
            
            #If exporting final reconstruction data to .imzML
            if self.sampleData.imzMLExportFlag:
                
                #Set the coordinates to save values for
                coordinates = list(map(tuple, list(np.ndindex(tuple(self.sampleData.finalDim)))))
                
                #Export all measured, reconstructed data in .imzML format
                if parallelization: self.reconImages = np.concatenate([ray.get(recon_Actor.getReconImages.remote()) for recon_Actor in self.recon_Actors])
                writer = ImzMLWriter(self.dir_sampleResults+self.sampleData.name+'_reconstructed', intensity_dtype=np.float32, mz_dtype=np.float32, spec_type='profile', mode='processed')
                _  = [writer.addSpectrum(self.sampleData.mzFinal, self.reconImages[:, coord[0], coord[1]], (coord[1]+1, coord[0]+1)) for coord in coordinates]
                writer.close()
                del self.reconImages, writer
                
                #Export the equivalent ground-truth measured data here to .imzML format if needed
                #allImages = np.concatenate([ray.get(recon_Actor.getAllImages.remote()) for recon_Actor in self.recon_Actors])
                #writer = ImzMLWriter(self.dir_sampleResults+self.sampleData.name+'_groundTruth', intensity_dtype=np.float32, mz_dtype=np.float32, spec_type='profile', mode='processed')
                #for coord in coordinates: writer.addSpectrum(self.sampleData.mzFinal, allImages[:, coord[0], coord[1]], (coord[1]+1, coord[0]+1))
                #writer.close()
                #del allImages, writer
            
            #If all channel evaluation or imzMLExportFlag, close all images file reference if applicable, remove all recon images from memory, purge/reset ray
            if self.sampleData.allChanEvalFlag or self.sampleData.imzMLExportFlag:
                if not parallelization:
                    self.sampleData.allImagesFile.close()
                    del self.sampleData.allImages, self.sampleData.allImagesFile
                    if self.sampleData.sampleType == 'DESI':
                        self.sampleData.squareAllImagesFile.close()
                        del self.sampleData.squareAllImages, self.sampleData.squareAllImagesFile
                else:
                    _ = [ray.get(recon_Actor.closeAllImages.remote()) for recon_Actor in self.recon_Actors]
                    self.recon_Actors.clear()
                    del self.recon_Actors
                
        #If this is a simulation, not for training database generation, then summarize PSNR/SSIM scores across all measurement steps
        if self.sampleData.simulationFlag and not self.sampleData.datagenFlag:
            self.chanAvgPSNRList = [np.nanmean(sample.chanImagesPSNRList) for sample in self.samples]
            self.sumImagePSNRList = [sample.sumImagePSNR for sample in self.samples]
            self.chanAvgSSIMList = [np.nanmean(sample.chanImagesSSIMList) for sample in self.samples]
            self.sumImageSSIMList = [sample.sumImageSSIM for sample in self.samples]
            
            #Compute all channel results if applicable
            if self.sampleData.allChanEvalFlag and self.sampleData.dataMSI:
                self.allAvgPSNRList = [np.nanmean(sample.allImagesPSNRList) for sample in self.samples]
                self.allAvgSSIMList = [np.nanmean(sample.allImagesSSIMList) for sample in self.samples]

        #If ERD and RD were computed (i.e., when not a training run) summarize ERD PSNR/SSIM scores
        if self.sampleData.simulationFlag and not self.sampleData.trainFlag: 
            self.ERDPSNRList = [sample.ERDPSNR for sample in self.samples]
            self.ERDSSIMList = [sample.ERDSSIM for sample in self.samples]
        
        #Do not generate visuals for c value optimization or in training if visualizeTrainingData disabled
        if not self.sampleData.bestCFlag and ((self.sampleData.datagenFlag and visualizeTrainingData) or not self.sampleData.datagenFlag): 
            
            #generate visualizations if they were not created during operation
            if not self.sampleData.liveOutputFlag:
                
                #Resize RDs
                for sample in self.samples: self.resizeRD(sample)
                
                if parallelization:
                    
                    #Setup an actor to hold global sampling progress across multiple processes
                    samplingProgress_Actor = SamplingProgress_Actor.remote()
            
                    #Setup visualization jobs and determine total amount of work that is going to be done
                    futures = [(self.samples[index], self.sampleData, self.dir_progression, self.dir_chanProgressions, parallelization, samplingProgress_Actor) for index in range(0, len(self.samples))]
                    maxProgress = len(futures)
                    
                    #Initialize a global progress bar and start parallel sampling operations
                    pbar = tqdm(total=maxProgress, desc = 'Visualizing', leave=False, ascii=asciiFlag)                    
                    computePool = Pool(numberCPUS)
                    results = computePool.starmap_async(visualizeStep_parhelper, futures)
                    computePool.close()
                    
                    #While some results have yet to be returned, regularly update the global progress bar, then obtain results and purge/reset ray
                    pbar.n = 0
                    pbar.refresh()
                    while (True):
                        pbar.n = np.clip(round(ray.get(samplingProgress_Actor.getCurrent.remote()),0), 0, maxProgress)
                        pbar.refresh()
                        if results.ready(): 
                            pbar.n = maxProgress
                            pbar.refresh()
                            pbar.close()
                            break
                        time.sleep(0.1)
                    computePool.join()
                    resetRay(numberCPUS)
                else: 
                    _ = [visualizeStep(sample, self.sampleData, self.dir_progression, self.dir_chanProgressions) for sample in tqdm(self.samples, desc='Visualizing', leave=False, ascii=asciiFlag)]

            #Combine total progression and individual channel images into animations
            dataFileNames = natsort.natsorted(glob.glob(self.dir_progression + 'progression_*.png'))
            height, width, layers = cv2.imread(dataFileNames[0]).shape
            animation = cv2.VideoWriter(self.dir_videos + 'progression.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (width, height))
            for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
            animation.release()
            animation = None
            for chanNum in tqdm(range(0, len(self.sampleData.chanValues)), desc='Channel Videos', leave = False, ascii=asciiFlag): 
                dataFileNames = natsort.natsorted(glob.glob(self.dir_chanProgressions[chanNum] + 'progression_*.png'))
                height, width, layers = cv2.imread(dataFileNames[0]).shape
                animation = cv2.VideoWriter(self.dir_videos + str(self.sampleData.chanValues[chanNum]) + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (width, height))
                for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                animation.release()
                animation = None

#Visualize single sample progression step
def visualizeStep(sample, sampleData, dir_progression, dir_chanProgressions, parallelization=False, samplingProgress_Actor=None):

    #Turn percent measured into a string
    percMeasured = "{:.2f}".format(sample.percMeasured)
    
    #Determine known information to be visualized
    if sampleData.impFlag or sampleData.postFlag: knownGT, knownRD, knownERD = False, False, True
    elif sampleData.trainFlag: knownGT, knownRD, knownERD = True, True, False
    elif sampleData.simulationFlag: knownGT, knownRD, knownERD = True, True, True
    
    #Turn ground-truth metrics into strings and set flag to indicate availability
    if knownGT:
        if not sampleData.datagenFlag:
            sumImagePSNR = "{:.2f}".format(sample.sumImagePSNR)
            sumImageSSIM = "{:.2f}".format(sample.sumImageSSIM)
            chanImageAvgPSNR = "{:.2f}".format(np.nanmean(sample.chanImagesPSNRList))
            chanImageAvgSSIM = "{:.2f}".format(np.nanmean(sample.chanImagesSSIMList))
        else:
            sumImagePSNR = "N/A"
            sumImageSSIM = "N/A"
            chanImageAvgPSNR = "N/A"
            chanImageAvgSSIM = "N/A"
        if sampleData.allChanEvalFlag and not sampleData.datagenFlag and sampleData.dataMSI: 
            allImageAvgPSNR = "{:.2f}".format(np.nanmean(sample.allImagesPSNRList))
            allImageAvgSSIM = "{:.2f}".format(np.nanmean(sample.allImagesSSIMList))
        else: 
            allImageAvgPSNR = "N/A"
            allImageAvgSSIM = "N/A"
    
    #Turn RD metrics into strings and set flag to indicate availability
    if knownRD and knownERD:
        erdPSNR = "{:.2f}".format(sample.ERDPSNR)
        erdSSIM = "{:.2f}".format(sample.ERDSSIM)
    
    #Setup measurement progression image variables if needed
    if len(sample.progMap.shape)>0:
        progMapValues = np.unique(sample.progMap[~np.isnan(sample.progMap)]).astype(int)
        cmapProgMap = plt.get_cmap('autumn', len(progMapValues))
        cmapProgMap.set_bad(color='black')
        boundValuesProgMap = np.linspace(1, progMapValues.max()+1, len(progMapValues)+1, dtype=int)
        normProgMap = matplotlib.colors.BoundaryNorm(boundValuesProgMap, cmapProgMap.N)
    
    #For each of the channels, generate visuals
    for chanNum in range(0, sampleData.numChannels):
        
        #Find minimum and maximum channel values for colorbars
        chanMinValue, chanMaxValue = np.min(sampleData.chanImages[chanNum]), np.max(sampleData.chanImages[chanNum])
        
        #Turn metrics into strings
        chanLabel = str(sampleData.chanValues[chanNum])
        if knownGT:
            if not sampleData.datagenFlag:
                chanImagesPSNR = "{:.2f}".format(sample.chanImagesPSNRList[chanNum])
                chanImagesSSIM = "{:.2f}".format(sample.chanImagesSSIMList[chanNum])
            else: 
                chanImagesPSNR = "N/A"
                chanImagesSSIM = "N/A"
        
        #Create a new figure
        if sampleData.impFlag or sampleData.postFlag: f = plt.figure(figsize=(30,5))
        else: f = plt.figure(figsize=(25,10))
        
        #Generate and apply a plot title, with metrics if applicable
        plotTitle = r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Channel:\ }$" + chanLabel + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured
        if knownGT:
            plotTitle += '\n' + r"$\bf{PSNR\ -\ All\ Channel\ Avg:\ }$" + allImageAvgPSNR + r"$\bf{\ \ Targeted\ Channel\ Avg:\ }$" + chanImageAvgPSNR + r"$\bf{\ \ Targeted\ Channel:\ }$" + chanImagesPSNR
            plotTitle += '\n' + r"$\bf{SSIM\ -\ All\ Channel\ Avg:\ }$" + allImageAvgSSIM + r"$\bf{\ \ Targeted\ Channel\ Avg:\ }$" + chanImageAvgSSIM + r"$\bf{\ \ Targeted\ Channel:\ }$" + chanImagesSSIM
        plt.suptitle(plotTitle)
        
        if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,3))
        else: ax = plt.subplot2grid((2,4), (0,0))
        im = ax.imshow(sampleData.chanImages[chanNum]*sample.mask, cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue, interpolation='none')
        ax.set_title('Measured')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,4))
        else: ax = plt.subplot2grid((2,4), (0,1))
        im = ax.imshow(sample.chanReconImages[chanNum], cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue, interpolation='none')
        ax.set_title('Reconstruction')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        if knownGT: 
            ax = plt.subplot2grid((2,4), (0,2))
            im = ax.imshow(sampleData.chanImages[chanNum], cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue, interpolation='none')
            ax.set_title('Ground-Truth')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            
            ax = plt.subplot2grid((2,4), (0,3))
            im = ax.imshow(abs(sampleData.chanImages[chanNum]-sample.chanReconImages[chanNum]), cmap='hot', aspect='auto', interpolation='none')
            ax.set_title('Absolute Difference')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,2))
        elif sampleData.trainFlag: ax = plt.subplot2grid((2,4), (1,1))
        elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,0))
        if len(sample.progMap.shape)>0:
            im = ax.imshow(sample.progMap+0.5, cmap=cmapProgMap, norm=normProgMap, aspect='auto', interpolation='none')
            ax.set_title('Measurement Progression')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            cbar.ax.minorticks_off()
            tickValues = np.linspace(1, progMapValues.max(), len(cbar.get_ticks()), dtype=int)
            cbar.set_ticks(tickValues+0.5)
            cbar.ax.set_yticklabels(list(map(str, tickValues)))
        else:
            im = ax.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1, interpolation='none')
            ax.set_title('Measurement Mask')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        if knownRD:
            if sampleData.trainFlag: ax = plt.subplot2grid((2,4), (1,2))
            elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,1))
            im = ax.imshow(sample.RDs[chanNum], cmap='viridis', aspect='auto', vmin=0, interpolation='none')
            ax.set_title('RD')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        if knownERD:
            if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,0))
            elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,2))
            im = ax.imshow(sample.ERDs[chanNum], cmap='viridis', aspect='auto', vmin=0, interpolation='none')
            ax.set_title('ERD')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            
            if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,1))
            elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,3))
            im = ax.imshow(np.nan_to_num(sample.processedERDs[chanNum], nan=0, posinf=0, neginf=0), cmap='viridis', aspect='auto', vmin=0, interpolation='none')
            ax.set_title('Processed ERD')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        #Save
        f.tight_layout()
        if not(sampleData.impFlag or sampleData.postFlag): f.subplots_adjust(top = 0.85)
        saveLocation = dir_chanProgressions[chanNum] + 'progression_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) +'.png'
        plt.savefig(saveLocation)
        plt.close()

        #Do borderless saves for each channel image here; skip mask/progMap as they will be produced in the progression output
        if knownERD:
            saveLocation = dir_chanProgressions[chanNum] + 'erd_original_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
            borderlessPlot(sample.ERDs[chanNum], saveLocation, aspect='auto', cmap='viridis', vmin=0)
            
            saveLocation = dir_chanProgressions[chanNum] + 'erd_processed_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
            borderlessPlot(np.nan_to_num(sample.processedERDs[chanNum], nan=0, posinf=0, neginf=0), saveLocation, aspect='auto', cmap='viridis', vmin=0)
        
        if knownRD:
            saveLocation = dir_chanProgressions[chanNum] + 'rd_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
            borderlessPlot(sample.RDs[chanNum], saveLocation, aspect='auto', cmap='viridis', vmin=0)
        
        if knownGT:
            saveLocation = dir_chanProgressions[chanNum] + 'groundTruth_channel_' + chanLabel + '.png'
            borderlessPlot(sampleData.chanImages[chanNum], saveLocation, cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue)

        saveLocation = dir_chanProgressions[chanNum] + 'reconstruction_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.chanReconImages[chanNum], saveLocation, cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue)
        
        saveLocation = dir_chanProgressions[chanNum] + 'measured_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.chanImages[chanNum]*sample.mask, saveLocation, cmap='hot', aspect='auto', vmin=chanMinValue, vmax=chanMaxValue)
        
    #For the overall progression, get min/max of the ground-truth sum image for visualization
    sumImageMinValue, sumImageMaxValue = np.min(sampleData.sumImage), np.max(sampleData.sumImage)
    
    #Create a new figure
    if sampleData.impFlag or sampleData.postFlag: f = plt.figure(figsize=(30,5))
    else: f = plt.figure(figsize=(25,10))

    #Generate and apply a plot title, with metrics if applicable
    plotTitle = r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured
    if knownGT:
        plotTitle += '\n' + r"$\bf{PSNR\ -\ All\ Channel\ Avg:\ }$" + allImageAvgPSNR + r"$\bf{\ \ Targeted\ Channel\ Avg:\ }$" + chanImageAvgPSNR + r"$\bf{\ \ Sum\ Image: }$" + sumImagePSNR
        plotTitle += '\n' + r"$\bf{SSIM\ -\ All\ Channel\ Avg:\ }$" + allImageAvgSSIM + r"$\bf{\ \ Targeted\ Channel\ Avg:\ }$" + chanImageAvgSSIM + r"$\bf{\ \ Sum\ Image: }$" + sumImageSSIM
    if knownRD and knownERD:
        plotTitle += r"$\bf{\ \ Avg ERD:\ }$" + erdPSNR 
        plotTitle += r"$\bf{\ \ Avg ERD:\ }$" + erdSSIM
    
    plt.suptitle(plotTitle)
    
    if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,3))
    else: ax = plt.subplot2grid((2,4), (0,0))
    im = ax.imshow(sampleData.sumImage*sample.mask, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue, interpolation='none')
    ax.set_title('Measured')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,4))
    else: ax = plt.subplot2grid((2,4), (0,1))
    im = ax.imshow(sample.sumReconImage, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue, interpolation='none')
    ax.set_title('Sum Image Reconstruction')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    if knownGT: 
        ax = plt.subplot2grid((2,4), (0,2))
        im = ax.imshow(sampleData.sumImage, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue, interpolation='none')
        ax.set_title('Sum Image Ground-Truth')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        ax = plt.subplot2grid((2,4), (0,3))
        im = ax.imshow(abs(sampleData.sumImage-sample.sumReconImage), cmap='hot', aspect='auto', interpolation='none')
        ax.set_title('Absolute Difference')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,2))
    elif sampleData.trainFlag: ax = plt.subplot2grid((2,4), (1,1))
    elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,0))
    if len(sample.progMap.shape)>0:
        im = ax.imshow(sample.progMap+0.5, cmap=cmapProgMap, norm=normProgMap, aspect='auto', interpolation='none')
        ax.set_title('Measurement Progression')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.ax.minorticks_off()
        tickValues = np.linspace(1, progMapValues.max(), len(cbar.get_ticks()), dtype=int)
        cbar.set_ticks(tickValues+0.5)
        cbar.ax.set_yticklabels(list(map(str, tickValues)))
    else:
        im = ax.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1, interpolation='none')
        ax.set_title('Measurement Mask')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if knownRD:
        if sampleData.trainFlag: ax = plt.subplot2grid((2,4), (1,2))
        elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,1))
        im = ax.imshow(sample.RD, cmap='viridis', aspect='auto', vmin=0, interpolation='none')
        ax.set_title('RD')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    if knownERD:
        if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,0))
        elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,2))
        im = ax.imshow(sample.ERD, cmap='viridis', aspect='auto', vmin=0, interpolation='none')
        ax.set_title('ERD')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        if sampleData.impFlag or sampleData.postFlag: ax = plt.subplot2grid((1,5), (0,1))
        elif sampleData.simulationFlag: ax = plt.subplot2grid((2,4), (1,3))
        im = ax.imshow(np.nan_to_num(sample.processedERD, nan=0, posinf=0, neginf=0), cmap='viridis', aspect='auto', vmin=0, interpolation='none')
        ax.set_title('Processed ERD')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    #Save
    f.tight_layout()
    f.subplots_adjust(top = 0.85)
    saveLocation = dir_progression + 'progression' + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '_avg.png'
    plt.savefig(saveLocation)
    plt.close()

    #Borderless saves
    if knownERD:
        saveLocation = dir_progression + 'ERD_original_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.ERD, saveLocation, aspect='auto', cmap='viridis', vmin=0)
        
        saveLocation = dir_progression + 'ERD_processed_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(np.nan_to_num(sample.processedERD, nan=0, posinf=0, neginf=0), saveLocation, aspect='auto', cmap='viridis', vmin=0)
    
    if knownRD:
        saveLocation = dir_progression + 'RD_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.RD, saveLocation, aspect='auto', cmap='viridis', vmin=0)
        
    if knownGT:
        saveLocation = dir_progression + 'groundTruth_sumImage_' + chanLabel + '.png'
        borderlessPlot(sampleData.sumImage, saveLocation, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    
    saveLocation = dir_progression + 'reconstruction_sumImage' + '_iter_' + str(sample.iteration) +  '_perc_' + str(sample.percMeasured) + '.png'
    borderlessPlot(sample.sumReconImage, saveLocation, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    
    saveLocation = dir_progression + 'measured_sumImage_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
    borderlessPlot(sample.sumImage*sample.mask, saveLocation, cmap='hot', aspect='auto', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    
    saveLocation = dir_progression + 'mask_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
    borderlessPlot(sample.mask, saveLocation, cmap='gray', aspect='auto', vmin=0, vmax=1)
    
    if len(sample.progMap.shape)>0:
        saveLocation = dir_progression + 'progressionMap_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.progMap+0.5, saveLocation, cmap=cmapProgMap, aspect='auto', vmin=None, vmax=None, norm=normProgMap)
    
    if parallelization: _ = ray.get(samplingProgress_Actor.update.remote(1))

def runSampling(sampleData, cValue, model, percToScan, percToViz, lineVisitAll, dir_Results, tqdmHide, samplingProgress_Actor=None, percProgUpdate=None):
    
    #If in parallel, ignore warnings
    if parallelization and not debugMode: 
        warnings.filterwarnings("ignore")
        logging.root.setLevel(logging.ERROR)

    #Make sure random selection is consistent
    if consistentSeed: 
        np.random.seed(0)
        random.seed(0)
    
    #If groupwise is active, specify how many points should be scanned each step
    if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random') and percToScan != None: sampleData.pointsToScan = int(np.ceil(((sampleData.stopPerc/100)*sampleData.area)/(sampleData.stopPerc/percToScan)))
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
    result = Result(sampleData, dir_Results, cValue)
    
    #Scan initial sets
    for initialSet in sampleData.initialSets: sample.performMeasurements(sampleData, tempScanData, result, initialSet, model, cValue, False)
    
    #Check stopping criteria, just in case of a bad input
    if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random' or not lineVisitAll) and (sample.percMeasured >= sampleData.stopPerc): sys.exit('\nError - All points were scanned or the stopping criteria have been met after the initial acquisition for sample: ' + sample.name)
    elif sampleData.scanMethod == 'linewise' and len(sampleData.linesToScan)-np.sum(np.sum(sample.mask, axis=1)>0) == 0: sys.exit('\nError - All lines were scanned after the inital acquisition for sample: ' + sample.name)
    if not sampleData.datagenFlag and np.sum(sample.processedERD) == 0: sys.exit('\nError - Initial ERD indicates there are no places to scan for sample: ' + sampleData.name + ' This probably means something went wrong during the sample read process, please check that the file formats are compatible.')
    
    #Perform the first update for the result
    result.update(sample, completedRunFlag)
    
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
        if ((percToViz != None) and ((sample.percMeasured-result.percsMeasured[-1]) >= percToViz)) or (percToViz == None) or (sampleData.scanMethod == 'linewise') or completedRunFlag: result.update(sample, completedRunFlag)
        
        #If using a global progress bar and percProgUpdate has been reached, then update the global sampling progress actor
        if samplingProgress_Actor != None and tqdmHide and (sample.percMeasured-lastPercMeasured >= percProgUpdate): 
            _ = ray.get(samplingProgress_Actor.update.remote(sample.percMeasured-lastPercMeasured))
            lastPercMeasured = copy.deepcopy(sample.percMeasured)

        #Update the progress bar
        if not tqdmHide:
            pbar.n = np.clip(round(sample.percMeasured,2), 0, sampleData.stopPerc)
            pbar.refresh()
    
    #Delete progress bar reference if it had been made
    if not tqdmHide: del pbar
    
    return result

#Compute approximated Reduction in Distortion (RD) values
def computeRD(sample, sampleData, tempScanData, cValue, updateLocations):
    
    #If not updating RD, then set locations and compute the sigma values of all unmeasured locations in square dimensionality
    if len(updateLocations) == 0:
        sigmaValues = tempScanData.neighborDistances[:,0]/cValue
        squareUnMeasuredLocations = tempScanData.squareUnMeasuredIdxs
        
    #Otherwise if updating RD (currently assumes the RDPP has not changed, as using prior reconstruction results
    #If reconstructions and RDPP have changed, then extract and calculate sigma values for applicable, affected locations
    else:
        
        #Identify unique unmeasured locations where the RD would be altered based on data at update locations (i.e., which unmeasured locations will have a new nearest measured neighbor), perform no update if there are none (Can happen with DESI)
        indices = np.unique(np.concatenate([np.argwhere(np.sum(updateLocation >= tempScanData.winStartPos, axis=1)+np.sum(updateLocation <= tempScanData.winStopPos, axis=1)==4) for updateLocation in updateLocations]).flatten())
        if len(indices) == 0: return
        squareUnMeasuredLocations = tempScanData.squareUnMeasuredIdxs[indices]
        
        #Set RD Values at affected unmeasured locations to 0
        sample.squareRDs[:, squareUnMeasuredLocations[:,0], squareUnMeasuredLocations[:,1]] = 0
    
        #Average the results together to form a single RD
        sample.squareRD = np.mean(sample.squareRDs, axis=0)
        
        #Resize as needed according to sample type, ensuring RD values at measured locations are zero
        if sampleData.sampleType == 'DESI': 
            sample.RD = resize(sample.squareRD, tuple(sampleData.finalDim), order=0)*(1-sample.mask)
            sample.RDs = np.moveaxis(resize(np.moveaxis(sample.squareRDs, 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)*(1-sample.mask)
        else: 
            sample.RD = sample.squareRD
            sample.RDs = sample.squareRDs
        
        #Ensure RD values at square measured locations to zero
        sample.squareRD = sample.squareRD*(1-sample.squareMask)
        sample.squareRDs = sample.squareRDs*(1-sample.squareMask)
        
        #Exit the method
        return
        
        #Note: The only time that the RD may be updated, rather than fully computed, is in the case of an oracle run or c value optimization with percToScan enabled
        #Therein, for SLADS and DLADS, the reconstructions and RDPPs are not being updated by ground-truth data, as that would require updating neighbor information
        #Since this is trying to be avoided with percToScan, actually computing updated RD values would have no benefit over percToViz with the current reconstruction method
        #Provided that IDW might be replaced in the future, the framework has been left to update RD values, assuming new reconstructions and RDPPs are available.
        #For now, when using percToScan, with an oracle run or c value optimization, it is temorarily assumed that the RD values for locations dependent on newly measured data are zero
        sys.exit('\nError - RD values cannot be truly/correctly updated unless reconstructions and RDPPs have also been updated. As the current reconstruction method requires updating neighbor information, this cannot currently be performed and yield a performance improvement.')
        
        #Compute the affected locations' new nearest neighbor distances and sigma values
        neighborDistances, _ = NearestNeighbors(n_neighbors=1).fit(updateLocations).kneighbors(squareUnMeasuredLocations)
        sigmaValues = tempScanData.neighborDistances[:,0][indices]/cValue
    
    #Compute window sizes and positions for unmeasured locations, updating a persistant reference for updating RD
    if not staticWindow: windowSizes = np.ceil(2*dynWindowSigMult*sigmaValues).astype(int)+1
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
        gaussianSignals = [signal.gaussian(newWindows[index], newSigmaValues[index]) for index in range(0, len(newSigmaValues))]
        sampleData.gaussianWindows.update(zip(newSigmaValues, [np.outer(gaussianSignals[index], gaussianSignals[index]) for index in range(0, len(gaussianSignals))]))
    
    #Zero-pad the RDPPs by the maximum radius and offset window positions accordingly
    maxRadius = np.max(radii)
    paddedRDPPs = np.pad(sample.RDPPs, [(0, 0), (maxRadius, maxRadius), (maxRadius, maxRadius)], mode='constant')
    offsetWinStartPos, offsetWinStopPos = winStartPos+maxRadius, winStopPos+maxRadius
    
    #Compute RD Values
    sample.squareRDs[:, squareUnMeasuredLocations[:,0], squareUnMeasuredLocations[:,1]] = np.asarray([np.sum(sampleData.gaussianWindows[sigmaValues[index]]*paddedRDPPs[:, offsetWinStartPos[index][0]:offsetWinStopPos[index][0]+1, offsetWinStartPos[index][1]:offsetWinStopPos[index][1]+1], axis=(1,2)) for index in range(0, len(squareUnMeasuredLocations))]).T
    
    #Average the results together to form a single RD
    sample.squareRD = np.mean(sample.squareRDs, axis=0)
    
    #Resize as needed according to sample type, ensuring RD values at measured locations are zero
    if sampleData.sampleType == 'DESI': 
        sample.RD = resize(sample.squareRD, tuple(sampleData.finalDim), order=0)*(1-sample.mask)
        sample.RDs = np.moveaxis(resize(np.moveaxis(sample.squareRDs, 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)*(1-sample.mask)
    else: 
        sample.RD = sample.squareRD
        sample.RDs = sample.squareRDs
    
    #Ensure RD values at square measured locations to zero
    sample.squareRD = sample.squareRD*(1-sample.squareMask)
    sample.squareRDs = sample.squareRDs*(1-sample.squareMask)

#Extract features of the reconstruction to use as inputs to SLADS(-Net) models
def computePolyFeatures(sampleData, tempScanData, reconImage):
    
    #Retreive recon values
    inputValues = reconImage[tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]]
    
    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in tempScanData.squareMeasuredIdxs]))
    measuredValues = reconImage[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Find neighbor information
    neighborValues = measuredValues[tempScanData.neighborIndices]
    
    #Create array to hold features
    feature = np.zeros((np.shape(tempScanData.squareUnMeasuredIdxs)[0],6), dtype=np.float32)
    
    #Compute std div features
    diffVect = computeDifference(neighborValues, np.transpose(np.matlib.repmat(inputValues, np.shape(neighborValues)[1],1)))
    feature[:,0] = np.sum(tempScanData.neighborWeights*diffVect, axis=1)
    feature[:,1] = np.sqrt(np.sum(np.power(diffVect,2),axis=1))
    
    #Compute distance/density features
    cutoffDist = np.ceil(np.sqrt((featDistCutoff/100)*(sampleData.area/np.pi)))
    feature[:,2] = tempScanData.neighborDistances[:,0]
    neighborsInCircle = np.sum(tempScanData.neighborDistances<=cutoffDist,axis=1)
    feature[:,3] = (1+(np.pi*(np.square(cutoffDist))))/(1+neighborsInCircle)
    
    #Compute gradient features; assume continuous features
    gradientImageX, gradientImageY = np.gradient(reconImage)
    feature[:,4] = abs(gradientImageY)[tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]]
    feature[:,5] = abs(gradientImageX)[tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]]
    
    #Fit polynomial features to the determined array
    polyFeatures = PolynomialFeatures(degree=2).fit_transform(feature)
    
    return polyFeatures

#Prepare data for DLADS or GLANDS model input; if a channel number was given then prepare it on its own, otherwise create a batch for the whole sample
def prepareInput(sample, sampleData, numChannel=None):
    
    inputStack = []
    if numChannel != None:
        
        #Normalize/Standardize/Rescale input data as configured
        inputReconImage = sample.squareChanReconImages[numChannel]
        if dataAdjust == 'rescale' and (erdModel=='DLADS' or erdModel=='GLANDS'):
            minValue = np.min(inputReconImage)
            tempA = (inputReconImage-minValue)
            tempB = (np.max(inputReconImage)-minValue)
            inputReconImage = np.divide(tempA, tempB, out=np.zeros_like(tempA), where=tempB!=0)
        elif dataAdjust == 'standardize' and (erdModel=='DLADS' or erdModel=='GLANDS'):
            tempA = (inputReconImage-np.mean(inputReconImage))
            tempB = np.std(inputReconImage)
            inputReconImage = np.divide(tempA, tempB, out=np.zeros_like(tempA), where=tempB!=0)
        
        #Add channels to the input stack
        if 'opticalData' in inputChannels: inputStack.append(sampleData.squareOpticalImage)
        if 'mask' in inputChannels: inputStack.append(sample.squareMask)
        if 'reconData' in inputChannels: inputStack.append(inputReconImage*(1-sample.squareMask))
        if 'measureData' in inputChannels: inputStack.append(inputReconImage*sample.squareMask)
        return np.dstack(inputStack)
    
    else: 
        
        #Normalize/Standardize/Rescale input data as configured
        inputReconImages = np.moveaxis(sample.squareChanReconImages, 0, -1)
        if dataAdjust == 'rescale' and (erdModel=='DLADS' or erdModel=='GLANDS'):
            minValues = np.min(inputReconImages, axis=(0,1))
            tempA = (inputReconImages-minValues)
            tempB = (np.max(inputReconImages, axis=(0,1))-minValues)
            inputReconImages = np.divide(tempA, tempB, out=np.zeros_like(tempA), where=tempB!=0)
        elif dataAdjust == 'standardize' and (erdModel=='DLADS' or erdModel=='GLANDS'):
            tempA = (inputReconImages-np.mean(inputReconImages))
            tempB = np.std(inputReconImages)
            inputReconImages = np.divide(tempA, tempB, out=np.zeros_like(tempA), where=tempB!=0)
        inputReconImages = np.moveaxis(inputReconImages, -1, 0)
        
        #Add channels to the input stack
        if 'opticalData' in inputChannels: inputStack.append(np.repeat(np.expand_dims(sampleData.squareOpticalImage, 0), len(inputReconImages), axis=0))
        if 'mask' in inputChannels: inputStack.append(np.repeat(np.expand_dims(sample.squareMask, 0), len(inputReconImages), axis=0))
        if 'reconData' in inputChannels: inputStack.append(inputReconImages*(1-sample.squareMask))
        if 'measureData' in inputChannels: inputStack.append(inputReconImages*sample.squareMask)
        return np.stack(inputStack, axis=-1)
        
#Determine the Estimated Reduction in Distortion
def computeERD(sample, sampleData, tempScanData, model):

    #Compute the ERD with the prescribed model; if configured to, only use a single channel
    if not chanSingle:
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
            for chanNum in range(0, len(sample.squareERDs)): sample.squareERDs[chanNum, tempScanData.squareUnMeasuredIdxs[:, 0], tempScanData.squareUnMeasuredIdxs[:, 1]] = ray.get(model.generateERD.remote(sample.polyFeatures[chanNum]))
        elif erdModel == 'DLADS': 
            
            #First try inferencing all m/z channels at the same time 
            if not sampleData.OOM_multipleChannels:
                try: sample.squareERDs = ray.get(model.generateERD.remote(makeCompatible(prepareInput(sample, sampleData)))).copy()
                except: 
                    sampleData.OOM_multipleChannels = True
                    if (len(gpus) > 0): print('\nWarning - Could not inference ERD for all channels of sample '+sampleData.name+' simultaneously on system GPU; will try processing channels iteratively.')
                    if (len(gpus) == 0): print('\nWarning - Could not inference ERD for all channels of sample '+sampleData.name+' simultaneously on system; will try processing channels iteratively.')
            
            #If multiple channels causes an OOM, then try running each channel through on its own
            if sampleData.OOM_multipleChannels and not sampleData.OOM_singleChannel:
                try: sample.squareERDs = np.asarray([ray.get(model.generateERD.remote(makeCompatible(prepareInput(sample, sampleData, chanNum))))[0,:,:].copy() for chanNum in range(0, len(sample.squareERDs))])
                except: sampleData.OOM_singleChannel = True
            
            #If an OOM occured for both mutiple and single channel inferencing, then exit; need to either restart program with no GPUs, or there isn't enough system RAM
            if sampleData.OOM_multipleChannels and sampleData.OOM_singleChannel and (len(gpus) > 0): sys.exit('\nError - Sample '+sampleData.name+' dimensions are too high for the ERD to be inferenced on system GPU; please try disabling the GPU in the CONFIG.')
            if sampleData.OOM_multipleChannels and sampleData.OOM_singleChannel and (len(gpus) == 0): sys.exit('\nError - Sample '+sampleData.name+' dimensions are too high for the ERD to be inferenced on this system by the loaded model.')
            
    else:
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
            ERDValues = ray.get(model.generateERD.remote(sample.polyFeatures[0]))
            for chanNum in range(0, len(sample.squareERDs)): sample.squareERDs[chanNum, tempScanData.squareUnMeasuredIdxs[:, 0], tempScanData.squareUnMeasuredIdxs[:, 1]] = ERDValues
        elif erdModel == 'DLADS': 
            sample.squareERDs[0] = ray.get(model.generateERD.remote(makeCompatible(prepareInput(sample, sampleData, 0))))[0,:,:].copy()
            for chanNum in range(1, len(sample.squareERDs)): sample.squareERDs[chanNum] = sample.squareERDs[0]
    
    #Set any negative/nan/inf values to 0 and average across all channels
    sample.squareERDs[sample.squareERDs<0] = 0
    sample.squareERDs = np.nan_to_num(sample.squareERDs, nan=0, posinf=0, neginf=0)
    sample.squareERD = np.mean(sample.squareERDs, axis=0)
    
    #Resize as needed according to sample type
    if sampleData.sampleType == 'DESI':
        sample.ERD = resize(sample.squareERD, tuple(sampleData.finalDim), order=0)
        sample.ERDs = np.moveaxis(resize(np.moveaxis(sample.squareERDs , 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)
    else:
        sample.ERD = sample.squareERD
        sample.ERDs = sample.squareERDs

#Determine which unmeasured points of a sample should be scanned given the current E/RD
def findNewMeasurementIdxs(sample, sampleData, tempScanData, result, model, cValue, percToScan):

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
                
                #Find next measurement location and store the chosen scanning location for later, actual measurement
                newIdx = sample.unMeasuredIdxs[np.argmax(sample.processedERD[sample.unMeasuredIdxs[:,0], sample.unMeasuredIdxs[:,1]])]
                newIdxs.append(newIdx.tolist())
                
                #Perform the measurement, using values from reconstruction 
                sample.performMeasurements(sampleData, tempScanData, result, newIdx, model, cValue, True)
                
                #When enough new locations have been determined, break from loop
                if (np.sum(sample.mask)-np.sum(result.lastMask)) >= sampleData.pointsToScan: break
                
            #Convert to array for indexing
            newIdxs = np.asarray(newIdxs)
        else:
            #Identify the unmeasured location with the highest processedERD value; return in a list to ensure it is iterable
            newIdxs = np.asarray([sample.unMeasuredIdxs[np.argmax(sample.processedERD[sample.unMeasuredIdxs[:,0], sample.unMeasuredIdxs[:,1]])].tolist()])
            
    elif sampleData.scanMethod == 'linewise':
        
        #If all locations on a chosen line should be scanned, select the line with maximum sum physical ERD
        if lineMethod =='fullLine':
            lineToScanIdx = np.nanargmax(np.nansum(sample.processedERD, axis=1))
            indexes = np.sort(np.argsort(sample.processedERD[lineToScanIdx])[::-1])
            newIdxs = np.column_stack([np.ones(len(indexes), dtype=np.float32)*lineToScanIdx, indexes]).astype(int)
        
        #If points on a chosen line should be chosen one-by-one, temporarily using reconstruction values for updating the ERD, selecting the line with maximum sum physical ERD
        elif lineMethod == 'percLine' and linePointSelection == 'single': 
            newIdxs = []
            lineToScanIdx = np.nanargmax(np.nansum(sample.processedERD, axis=1))
            while True:
                
                #If there are no remaining points to scan on this line with physical ERD > 0 or enough new locations have been found, break from loop
                if (np.sum(sample.processedERD[lineToScanIdx]) <= 0) or (len(newIdxs) >= sampleData.pointsToScan[lineToScanIdx]): break
                
                #Identify the next scanning location and store it for later, actual measurement
                newIdxs.append([lineToScanIdx, np.argmax(sample.processedERD[lineToScanIdx])])
                
                #Perform the measurement using values from reconstruction 
                sample.performMeasurements(sampleData, tempScanData, result, np.asarray(newIdxs[-1]), model, cValue, True)
                
            #Convert to array for indexing, sorting columns according to physical scanning order
            newIdxs = np.asarray(newIdxs)
            newIdxs[:,1] = np.sort(newIdxs[:,1])
        
        #If locations on the line should be selected in one step/group, select that group on the line with maximum sum physical ERD
        elif lineMethod == 'percLine' and linePointSelection == 'group':
            lineToScanIdx = np.nanargmax(np.nansum(sample.processedERD, axis=1))
            indexes = np.sort(np.argsort(sample.processedERD[lineToScanIdx])[::-1][:sampleData.pointsToScan[lineToScanIdx]])
            newIdxs = np.column_stack([np.ones(len(indexes), dtype=np.float32)*lineToScanIdx, indexes]).astype(int)
        
        #If a segment of a line should be scanned using a minimum percentage, choose the line with maximum sum physical ERD
        elif lineMethod == 'segLine' and segLineMethod == 'minPerc': 
           lineToScanIdx = np.nanargmax(np.nansum(sample.processedERD, axis=1))
           indexes = np.sort(np.argsort(sample.processedERD[lineToScanIdx])[::-1][:sampleData.pointsToScan[lineToScanIdx]])
           if len(indexes)>0: newIdxs = np.column_stack([np.ones(indexes[-1]-indexes[0]+1, dtype=np.float32)*lineToScanIdx, np.arange(indexes[0],indexes[-1]+1)]).astype(int)

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

#Perform the reconstruction using IDW (inverse distance weighting); retrieve measured values, compute reconstruction values, and combine into a new image; if 3D do all channels at once
def computeReconIDW(inputImage, tempScanData):
    reconImage = copy.deepcopy(inputImage)
    if len(tempScanData.squareUnMeasuredIdxs) > 0:
        if len(inputImage.shape) == 3: reconImage[:, tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]] = np.sum(inputImage[:, tempScanData.squareMeasuredIdxs[:,0], tempScanData.squareMeasuredIdxs[:,1]][:, tempScanData.neighborIndices]*tempScanData.neighborWeights, axis=-1)
        else: reconImage[tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]] = np.sum(inputImage[tempScanData.squareMeasuredIdxs[:,0], tempScanData.squareMeasuredIdxs[:,1]][tempScanData.neighborIndices]*tempScanData.neighborWeights, axis=1)
    return reconImage

#Rescale spatial dimensions of tensor x to match to those of tensor y
def customResize(x, y):
    x = image_ops.resize_images_v2(x, array_ops.shape(y)[1:3], method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
    nshape = tuple(y.shape.as_list())
    x.set_shape((None, nshape[1], nshape[2], None))
    return x
    
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

#Convert image into TF model compatible shapes/tensors
def makeCompatible(image):
    
    #Turn into an array before processings; will produce an error in the event of dimensional incompatability
    image = np.asarray(image)

    #Reshape for tensor transition, as needed by number of channels
    if len(image.shape) > 3: return image
    elif len(image.shape) > 2: return image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    else: return image.reshape((1,image.shape[0],image.shape[1],1))

#Define process for converting input images into input samples
def processImages(baseFolder, filenames, label):

    #If there are files to be processed, then convert each
    if len(filenames) > 0:
        for filename in tqdm(filenames, desc='Converting '+ label+ ' Images', leave=True, ascii=asciiFlag):
            
            #Extract name and verify the image has a supported extension and number of channels
            basename, extension = os.path.splitext(filename)
            basename, numChannels = os.path.basename(basename).split('-numChan-')
            numChannels = int(numChannels)
            if extension not in ['.png', '.jpg', '.tiff']: 
                print('Error - Skipping file, as image extenstion is not currently compatible: ' + filename)
                break
            image = cv2.imread(filename)
            if len(image.shape) > 3:
                print('Error - Skipping file, as image contains more than 3 channels, which is not currently supported: ' + filename)
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
    
#Truncate a value to a given precision 
def truncate(value, decimalPlaces=0): return np.trunc(value*10**decimalPlaces)/(10**decimalPlaces)

def borderlessPlot(image, saveLocation, cmap='viridis', aspect='auto', vmin=None, vmax=None, norm=None):
    f=plt.figure()
    ax=f.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(image, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax, interpolation='none')
    extent = ax.get_window_extent().transformed(f.dpi_scale_trans.inverted())
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


