#==================================================================
#GLOBAL METHOD AND CLASS DEFINITIONS
#==================================================================

#Object for initializing and storing sample metadata
class SampleData:
    def __init__(self, sampleFolder, initialPercToScan, stopPerc, scanMethod, lineRevist, postFlag, simulationFlag, trainFlag):
        
        #Save options as internal variables
        self.scanMethod = scanMethod
        self.initialPercToScan = initialPercToScan
        if lineVisitAll and self.scanMethod == 'linewise': sampleData.stopPerc = 100
        else: self.stopPerc = stopPerc
        self.lineRevist = lineRevist
        self.postFlag = postFlag
        self.simulationFlag = simulationFlag
        self.trainFlag = trainFlag
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
        
        #Set global variables to indicate that OOM error states have not yet occurred; limited handle for ERD inferencing limitations
        self.OOM_multipleChannels, self.OOM_singleChannel = False, False
        
        #Store location of MSI data and sample name
        self.sampleFolder = sampleFolder
        self.name = os.path.basename(sampleFolder)
        if impModel: self.name = impSampleName
        
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
            self.ppm = float(sampleInfo[lineIndex].rstrip())*1e-6
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
            sys.exit('Error - Unknown sample type: ' + self.sampleType + ' specified in sampleInfo.txt for : ' + sampleFolder)

        #MSI specific
        if self.sampleType == 'MALDI' or self.sampleType == 'DESI':
        
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
            scanFileNames = natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'+self.lineExt), reverse=False)
            if self.sampleType == 'DESI' and self.ignoreMissingLines:
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
            except: sys.exit('Error - Unable to load measurement mask for sample: ' + self.name)
        
        #Establish sample area to measure; do not apply percFOVMask in training/validation database
        if self.useMaskFOV and percFOVMask and not self.trainFlag: self.area = np.sum(self.maskFOV)
        else: self.area = int(round(self.finalDim[0]*self.finalDim[1]))
        
        #If not just post-processing, setup initial sets
        if not self.postFlag: self.generateInitialSets(self.scanMethod)
        
        #Determine non-overlapping bins for MSI data based on ppm
        if self.sampleType=='DESI' or self.sampleType=='MALDI':
            
            #Determine minimum difference between m/z at the lower m/z boundary and extract releavant precision information
            mzMinDiff = (self.mzLowerBound*self.ppmPos)-(self.mzLowerBound*self.ppmNeg)
            self.mzPrecision = truncate(mzMinDiff, -int(np.floor(np.log10(mzMinDiff))))
            self.mzRound = -math.floor(np.log10(self.mzPrecision))
            
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
            
            #For each of the m/z values in the maximum precision distribution, find which final m/z bin it will map to
            self.mzDistIndices = []
            mzFinalIndex = 0
            for mzIndex in range(0, len(self.mzInitialDist)):
                while self.mzUpperValues[mzFinalIndex] < self.mzInitialDist[mzIndex]: mzFinalIndex+=1
                self.mzDistIndices.append(mzFinalIndex)
                
            #Find unique index mapping between maximum precision distribution and the non-overlapping ppm bins
            self.mzIndices, self.mzOriginalIndices = np.unique(self.mzDistIndices, return_index=True)
            
            #Load targeted m/z values and set corresponding ranges 
            try: self.chanValues = np.loadtxt(self.sampleFolder+os.path.sep+'channels.csv', delimiter=',')
            except: self.chanValues = np.loadtxt('channels.csv', delimiter=',')
            self.numChannels = len(self.chanValues)
            self.mzRanges = np.round(np.column_stack((self.chanValues*self.ppmNeg, self.chanValues*self.ppmPos)), self.mzRound)
            
            #Now that all of the final dimensions have been determined, setup .hdf5 file locations, and either a shared memory actor or array, for storing results 
            self.allImagesPath = self.sampleFolder+os.path.sep+'allImages.hdf5'
            if os.path.exists(self.allImagesPath): os.remove(self.allImagesPath)
            if self.sampleType=='MALDI': 
                self.squareAllImagesPath = None
            elif self.sampleType=='DESI':
                self.squareAllImagesPath = self.sampleFolder+os.path.sep+'squareAllImages.hdf5'
                if os.path.exists(self.squareAllImagesPath): os.remove(self.squareAllImagesPath)
            
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
                self.reader_MSI_Actor = Reader_MSI_Actor.remote(self.sampleType, len(self.mzFinal), len(self.chanValues), self.finalDim[0], self.finalDim[1], self.allImagesPath, self.squareAllImagesPath)
            else: 
                self.allImages = np.zeros((len(self.mzFinal), self.finalDim[0], self.finalDim[1]), dtype=np.float32)
        
        #Setup targeted channel and sum images for holding data
        self.chanImages = np.zeros((self.numChannels, self.finalDim[0], self.finalDim[1]), dtype=np.float32)
        self.sumImage = np.zeros((self.finalDim), dtype=np.float32)

        #If a simulation or post-processing, read all of the sample data and save in hdf5, optimized for loading whole channel images
        if self.simulationFlag or self.postFlag: 
            self.readScanData()
            if self.sampleType == 'MALDI' or self.sampleType == 'DESI': 
                if parallelization: 
                    self.allImagesMax = ray.get(self.reader_MSI_Actor.writeToDisk.remote(self.squareDim))
                    #if not self.trainFlag: self.allImages_id = ray.get(reader_MSI_Actor.shareAllImages.remote())
                    #del self.allImages (if retrieved from shared memory!)
                    del self.reader_MSI_Actor, self.mzOriginalIndices_id, self.mzRanges_id
                    if self.sampleType == 'DESI': del self.mzFinalGrid_id, self.chanFinalGrid_id
                    gc.collect() #Force garbage collection after deletion instead of reseting ray (this doesn't appear to be effective for pool calls though)
                else:
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
                    newIdxs = newIdxs[:int(np.ceil((self.stopPerc/100)*len(newIdxs)))]
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
        
        #MALDI experimental operation is not yet implemented, where the program should just read new positions referencing the chosen new idxs
        
        #DEPRECATED METHOD: At the cost of accuracy in the channel images can find static indexes in aligned bin data and pull data from total images
        #During version development included option to determine indexes in sampleData initialization and use flag 'fastChannelLoading' to allow for mz data approximation.
        #Flag disabled setting the chanValues here and in the corresponding parallel method
        #This method reference will be deleted in the next version, but kept here in case of future requirements
        #Relative accuracy difference was not benchmarked.
        #if fastChannelLoading: 
        #    self.chanIndexes = [bisect_left(self.mzLowerValues, chanValue)-1 for chanValue in self.chanValues]
        #    self.chanValues = self.mzFinal[self.chanIndexes]
        
        #Obtain and sort the available line files pertaining to the current scan
        scanFileNames = natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'+self.lineExt), reverse=False)
        
        #If IMAGE, each file corresponds to a channel, read in each and sum of all data, augmenting list of channel labels
        if self.sampleType == 'IMAGE':
            for chanNum in range(0, len(scanFileNames)):
                self.chanValues.append(scanFileNames[chanNum].split('chan-')[1].split('.')[0])
                try: self.chanImages[chanNum] = cv2.imread(scanFileNames[chanNum], 0)
                except: sys.exit('Error - Unable to read file' + scanFileNames[chanNum])
            self.sumImage = np.sum(np.atleast_3d(self.chanImages), axis=0)

        #If MALDI, then there is only a single file with all of the spectral data
        elif self.sampleType == 'MALDI':
        
            #Establish file pointer for the single imzML file and verify it is readable
            try: data = ImzMLParser(scanFileNames[0])
            except: sys.exit('Error - Unable to read file' + scanFileNames[0])
            
            #Adjust stored coordinates to be zero-based
            coordinates = np.asarray(data.coordinates)-1
            
            #If parallelization is disabled then read in data sequentially, otherwise pass writable coordinates to parallel actor
            if not parallelization:
                for i, (x, y, z) in tqdm(enumerate(coordinates), total = len(coordinates), desc='Reading', leave=False, ascii=asciiFlag):
                    mzs, ints = data.getspectrum(i)
                    self.sumImage[y, x] = np.sum(ints)
                    filtIndexLow, filtIndexHigh = bisect_left(mzs, self.mzLowerBound), bisect_right(mzs, self.mzUpperBound)
                    self.allImages[:, y, x] = np.add.reduceat(mzFastIndex(mzs[filtIndexLow:filtIndexHigh], ints[filtIndexLow:filtIndexHigh], self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount), self.mzOriginalIndices).astype(np.float32)
                    self.chanImages[:, y, x] = [np.sum(ints[bisect_left(mzs, mzRange[0]):bisect_right(mzs, mzRange[1])]) for mzRange in self.mzRanges]
            else:
                _ = ray.get(self.reader_MSI_Actor.setCoordinates.remote(coordinates))
                _ = ray.get([msi_parhelper.remote(self.reader_MSI_Actor, scanFileNames, indexes, self.mzOriginalIndices_id, self.mzRanges_id, self.sampleType, self.mzLowerBound, self.mzUpperBound, self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount, self.mask, self.newTimes, self.finalDim, self.sampleWidth, self.scanRate) for indexes in np.array_split(np.arange(0, len(coordinates)), numberCPUS)])
            
            #Close the MSI file
            del data
        
        #If DESI, each file corresponds to a full line of data
        elif self.sampleType == 'DESI':
        
            #If line revisiting is disabled, identify which files have not yet been scanned
            if self.lineRevist == False: scanFileNames = natsort.natsorted(list(set(scanFileNames)-set(self.readScanFiles)), reverse=False)
            
            #If parallelization is disabled then read in data sequentially
            if not parallelization:
            
                #Extract line number from the filenames, removing leading zeros, subtract 1 for zero indexing, and obtain correct physical row indexes from LUT if applicable
                for scanFileName in scanFileNames:
                    
                    #Load the line data and flag errors during the process (primarily checking for files without data)
                    errorFlag = False
                    try: data = mzFile(scanFileName)
                    except: errorFlag = True
                    
                    #Extract the file number and if unordered find corresponding line number in LUT, otherwise line number is the file number
                    if not errorFlag:
                        fileNum = int(scanFileName.split('line-')[1].split('.')[0].lstrip('0'))-1
                        if self.unorderedNames: 
                            try: lineNum = self.physicalLineNums[fileNum+1]
                            except: errorFlag = True #print('\nWarning - Attempt to find the physical line number for the file: ' + scanFileName + ' has failed; the file will therefore be ignored this iteration.')
                        else: lineNum = fileNum
                    
                    #If error still has not occurred 
                    if not errorFlag:
                    
                        #If ignoring missing lines, then determine the offset for correct indexing
                        if self.ignoreMissingLines and len(self.missingLines) > 0: lineNum -= int(np.sum(lineNum > self.missingLines))
                    
                        #Add file name to those that will have been already scanned (when this process finishes)
                        self.readScanFiles.append(scanFileName)
                            
                        #Extract original measurement times and setup/read TIC data as applicable
                        if data.format == 'Bruker': 
                            sumImageLine = []
                            origTimes = np.asarray(data.ms1_frames)[:,1]
                        else: 
                            imageData = np.asarray(data.xic(data.time_range()[0], data.time_range()[1]))
                            origTimes, sumImageLine = imageData[:,0], imageData[:,1]
                        
                        #Force original times memory allocation to be contigous
                        origTimes = np.ascontiguousarray(origTimes)
                        
                        #Offset the original measurement times, such that the first position's time equals 0
                        origTimes -= np.min(origTimes)
                        
                        #If the data is being sparesly acquired in lines, then the listed times in the file need to be shifted
                        if (impModel or postModel) and impOffset and scanMethod == 'linewise' and (lineMethod == 'segLine' or lineMethod == 'fullLine'): origTimes += (np.argwhere(self.mask[lineNum]==1).min()/self.finalDim[1])*(((self.sampleWidth*1e3)/self.scanRate)/60)
                        elif (impModel or postModel) and impOffset: sys.exit('Error - Using implementation or post-process modes with an offset but not segmented-linewise operation is not currently a supported configuration.')
                        
                        #Seup storage locations for each measured location
                        chanDataLine, mzDataLine = [[] for _ in range(0, len(self.mzRanges))], []
                        
                        #Set positions to be scanned for the line
                        if data.format == 'Bruker': positions = range(1, len(origTimes)+1)
                        else: positions = range(data.scan_range()[0], data.scan_range()[1]+1)
                        
                        #Read in and process spectrum data for each location
                        for pos in positions:
                            if data.format == 'Bruker':
                                frameData = np.asarray(data.frame(pos))
                                mzs, ints = frameData[:,0], frameData[:,2]
                                sumImageLine.append(np.sum(ints))
                            else: 
                                scanData = np.array(data.scan(pos, 'profile'))
                                mzs, ints = scanData[:,0], scanData[:,1]
                            filtIndexLow, filtIndexHigh = bisect_left(mzs, self.mzLowerBound), bisect_right(mzs, self.mzUpperBound)
                            mzDataLine.append(np.add.reduceat(mzFastIndex(mzs[filtIndexLow:filtIndexHigh], ints[filtIndexLow:filtIndexHigh], self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount), self.mzOriginalIndices))
                            for mzRangeNum in range(0, len(self.mzRanges)):
                                mzRange = self.mzRanges[mzRangeNum]
                                chanDataLine[mzRangeNum].append(np.sum(ints[bisect_left(mzs, mzRange[0]):bisect_right(mzs, mzRange[1])]))
                                
                        #Create regular grid interpolators, aligning all/targeted m/z data, and storing results
                        self.allImages[:, lineNum, :] = scipy.interpolate.RegularGridInterpolator((origTimes, self.mzFinal), np.asarray(mzDataLine, dtype='float64'), bounds_error=False, fill_value=0)(self.mzFinalGrid).astype('float32')
                        self.chanImages[:, lineNum, :] = scipy.interpolate.RegularGridInterpolator((origTimes, self.chanValues), np.asarray(chanDataLine, dtype='float64').T, bounds_error=False, fill_value=0)(self.chanFinalGrid).astype('float32')
                        self.sumImage[lineNum, :] = np.interp(self.newTimes, origTimes, np.nan_to_num(sumImageLine, nan=0, posinf=0, neginf=0), left=0, right=0)
                    
                    #Close the mzFile
                    data.close()
            
            #Otherwise read data in parallel and perform remaining interpolations of any remaining m/z data to regular grid in serial (parallel operation is too memory intensive)
            else:
                _ = ray.get([msi_parhelper.remote(self.reader_MSI_Actor, scanFileNames, indexes, self.mzOriginalIndices_id, self.mzRanges_id, self.sampleType, self.mzLowerBound, self.mzUpperBound, self.mzLowerIndex, self.mzPrecision, self.mzRound, self.mzInitialCount, self.mask, self.newTimes, self.finalDim, self.sampleWidth, self.scanRate, self.mzFinal, self.mzFinalGrid_id, self.chanValues, self.chanFinalGrid_id, impModel, postModel, impOffset, scanMethod, lineMethod, self.physicalLineNums, self.ignoreMissingLines, self.missingLines, self.unorderedNames) for indexes in np.array_split(np.arange(0, len(scanFileNames)), numberCPUS)])
                _ = ray.get(self.reader_MSI_Actor.interpolateDESI.remote(self.mzFinal, self.mzFinalGrid))
                for scanFileName in ray.get(self.reader_MSI_Actor.getReadScanFiles.remote()): self.readScanFiles.append(scanFileName)

        #If parallelization is enabled, read MSI data in parallel, retrieve from shared memory, and process data into accessible shape
        if parallelization:
            self.chanImages = np.moveaxis(ray.get(self.reader_MSI_Actor.getChanImages.remote()), -1, 0)
            #self.allImages = np.moveaxis(ray.get(self.reader_MSI_Actor.getAllImages.remote()), -1, 0)
            self.sumImage = ray.get(self.reader_MSI_Actor.getSumImage.remote())
            
        #If DESI MSI, then need to resize the images to obtain square dimensionality, otherwise the square dimensions are equal to the original
        if self.sampleType == 'DESI': self.squareChanImages = np.moveaxis(resize(np.moveaxis(self.chanImages, 0, -1), tuple(self.squareDim), order=0), -1, 0)
        else: self.squareChanImages = self.chanImages
            
        #Find the maximum value in each channel image for easy referencing
        self.chanImagesMax = np.max(self.chanImages, axis=(1,2))
        
        #DEBUG: Verify file reading functionality through training sample visualization
        #plt.imshow(self.sumImage, cmap='hot', aspect='auto')
        #plt.savefig('TEMP/sumImage.png')
        #plt.close()
        #for chanIndex in range(0, len(self.chanImages)):
        #    plt.imshow(self.chanImages[chanIndex], cmap='hot', aspect='auto')
        #    plt.savefig('TEMP/chanImage_'+str(self.chanValues[chanIndex])+'.png')
        #    plt.close()
        #if parallelization: self.allImages = np.moveaxis(ray.get(self.reader_MSI_Actor.getAllImages.remote()), -1, 0)
        #for mzIndex in range(0, len(self.allImages)):
        #    plt.imshow(self.allImages[mzIndex], cmap='hot', aspect='auto')
        #    plt.savefig('TEMP/mzImage_'+str(self.mzFinal[mzIndex])+'.png')
        #    plt.close()

#Relevant sample data at each time step; static information should be held in corresponding SampleData object
class Sample:
    def __init__(self, sampleData):
        
        #Initialize measurement masks and other variables that are expected to exist
        self.mask = np.zeros((sampleData.finalDim), dtype=np.float32)
        if sampleData.sampleType == 'DESI': self.squareMask = resize(self.mask, tuple(sampleData.squareDim), order=0)
        else: self.squareMask = self.mask
        self.squareRD = np.zeros((sampleData.squareDim), dtype=np.float32)
        self.squareRDs = np.zeros((sampleData.numChannels, sampleData.squareDim[0], sampleData.squareDim[1]), dtype=np.float32)
        self.squareERD = np.zeros((sampleData.squareDim), dtype=np.float32)
        self.squareERDs = np.zeros((sampleData.numChannels, sampleData.squareDim[0], sampleData.squareDim[1]), dtype=np.float32)
        self.percMeasured = 0
        self.iteration = 0
        
        #If post-processing, link to the sampled mask
        if sampleData.postFlag: self.mask = sampleData.mask
        
    def performMeasurements(self, sampleData, result, newIdxs, model, cValue, bestCFlag, oracleFlag, datagenFlag, fromRecon):

        #Ensure newIdxs are indexible in 2 dimensions and update mask; post-processing will send empty set
        if not sampleData.postFlag:
            newIdxs = np.atleast_2d(newIdxs)
            self.mask[newIdxs[:,0], newIdxs[:,1]] = 1
        
        #Update which physical positions have not yet been measured for new measurement location(s) selection
        if sampleData.useMaskFOV: self.unMeasuredIdxs = np.transpose(np.where((self.mask==0) & (sampleData.maskFOV==1)))
        else: self.unMeasuredIdxs = np.transpose(np.where(self.mask==0))
        
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
        
        #For DESI, resize the mask to enforce square pixels
        if sampleData.sampleType == 'DESI': self.squareMask = resize(self.mask, tuple(sampleData.squareDim), order=0)
        else: self.squareMask = self.mask
        
        #Extract measured and unmeasured locations, considering FOV mask if applicable
        squareMeasuredIdxs = np.transpose(np.where(self.squareMask==1))
        if sampleData.useMaskFOV: squareUnMeasuredIdxs = np.transpose(np.where((self.squareMask==0) & (sampleData.squareMaskFOV==1)))
        else: squareUnMeasuredIdxs = np.transpose(np.where(self.squareMask==0))
        
        #Determine neighbor information for unmeasured locations
        if len(squareUnMeasuredIdxs) > 0: neighborIndices, neighborWeights, neighborDistances = findNeighbors(squareMeasuredIdxs, squareUnMeasuredIdxs)
        else: neighborIndices, neighborWeights, neighborDistances = [], [], []
        
        #Compute the reconstructions (using square pixels) if new data is acquired
        if not fromRecon:
        
            #Update the iteration counter
            self.iteration += 1
            
            #Compute reconstructions and average for visualization, if DESI then resize to physical dimensions 
            if sampleData.sampleType == 'DESI':
                self.squareSumImageReconImage = computeReconIDW(resize(self.sumImage, tuple(sampleData.squareDim), order=0), squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
                self.squareChanReconImages = computeReconIDW(np.moveaxis(resize(np.moveaxis(self.chanImages, 0, -1), tuple(sampleData.squareDim), order=0), -1, 0), squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
                self.sumImageReconImage = resize(self.squareSumImageReconImage, tuple(sampleData.finalDim), order=0)
                self.chanReconImages = np.moveaxis(resize(np.moveaxis(self.squareChanReconImages , 0, -1), tuple(sampleData.finalDim), order=0), -1, 0)
            else:
                self.squareSumImageReconImage = computeReconIDW(self.sumImage, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
                self.squareChanReconImages = computeReconIDW(self.chanImages, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights)
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
                self.RD = np.zeros(sampleData.finalDim, dtype=np.float32)
                self.squareRD = np.zeros(sampleData.squareDim, dtype=np.float32)
                self.squareRDs = np.zeros((sampleData.numChannels, sampleData.squareDim[0], sampleData.squareDim[1]), dtype=np.float32)
                self.squareRDValues = self.squareRDs[:, squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]]
                self.squareERD = self.squareRD
            else: self.squareERD = np.zeros(sampleData.squareDim, dtype=np.float32)
        elif oracleFlag or bestCFlag:
        
            #If this is a full measurement step, compute the RDPP
            if not fromRecon: 
                #If dataAdjust is enabled, and using DLADS or GLANDS, then can optionally rescale RDPP computation inputs to between 0 and 1 or standardize them
                if dataAdjust == 'rescale' and (erdModel=='DLADS' or erdModel=='GLANDS'):
                    minValues, maxValues = np.min(sampleData.squareChanImages, axis=(1,2)), np.max(sampleData.squareChanImages, axis=(1,2))
                    minMaxDiffs = (maxValues-minValues)
                    self.RDPPs = np.moveaxis(abs(((np.moveaxis(sampleData.squareChanImages, 0, -1)-minValues)/minMaxDiffs)-((np.moveaxis(self.squareChanReconImages, 0, -1)-minValues)/minMaxDiffs)), -1, 0)
                elif dataAdjust == 'standardize' and (erdModel=='DLADS' or erdModel=='GLANDS'):
                    meanValues, stdValues =  np.mean(sampleData.squareChanImages, axis=(1,2)), np.std(sampleData.squareChanImages, axis=(1,2))
                    self.RDPPs = np.moveaxis(abs(((np.moveaxis(sampleData.squareChanImages, 0, -1)-meanValues)/stdValues)-((np.moveaxis(self.squareChanReconImages, 0, -1)-meanValues)/stdValues)), -1, 0)
                else: self.RDPPs = abs(sampleData.squareChanImages-self.squareChanReconImages)
            
            #If this is a full measurement step, compute the RDPP
            #if not fromRecon: self.RDPPs = abs(sampleData.squareChanImages-self.squareChanReconImages)
                
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
        
        #For processed ERD, mask by FOV, set measured locations to 0, ensure >= values, rescale for potential Otsu, and prevent line revisitation as specified
        if sampleData.useMaskFOV: self.physicalERD = copy.deepcopy(self.ERD)*sampleData.maskFOV
        else: self.physicalERD = copy.deepcopy(self.ERD)
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
        
        #Determine if all channel evaluation is going to be performed at completion
        if self.sampleData.simulationFlag and not self.liveOutputFlag and not self.datagenFlag and ((self.bestCFlag and cAllChanOpt) or not self.bestCFlag): self.allChanEval = True
        else: self.allChanEval = False
        
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
        if (self.sampleData.sampleType == 'MALDI' or self.sampleData.sampleType == 'DESI') and self.sampleData.simulationFlag and self.liveOutputFlag:
            
            #If operating in parallel, create actors for reconstruction and load portions of the data into each, otherwise load data into main memory
            if parallelization:
                self.recon_Actors = [Recon_Actor.remote(indexes, self.sampleData.sampleType, self.sampleData.squareDim, self.sampleData.finalDim, self.sampleData.allImagesMax) for indexes in np.array_split(np.arange(0, len(self.sampleData.mzFinal)), numberCPUS)]
                #If performing a non-training simulation, then potentially could pass a ray object id rather than reading in from hdf5...
                #_ = [ray.get(reconIDW_Actor.setupFromShared.remote(self.allImages_id)) for reconIDW_Actor in reconIDW_Actors]
            else:
                self.sampleData.allImagesFile = h5py.File(self.sampleData.allImagesPath, 'a')
                self.sampleData.allImages = self.sampleData.allImagesFile['allImages'][:]
        
    def update(self, sample):
    
        #Update measurement mask and percentage of FOV measured at this step 
        self.lastMask = copy.deepcopy(sample.mask)
        self.percsMeasured.append(copy.deepcopy(sample.percMeasured))
        
        #If outputs should be produced at every update step, then do so, determining related metrics as needed
        if self.liveOutputFlag: 
            if self.sampleData.simulationFlag: self.extractSimulationData(sample, self.sampleData)
            visualize_serial(sample, self.sampleData, self.dir_progression, self.dir_chanProgressions, self.datagenFlag)
        
        #Save a copy of the measurement step for later evaluation
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
        
        #Extract measured and unmeasured locations for the measured mask
        squareMeasuredIdxs = np.transpose(np.where(sample.squareMask==1))
        if self.sampleData.useMaskFOV: squareUnMeasuredIdxs = np.transpose(np.where((sample.squareMask==0) & (self.sampleData.squareMaskFOV==1)))
        else: squareUnMeasuredIdxs = np.transpose(np.where(sample.squareMask==0))
                
        #Determine neighbor information for unmeasured locations
        if len(squareUnMeasuredIdxs) > 0: neighborIndices, neighborWeights, neighborDistances = findNeighbors(squareMeasuredIdxs, squareUnMeasuredIdxs)
        else: neighborIndices, neighborWeights, neighborDistances = [], [], []
        
        #Find PSNR/SSIM scores for all channel reconstructions
        sample.chanImagesPSNRList = [compare_psnr(self.sampleData.chanImages[index], sample.chanReconImages[index], data_range=self.sampleData.chanImagesMax[index]) for index in range(0, len(self.sampleData.chanImages))]
        sample.sumImagePSNR = compare_psnr(self.sampleData.sumImage, sample.sumImageReconImage, data_range=np.max(self.sampleData.sumImage))
        sample.chanImagesSSIMList = [compare_ssim(self.sampleData.chanImages[index], sample.chanReconImages[index], data_range=self.sampleData.chanImagesMax[index]) for index in range(0, len(self.sampleData.chanImages))]
        sample.sumImageSSIM = compare_ssim(self.sampleData.sumImage, sample.sumImageReconImage, data_range=np.max(self.sampleData.sumImage))
        
        #MSI Specific; if enabled then perform and evaluate reconstructions over the whole spectrum for the data known at each considered measurement step
        if self.allChanEval and (self.sampleData.sampleType == 'MALDI' or self.sampleData.sampleType == 'DESI'):
            
            #If operating in parallel, utilize actors created in the complete() method, or at initialization for live simulation; for serial, could vectorize, but extremely RAM intensive and usable batch size is unpredictable per system
            if parallelization:
                unMeasuredIdxs_id, measuredIdxs_id, neighborIndices_id, neighborWeights_id, squareMask_id = ray.put(squareUnMeasuredIdxs), ray.put(squareMeasuredIdxs), ray.put(neighborIndices), ray.put(neighborWeights), ray.put(sample.squareMask)
                _ = ray.get([recon_Actor.applyMask.remote(squareMask_id) for recon_Actor in self.recon_Actors])
                _ = ray.get([recon_Actor.computeRecon.remote(unMeasuredIdxs_id, measuredIdxs_id, neighborIndices_id, neighborWeights_id) for recon_Actor in self.recon_Actors])
                _ = ray.get([recon_Actor.computeMetrics.remote() for recon_Actor in self.recon_Actors])
                sample.allImagesPSNRList = np.concatenate([ray.get(recon_Actor.getPSNR.remote()) for recon_Actor in self.recon_Actors])
                sample.allImagesSSIMList = np.concatenate([ray.get(recon_Actor.getSSIM.remote()) for recon_Actor in self.recon_Actors])
                del unMeasuredIdxs_id, measuredIdxs_id, neighborIndices_id, neighborWeights_id, squareMask_id
            else:
                if self.sampleData.sampleType == 'DESI': self.reconImages = self.sampleData.squareAllImages*sample.squareMask
                else: self.reconImages = self.sampleData.allImages*sample.squareMask
                self.reconImages = np.array([computeReconIDW(self.reconImages[index], squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights) for index in range(0, len(self.reconImages))], dtype=np.float32)
                if self.sampleData.sampleType == 'DESI': self.reconImages = np.moveaxis(resize(np.moveaxis(self.reconImages, 0, -1), tuple(self.sampleData.finalDim), order=0), -1, 0)
                self.allImagesPSNRList = [compare_psnr(self.sampleData.allImages[index], self.reconImages[index], data_range=self.allImagesMax[index]) for index in range(0, len(self.sampleData.allImages))]
                self.allImagesSSIMList = [compare_ssim(self.sampleData.allImages[index], self.reconImages[index], data_range=self.allImagesMax[index]) for index in range(0, len(self.sampleData.allImages))]
        
        #Otherwise assume all images results are the same as for targeted channels; i.e. all channels were targeted
        else:
            sample.allImagesPSNRList = sample.chanImagesPSNRList
            sample.allImagesSSIMList = sample.chanImagesSSIMList
            
        #Prior to and for model training there is RD, but no ERD
        if not self.sampleData.trainFlag:
            
            #Compute RD; if every location has been scanned all positions are zero
            if len(squareUnMeasuredIdxs) == 0: 
                sample.squareRD = np.zeros(self.sampleData.squareDim, dtype=np.float32)
                sample.RD = np.zeros(self.sampleData.finalDim, dtype=np.float32)
            else: 
                #If dataAdjust is enabled, and using either DLADS or GLANDS, then can optionally rescale RDPP computation inputs to between 0 and 1 or standardize them
                if dataAdjust == 'rescale' and (erdModel=='DLADS' or erdModel=='GLANDS'):
                    minValues, maxValues = np.min(self.sampleData.squareChanImages, axis=(1,2)), np.max(self.sampleData.squareChanImages, axis=(1,2))
                    minMaxDiffs = (maxValues-minValues)
                    sample.RDPPs = np.moveaxis(abs(((np.moveaxis(self.sampleData.squareChanImages, 0, -1)-minValues)/minMaxDiffs)-((np.moveaxis(sample.squareChanReconImages, 0, -1)-minValues)/minMaxDiffs)), -1, 0)
                elif dataAdjust == 'standardize' and (erdModel=='DLADS' or erdModel=='GLANDS'):
                    meanValues, stdValues =  np.mean(self.sampleData.squareChanImages, axis=(1,2)), np.std(self.sampleData.squareChanImages, axis=(1,2))
                    sample.RDPPs = np.moveaxis(abs(((np.moveaxis(self.sampleData.squareChanImages, 0, -1)-meanValues)/stdValues)-((np.moveaxis(sample.squareChanReconImages, 0, -1)-meanValues)/stdValues)), -1, 0)
                else: sample.RDPPs = abs(self.sampleData.squareChanImages-sample.squareChanReconImages)
                
                computeRD(sample, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborDistances, self.cValue, self.bestCFlag, self.datagenFlag, False, self.liveOutputFlag, self.impModel)

            #Determine SSIM/PSNR between averaged RD and ERD
            maxRangeValue = np.max([sample.squareRD, sample.squareERD])
            sample.ERDPSNR = compare_psnr(sample.squareRD, sample.squareERD, data_range=maxRangeValue)
            sample.ERDSSIM = compare_ssim(sample.squareRD, sample.squareERD, data_range=maxRangeValue)
        
        #Resize RD(s) for final visualizations; has to be done here for live output case, but in complete() method otherwise
        if self.liveOutputFlag: self.resizeRD(sample)

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
        
        #Make sure samples is writable
        self.samples = copy.deepcopy(self.samples)
        
        #If all channel reconstructions are needed, then setup actors if in parallel, or load data into main memory
        if self.allChanEval or (imzMLExport and not self.sampleData.trainFlag):
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
        
        #If all channel evaluation and simulation then extract performance metrics, if not live output and not for training database generation
        if self.allChanEval and (self.sampleData.simulationFlag and not self.liveOutputFlag and not self.datagenFlag):
            for sample in tqdm(self.samples, desc='RD/Metrics Extraction', leave=False, ascii=asciiFlag): self.extractSimulationData(sample)
            
        #If exporting final reconstruction data to .imzML
        if imzMLExport and not self.sampleData.trainFlag:
            
            #Set the coordinates to save values for
            coordinates = list(map(tuple, list(np.ndindex(tuple(self.sampleData.finalDim)))))
            
            #Export all measured, reconstructed data in .imzML format
            reconImages = np.concatenate([ray.get(recon_Actor.getReconImages.remote()) for recon_Actor in self.recon_Actors])
            writer = ImzMLWriter(self.dir_sampleResults+self.sampleData.name+'_reconstructed', intensity_dtype=np.float32, mz_dtype=np.float32, spec_type='profile', mode='processed')
            for coord in coordinates: writer.addSpectrum(self.sampleData.mzFinal, reconImages[:, coord[0], coord[1]], (coord[1]+1, coord[0]+1))
            writer.close()
            del reconImages, writer
            
            #Export the equivalent ground-truth measured data here if needed
            #allImages = np.concatenate([ray.get(recon_Actor.getAllImages.remote()) for recon_Actor in self.recon_Actors])
            #writer = ImzMLWriter(self.dir_sampleResults+self.sampleData.name+'_groundTruth', intensity_dtype=np.float32, mz_dtype=np.float32, spec_type='profile', mode='processed')
            #for coord in coordinates: writer.addSpectrum(self.sampleData.mzFinal, allImages[:, coord[0], coord[1]], (coord[1]+1, coord[0]+1))
            #writer.close()
            #del allImages, writer
        
        #If all channel evaluation or imzMLExport (not training), close all images file reference if applicable, remove all recon images from memory, purge/reset ray
        if self.allChanEval or (imzMLExport and not self.sampleData.trainFlag):
            if not parallelization:
                self.sampleData.allImagesFile.close()
                del self.sampleData.allImages, self.reconImages, self.sampleData.allImagesFile
                if self.sampleData.sampleType == 'DESI':
                    self.sampleData.squareAllImagesFile.close()
                    del self.sampleData.squareAllImages, self.sampleData.squareAllImagesFile
            else:
                _ = [ray.get(recon_Actor.closeAllImages.remote()) for recon_Actor in self.recon_Actors]
                self.recon_Actors.clear()
                del self.recon_Actors
                
        #If this is a simulation, not for training database generation, then summarize PSNR/SSIM scores across all measurement steps
        if self.sampleData.simulationFlag and not self.datagenFlag:
            self.chanAvgPSNRList = [np.nanmean(sample.chanImagesPSNRList) for sample in self.samples]
            self.sumImagePSNRList = [sample.sumImagePSNR for sample in self.samples]
            self.chanAvgSSIMList = [np.nanmean(sample.chanImagesSSIMList) for sample in self.samples]
            self.sumImageSSIMList = [sample.sumImageSSIM for sample in self.samples]
            
            #If not MSI, then all image results are the same as for targeted channels
            if self.sampleData.sampleType == 'MALDI' or self.sampleData.sampleType == 'DESI':
                self.allAvgPSNRList = [np.nanmean(sample.allImagesPSNRList) for sample in self.samples]
                self.allAvgSSIMList = [np.nanmean(sample.allImagesSSIMList) for sample in self.samples]
            else:
                self.allAvgPSNRList = self.chanAvgPSNRList
                self.allAvgSSIMList = self.chanAvgSSIMList

        #If ERD was computed (i.e., when not a training run) summarize ERD PSNR/SSIM scores
        if not self.sampleData.trainFlag:
            self.ERDPSNRList = [sample.ERDPSNR for sample in self.samples]
            self.ERDSSIMList = [sample.ERDSSIM for sample in self.samples]
        
        #Do not generate visuals for c value optimization
        if not self.bestCFlag: 
            #Resize RDs and generate visualizations if they were not created during operation; if in parallel purge/reset ray after computation
            if not self.liveOutputFlag:
                for sample in self.samples: self.resizeRD(sample)
                if parallelization:
                    futures = [(self.samples[index], self.sampleData, self.dir_progression, self.dir_chanProgressions, self.datagenFlag) for index in range(0, len(self.samples))]
                    computePool = Pool(numberCPUS)
                    results = computePool.starmap_async(visualize_serial, futures)
                    computePool.close()
                    computePool.join()
                    #gc.collect() #Try manually running garbage collection afer deletion instead of reseting ray completely
                    resetRay(numberCPUS)
                else: 
                    _ = [visualize_serial(sample, self.sampleData, self.dir_progression, self.dir_chanProgressions, self.datagenFlag) for sample in tqdm(self.samples, desc='Steps', leave=False, ascii=asciiFlag)]

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
def visualize_serial(sample, sampleData, dir_progression, dir_chanProgressions, datagenFlag):

    #If running visualization in parallel, try to set backend to avoid main thread/loop issues
    if parallelization: matplotlib.use('Agg')

    #Turn percent measured into a string
    percMeasured = "{:.2f}".format(sample.percMeasured)
    
    #Turn metrics into strings
    if sampleData.simulationFlag and not datagenFlag: 
        sumImagePSNR = "{:.2f}".format(sample.sumImagePSNR)
        sumImageSSIM = "{:.2f}".format(sample.sumImageSSIM)
        chanImageAvgPSNR = "{:.2f}".format(np.nanmean(sample.chanImagesPSNRList))
        chanImageAvgSSIM = "{:.2f}".format(np.nanmean(sample.chanImagesSSIMList))
        allImageAvgPSNR = "{:.2f}".format(np.nanmean(sample.allImagesPSNRList))
        allImageAvgSSIM = "{:.2f}".format(np.nanmean(sample.allImagesSSIMList))
    if sampleData.simulationFlag and not sampleData.trainFlag: 
        erdPSNR = "{:.2f}".format(sample.ERDPSNR)
        erdSSIM = "{:.2f}".format(sample.ERDSSIM)
        
    #For each of the channels, generate visuals
    for chanNum in range(0, sampleData.numChannels):
        
        #Find minimum and maximum channel values for colorbars
        chanMinValue, chanMaxValue = np.min(sampleData.chanImages[chanNum]), np.max(sampleData.chanImages[chanNum])
        
        #Turn metrics into strings
        chanLabel = str(sampleData.chanValues[chanNum])
        if sampleData.simulationFlag and not datagenFlag:
            chanImagesPSNR = "{:.2f}".format(sample.chanImagesPSNRList[chanNum])
            chanImagesSSIM = "{:.2f}".format(sample.chanImagesSSIMList[chanNum])
        
        #If a simulation, then need room on visualizations for showing ERD, ground-truth, and ground-truth difference
        if sampleData.simulationFlag: f = plt.figure(figsize=(20,10))
        else: f = plt.figure(figsize=(20,5.3865))
        
        #TODO: If a simulation, then need room on visualizations for showing ERD, ground-truth, and ground-truth difference; if no metrics, then don't need extra title room?
        #if sampleData.simulationFlag and not datagenFlag: f = plt.figure(figsize=(20,9))
        #elif sampleData.simulationFlag: f = plt.figure(figsize=(20,10))
        #else: f = plt.figure(figsize=(20,5.3865))
        
        #Generate and apply a plot title, with metrics if applicable
        plotTitle = r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Channel:\ }$" + chanLabel + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured
        if sampleData.simulationFlag and not datagenFlag:
            plotTitle += '\n' + r"$\bf{PSNR\ -\ All\ Channel\ Avg:\ }$" + allImageAvgPSNR + r"$\bf{\ \ Targeted Channel Avg:\ }$" + chanImageAvgPSNR + r"$\bf{\ \ Targeted\ Channel:\ }$" + chanImagesPSNR
            plotTitle += '\n' + r"$\bf{SSIM\ -\ All\ Channel\ Avg:\ }$" + allImageAvgSSIM + r"$\bf{\ \ Targeted Channel Avg:\ }$" + chanImageAvgSSIM + r"$\bf{\ \ Targeted\ Channel:\ }$" + chanImagesSSIM
        plt.suptitle(plotTitle)
        
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
        
        if not sampleData.trainFlag:
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
        plt.close(f)

        #Do borderless saves for each channel image here; mask will be the same as produced in the progression output
        if not sampleData.trainFlag:
            saveLocation = dir_chanProgressions[chanNum] + 'erd_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
            borderlessPlot(sample.ERDs[chanNum], saveLocation, cmap='viridis', vmin=0)
        
        if sampleData.simulationFlag:
            saveLocation = dir_chanProgressions[chanNum] + 'rd_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
            borderlessPlot(sample.RDs[chanNum], saveLocation, cmap='viridis', vmin=0)
            
            saveLocation = dir_chanProgressions[chanNum] + 'groundTruth_channel_' + chanLabel + '.png'
            borderlessPlot(sampleData.chanImages[chanNum], saveLocation, cmap='hot', vmin=chanMinValue, vmax=chanMaxValue)

        saveLocation = dir_chanProgressions[chanNum] + 'reconstruction_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.chanReconImages[chanNum], saveLocation, cmap='hot', vmin=chanMinValue, vmax=chanMaxValue)
        
        saveLocation = dir_chanProgressions[chanNum] + 'measured_channel_' + chanLabel + '_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.chanImages[chanNum], saveLocation, cmap='hot', vmin=chanMinValue, vmax=chanMaxValue)
        
    #For the overall progression, get min/max of the ground-truth sum image for visualization
    sumImageMinValue, sumImageMaxValue = np.min(sampleData.sumImage), np.max(sampleData.sumImage)
    
    #If a simulation, then need room on visualizations for showing ERD, ground-truth, and ground-truth difference
    if sampleData.simulationFlag: f = plt.figure(figsize=(20,10))
    else: f = plt.figure(figsize=(20,5.3865))

    #Generate and apply a plot title, with metrics if applicable
    plotTitle = r"$\bf{Sample:\ }$" + sampleData.name + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured
    if sampleData.simulationFlag and not sampleData.trainFlag and not datagenFlag:
        plotTitle += '\n' + r"$\bf{PSNR\ -\ All\ Channel\ Avg:\ }$" + allImageAvgPSNR + r"$\bf{\ \ Sum\ Image: }$" + sumImagePSNR + r"$\bf{\ \ ERD:\ }$" + erdPSNR 
        plotTitle += '\n' + r"$\bf{SSIM\ -\ All\ Channel\ Avg:\ }$" + allImageAvgSSIM + r"$\bf{\ \ Sum\ Image: }$" + sumImageSSIM + r"$\bf{\ \ ERD:\ }$" + erdSSIM
    elif sampleData.simulationFlag and sampleData.trainFlag and not datagenFlag:
        plotTitle += '\n' + r"$\bf{PSNR\ -\ All\ Channel\ Avg:\ }$" + allImageAvgPSNR + r"$\bf{\ \ Sum\ Image: }$" + sumImagePSNR
        plotTitle += '\n' + r"$\bf{SSIM\ -\ All\ Channel\ Avg:\ }$" + allImageAvgSSIM + r"$\bf{\ \ Sum\ Image: }$" + sumImageSSIM
    plt.suptitle(plotTitle)
    
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
    
    if not sampleData.trainFlag:
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
    plt.close(f)

    #Borderless saves
    saveLocation = dir_progression + 'reconstruction_sumImage' + '_iter_' + str(sample.iteration) +  '_perc_' + str(sample.percMeasured) + '.png'
    borderlessPlot(sample.sumImageReconImage, saveLocation, cmap='hot', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    
    saveLocation = dir_progression + 'mask_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
    borderlessPlot(sample.mask, saveLocation, cmap='gray')
    
    if not sampleData.trainFlag:
        saveLocation = dir_progression + 'ERD_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.ERD, saveLocation, cmap='viridis')
    
    saveLocation = dir_progression + 'measured_sumImage_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
    borderlessPlot(sample.sumImage, saveLocation, cmap='hot', vmin=sumImageMinValue, vmax=sumImageMaxValue)
    
    if sampleData.simulationFlag:
        saveLocation = dir_progression + 'RD_iter_' + str(sample.iteration) + '_perc_' + str(sample.percMeasured) + '.png'
        borderlessPlot(sample.RD, saveLocation, cmap='viridis')

def runSampling(sampleData, cValue, model, percToScan, percToViz, bestCFlag, oracleFlag, lineVisitAll, liveOutputFlag, dir_Results, datagenFlag, impModel, tqdmHide, samplingProgress_Actor=None, percProgUpdate=None):

    #Make sure random selection is consistent
    if consistentSeed: 
        np.random.seed(0)
        random.seed(0)
    
    #If groupwise is active, specify how many points should be scanned each step
    if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random') and percToScan != None: sampleData.pointsToScan = int(np.ceil(((sampleData.stopPerc/100)*sampleData.area)/(sampleData.stopPerc/percToScan)))
    elif sampleData.scanMethod == 'linewise' and sampleData.useMaskFOV: sampleData.pointsToScan = [int(np.ceil((sampleData.stopPerc/100)*np.sum(sampleData.maskFOV[lineIndex]))) for lineIndex in range(0, sampleData.finalDim[0])]
    elif sampleData.scanMethod == 'linewise': sampleData.pointsToScan = [int(np.ceil((sampleData.stopPerc/100)*sampleData.finalDim[1])) for _ in range(0, sampleData.finalDim[0])]
    
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
    elif sampleData.scanMethod == 'linewise' and len(sampleData.linesToScan)-np.sum(np.sum(sample.mask, axis=1)>0) == 0: completedRunFlag = True
    if not datagenFlag and np.sum(sample.physicalERD) == 0: completedRunFlag = True
    
    #Perform the first update for the result
    result.update(sample)
    
    #Until the stopping criteria has been met
    with tqdm(total = round(float(sampleData.stopPerc),2), desc = '% Sampled', leave=False, ascii=asciiFlag, disable=tqdmHide) as pbar:

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
            newIdxs = findNewMeasurementIdxs(sample, sampleData, result, model, cValue, percToScan, oracleFlag, bestCFlag, datagenFlag)
            
            #Perform measurements, reconstructions and ERD/RD computations
            if len(newIdxs) != 0: sample.performMeasurements(sampleData, result, newIdxs, model, cValue, bestCFlag, oracleFlag, datagenFlag, False)
            else: break
            
            #Check stopping criteria
            if (sampleData.scanMethod == 'pointwise' or sampleData.scanMethod == 'random' or not lineVisitAll) and (sample.percMeasured >= sampleData.stopPerc): completedRunFlag = True
            elif sampleData.scanMethod == 'linewise' and len(sampleData.linesToScan)-np.sum(np.sum(sample.mask, axis=1)>0) == 0: completedRunFlag = True
            if not datagenFlag and np.sum(sample.physicalERD) == 0: completedRunFlag = True
            
            #If viz limit, only update when percToViz has been met; otherwise update every iteration
            if ((percToViz != None) and ((sample.percMeasured-result.percsMeasured[-1]) >= percToViz)) or (percToViz == None) or (sampleData.scanMethod == 'linewise') or completedRunFlag: result.update(sample)
            
            #If using a global progress bar and percProgUpdate has been reached, then update the global sampling progress actor
            if samplingProgress_Actor != None and tqdmHide and (sample.percMeasured-lastPercMeasured >= percProgUpdate): 
                _ = ray.get(samplingProgress_Actor.update.remote(sample.percMeasured-lastPercMeasured))
                lastPercMeasured = copy.deepcopy(sample.percMeasured)

            #Update the progress bar
            if not tqdmHide:
                pbar.n = np.clip(round(sample.percMeasured,2), 0, sampleData.stopPerc)
                pbar.refresh()
            
        #MSI experimental specific; after scanning has completed, store data as a readable hdf5 file on disk; optimize chunks for loading whole m/z images; close file reference and delete object
        if not sampleData.simulationFlag and not sampleData.postFlag and (sampleData.sampleType == 'MALDI' or sampleData.sampleType == 'DESI'): 
            sampleData.allImages = sampleData.allImagesFile.create_dataset(name='allImages', data=sampleData.allImages, chunks=(1, sampleData.finalDim[0], sampleData.finalDim[1]))
            sampleData.allImagesFile.close()
            del sampleData.allImagesFile
            del sampleData.allImages
            
    return result
    
#Compute approximated Reduction in Distortion (RD) values
def computeRD(sample, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborDistances, cValue, bestCFlag, datagenFlag, update, liveOutputFlag, impModel):
    
    #If a full calculation of RD then use the squareUnMeasured locations, otherwise find those that should be updated
    if not update: 
        squareUnMeasuredLocations = squareUnMeasuredIdxs
        neighborDistances = neighborDistances[:,0]
    else:
        squareUnMeasuredLocations = np.empty((0,2), dtype=int)
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
    else: windowSizes = (np.ones((len(sigmaValues)), dtype=np.float32)*staticWindowSize)
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
    
    #Set RD values at measured locations to zero
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
    
    #Create array to hold features
    feature = np.zeros((np.shape(squareUnMeasuredIdxs)[0],6), dtype=np.float32)
    
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

#Prepare data for DLADS or GLANDS model input, with the option to rescale the input recon data to between 0 and 1, or standardize it
def prepareInput(sample, numChannel):
    inputReconImage = sample.squareChanReconImages[numChannel]
    if dataAdjust == 'rescale' and (erdModel=='DLADS' or erdModel=='GLANDS'):
        minValue = np.min(inputReconImage)
        inputReconImage = (inputReconImage-minValue)/(np.max(inputReconImage)-minValue)
    elif dataAdjust == 'standardize' and (erdModel=='DLADS' or erdModel=='GLANDS'):
        inputReconImage = (inputReconImage-np.mean(inputReconImage))/np.std(inputReconImage)
    if erdModel == 'DLADS': return np.dstack((sample.squareMask, inputReconImage*(1-sample.squareMask), inputReconImage*sample.squareMask))
    elif erdModel == 'GLANDS': return np.dstack((sample.squareMask, inputReconImage*sample.squareMask))

#Determine the Estimated Reduction in Distortion
def computeERD(sample, sampleData, model, squareUnMeasuredIdxs, squareMeasuredIdxs):

    #Compute the ERD with the prescribed model; if configured to, only use a single channel
    if not chanSingle:
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
            for chanNum in range(0, len(sample.squareERDs)): sample.squareERDs[chanNum, squareUnMeasuredIdxs[:, 0], squareUnMeasuredIdxs[:, 1]] = ray.get(model.generateERD.remote(sample.polyFeatures[chanNum]))
        elif erdModel == 'DLADS': 
        
            #First try inferencing all m/z channels at the same time 
            if not sampleData.OOM_multipleChannels:
                try: sample.squareERDs = ray.get(model.generateERD.remote(makeCompatible([prepareInput(sample, chanNum) for chanNum in range(0, len(sample.squareERDs))]))).copy()
                except: 
                    sampleData.OOM_multipleChannels = True
                    if (len(gpus) > 0): print('\nWarning - Could not inference ERD for all channels of sample '+sampleData.name+' simultaneously on system GPU; will try processing channels iteratively.')
                    if (len(gpus) == 0): print('\nWarning - Could not inference ERD for all channels of sample '+sampleData.name+' simultaneously on system; will try processing channels iteratively.')
            
            #If multiple channels causes an OOM, then try running each channel through on its own
            if sampleData.OOM_multipleChannels and not sampleData.OOM_singleChannel:
                try: sample.squareERDs = np.asarray([ray.get(model.generateERD.remote(makeCompatible(prepareInput(sample, chanNum))))[0,:,:].copy() for chanNum in range(0, len(sample.squareERDs))])
                except: sampleData.OOM_singleChannel = True
            
            #If an OOM occured for both mutiple and single channel inferencing, then exit; need to either restart program with no GPUs, or there isn't enough system RAM
            if sampleData.OOM_multipleChannels and sampleData.OOM_singleChannel and (len(gpus) > 0): sys.exit('Error - Sample '+sampleData.name+' dimensions are too high for the ERD to be inferenced on system GPU; please try disabling the GPU in the CONFIG.')
            if sampleData.OOM_multipleChannels and sampleData.OOM_singleChannel and (len(gpus) == 0): sys.exit('Error - Sample '+sampleData.name+' dimensions are too high for the ERD to be inferenced on this system by the loaded model.')
            
    else:
        if erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
            ERDValues = ray.get(model.generateERD.remote(sample.polyFeatures[0]))
            for chanNum in range(0, len(sample.squareERDs)): sample.squareERDs[chanNum, squareUnMeasuredIdxs[:, 0], squareUnMeasuredIdxs[:, 1]] = ERDValues
        elif erdModel == 'DLADS': 
            sample.squareERDs[0] = ray.get(model.generateERD.remote(makeCompatible(prepareInput(sample, 0))))[0,:,:].copy()
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
                if len(newIdxs) >= sampleData.pointsToScan[lineToScanIdx]: break
                
            #Convert to array for indexing
            newIdxs = np.asarray(newIdxs)
            
            #Sort columns for progressive physical scanning order
            newIdxs[:,1] = np.sort(newIdxs[:,1])
            
        #If points on the line should be selected in one step/group
        elif lineMethod == 'percLine' and linePointSelection == 'group':
            indexes = np.sort(np.argsort(sample.physicalERD[lineToScanIdx])[::-1][:sampleData.pointsToScan[lineToScanIdx]])
            newIdxs = np.column_stack([np.ones(len(indexes), dtype=np.float32)*lineToScanIdx, indexes]).astype(int)
        
        #If all the points on a chosen line should be scanned
        if lineMethod =='fullLine':
            indexes = np.sort(np.argsort(sample.physicalERD[lineToScanIdx])[::-1])
            newIdxs = np.column_stack([np.ones(len(indexes), dtype=np.float32)*lineToScanIdx, indexes]).astype(int)
        
        #==========================================
        #PARTIAL LINE BY START/END POINTS
        #==========================================
        #Choose segment to scan on line
        if lineMethod == 'segLine': 
            if segLineMethod == 'otsu':
                indexes = np.sort(np.where(sample.physicalERD[lineToScanIdx]>skimage.filters.threshold_otsu(sample.physicalERD, nbins=100))[0])
                if len(indexes)>0: 
                    indexes = np.arange(indexes[0],indexes[-1]+1)
                    newIdxs = np.column_stack([np.ones(len(indexes), dtype=np.float32)*lineToScanIdx, indexes]).astype(int)
            elif segLineMethod == 'minPerc':
                indexes = np.sort(np.argsort(sample.physicalERD[lineToScanIdx])[::-1][:sampleData.pointsToScan[lineToScanIdx]])
                if len(indexes)>0: newIdxs = np.column_stack([np.ones(indexes[-1]-indexes[0]+1, dtype=np.float32)*lineToScanIdx, np.arange(indexes[0],indexes[-1]+1)]).astype(int)
        #==========================================
        
        #==========================================
        #SELECTION SAFEGUARD
        #==========================================
        #If there are not enough locations selected, then return no new measurement locations which will terminate scanning
        if len(newIdxs) < int(round(0.01*sample.mask.shape[1])): return []
        #==========================================
        
    return newIdxs

#Re-index a set of m/z values to a common grid
@jit(nopython=True, nogil=True)
def mzFastIndex(mz, values, mzLowerIndex, mzPrecision, mzRound, mzInitialCount):
    indices = np.empty(len(mz), dtype=np.float32)
    np.round(np.floor(mz/mzPrecision)*mzPrecision, mzRound, indices)
    mzValues = np.zeros(mzInitialCount, dtype=np.float32)
    mzValues[(indices/mzPrecision).astype(np.int32)-mzLowerIndex] = values
    return mzValues

#Calculate k-nn and determine inverse distance weights
def findNeighbors(measuredIdxs, unMeasuredIdxs):
    neighborDistances, neighborIndices = NearestNeighbors(n_neighbors=numNeighbors).fit(measuredIdxs).kneighbors(unMeasuredIdxs)
    unNormNeighborWeights = 1.0/(neighborDistances**2.0)
    neighborWeights = unNormNeighborWeights/(np.sum(unNormNeighborWeights, axis=1))[:, np.newaxis]
    return neighborIndices, neighborWeights, neighborDistances

#Perform the reconstruction using IDW (inverse distance weighting); retrieve measured values, compute reconstruction values, and combine into a new image; if 3D do all channels at once
def computeReconIDW(inputImage, squareMeasuredIdxs, squareUnMeasuredIdxs, neighborIndices, neighborWeights):
    reconImage = copy.deepcopy(inputImage)
    if len(squareUnMeasuredIdxs) > 0:
        if len(inputImage.shape) == 3: reconImage[:, squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]] = np.sum(inputImage[:, squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]][:, neighborIndices]*neighborWeights, axis=-1)
        else: reconImage[squareUnMeasuredIdxs[:,0], squareUnMeasuredIdxs[:,1]] = np.sum(inputImage[squareMeasuredIdxs[:,0], squareMeasuredIdxs[:,1]][neighborIndices]*neighborWeights, axis=1)
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
    #if dataRescaling: outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(conv8)
    #else: outputs = Conv2D(1, 1, activation='relu', padding='same')(conv8)
    outputs = Conv2D(1, 1, activation='relu', padding='same')(conv8)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
    
def aug_unet(numFilters, numChannels):
    inputs = Input(shape=(None,None,numChannels), batch_size=None)
    
    combinedInputOutput = tf.stack([inputs, outputs], axis=0)
    #combinedInputOutput = tf.stack([inputs.to_tensor(), outputs.to_tensor()], axis=0)
    combinedInputOutput = tf.keras.layers.RandomFlip('horizontal_and_vertical')(combinedInputOutput)
    combinedInputOutput = tf.keras.layers.RandomRotation(factor = (-0.125, 0.125), fill_mode='constant')(combinedInputOutput)
    combinedInputOutput = tf.keras.layers.RandomTranslation(height_factor=(-0.25, 0.25), width_factor=(-0.25, 0.25), fill_mode = 'constant')(combinedInputOutput)
    augInputs, augOutputs = combinedInputOutput[0], combinedInputOutput[1]
    
    
    conv0 = downConv(numFilters, augInputs)
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
    return tf.keras.Model(inputs=inputs, outputs=augOutputs)

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
    
#Truncate a value to a given precision 
def truncate(value, decimalPlaces=0): return np.trunc(value*10**decimalPlaces)/(10**decimalPlaces)

def borderlessPlot(image, saveLocation, cmap='viridis', vmin=None, vmax=None):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(image, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(saveLocation, bbox_inches=extent)
    plt.close(fig)

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

