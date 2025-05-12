#==================================================================
#REMOTE
#==================================================================

#Define actor for utilizing trained models, enabling worker reuse; do not allocate a dedicated CPU thread
@ray.remote(num_cpus=0)
class Model_Actor:
    def __init__(self, erdModel, modelDirectory, modelName, gpuNum=-1):
        
        setupLogging()
        
        #Store local variables
        self.erdModel = erdModel
        self.modelDirectory = modelDirectory
        self.modelName = modelName
        
        #Reload system GPU ids into environment (bypass Ray hiding them, which leads to the incorrect GPU selection)
        os.environ["CUDA_VISIBLE_DEVICES"] = systemGPUs
        if gpuNum >=0: self.local_gpus = [gpuNum]
        else: self.local_gpus = []
        
    #Must perform model loading/setup in sequence to avoid potential file conflicts (extracting model archive) when creating multiple actors
    def setup(self):
        if 'SLADS' in self.erdModel: self.model = np.load(self.modelDirectory + self.modelName +'.npy', allow_pickle=True).item()
        elif self.erdModel == 'DLADS-TF-DEP': self.model = DLADS_TF_DEP(False, self.local_gpus, self.modelDirectory, self.modelName)
        elif self.erdModel == 'DLADS-TF-SYNC': self.model = DLADS_TF_SYNC(False, self.local_gpus, self.modelDirectory, self.modelName)
        elif self.erdModel == 'DLADS-PY-SYNC': self.model = DLADS_PY_SYNC(False, self.local_gpus, self.modelDirectory, self.modelName)
        elif self.erdModel == 'GLANDS': sys.exit('\nError - GLANDS loading not yet defined')
        else: sys.exit('\nError - Specified model has not been defined.')
    
    #Preventing multiple concurrent calls, inference an input using the trained model
    def generate(self, input, reconOnly=False):
        return self.model.predict(input)

#Ray actor for holding global progress in parallel sampling operations
@ray.remote(num_cpus=0)
class SamplingProgress_Actor:
    def __init__(self): 
        setupLogging()
        self.current = 0.0
    def update(self, amount): self.current += amount
    def getCurrent(self): return self.current

#Ray actor for computing reconstructions of all images in parallel
#Application of mask, computation of reconstruction, and metric calculation are purposefully split to prevent OOM errors
@ray.remote
class Recon_Actor:
    
    #Set internal parameters for handling image reconstruction process
    def __init__(self, indexes, sampleType, squareDim, finalDim, allImagesMin, allImagesMax, allImagesPath, squareOpticalImage, erdModel):
        
        setupLogging()
        
        self.indexes = indexes
        self.sampleType = sampleType
        self.squareDim = squareDim
        self.finalDim = finalDim
        self.allImagesMin = allImagesMin[self.indexes]
        self.allImagesMax = allImagesMax[self.indexes]
        self.squareOpticalImage = squareOpticalImage
        self.erdModel = erdModel
        
        if self.erdModel == 'GLANDS': sys.exit('\nError - Parallelized recon actors are not supported for GLANDS model.') 
        
        #Load in reference for all images file and setup for performing reconstructions
        self.allImagesFile = h5py.File(allImagesPath, 'r')
        self.allImages = self.allImagesFile['allImages'][self.indexes]
    
    #Compute reconstructions for a given measurement step
    def computeRecon(self, tempScanData, squareMask, mask):
        
        #Apply mask to the complete data to obtain sparse images, then perform reconstructions
        if self.sampleType == 'MALDI': 
            self.reconImages = self.allImages*squareMask
            for index in range(0, len(self.indexes)): self.reconImages[index] = computeReconIDW(self.reconImages[index], tempScanData)
        elif self.sampleType == 'DESI': 
            self.reconImages = np.zeros((len(self.indexes), self.finalDim[0], self.finalDim[1]))
            for index in range(0, len(self.indexes)): self.reconImages[index] = resize(computeReconIDW(resize(self.allImages[index], tuple(self.squareDim), order=0)*squareMask, tempScanData), tuple(self.finalDim), order=0)
            self.reconImages = (self.reconImages*(1-mask)) + (self.allImages*mask)
        
        #
        #elif self.sampleType == 'DESI': self.reconImages = self.squareAllImages*squareMask
        #self.reconImages = np.moveaxis(resize(np.moveaxis(self.allImages, 0, -1), tuple(squareDim), order=0), -1, 0)*squareMask
        #
        #Compute IDW reconstructions
        #for index in range(0, len(self.indexes)): self.reconImages[index] = computeReconIDW(self.reconImages[index], tempScanData)
        #
        #Resize DESI data back to physical dimensions and copy back the original measured values to reconstructions; creating new holding array for resized results and looping is needed for memory efficiency
        #if self.sampleType == 'DESI': 
        #    
        #    for index in range(0, len(self.indexes)): resizedReconImages[index] = resize(self.reconImages[index], tuple(self.finalDim), order=0)
        #    self.reconImages = resizedReconImages
        #    del resizedReconImages
        #    self.reconImages = (self.reconImages*(1-mask)) + (self.allImages*mask)
        
    #Compute NRMSE/SSIM for reconstructions
    def computeMetrics(self):
        self.imagesNRMSEList, self.imagesSSIMList, self.imagesPSNRList = [], [], []
        for index in range(0, len(self.indexes)):
            score_PSNR, score_SSIM, score_NRMSE  = compareImages(self.allImages[index], self.reconImages[index], self.allImagesMin[index], self.allImagesMax[index])
            self.imagesPSNRList.append(score_PSNR)
            self.imagesSSIMList.append(score_SSIM)
            self.imagesNRMSEList.append(score_NRMSE)
    
    #Return allImages already loaded; Warning: Performing any operations beyond just the return will induce significant copy overhead!
    def getAllImages(self):
        return self.allImages
    
    def getNRMSE(self):
        return self.imagesNRMSEList
    
    def getSSIM(self):
        return self.imagesSSIMList
        
    def getPSNR(self):
        return self.imagesPSNRList
    
    #Obtain reconstructions; Warning: Performing any operations beyond just the return will induce significant copy overhead!
    def getReconImages(self):
        return self.reconImages
    
    #Close any open filesy
    def closeAllImages(self):
        self.allImagesFile.close()
        del self.allImagesFile
        _ = cleanup()

#Ray actor for holding MSI data in a shared memory location for parallel operations; do not allocate a dedicated CPU thread
@ray.remote(num_cpus=0)
class Reader_MSI_Actor:
    
    #Create buffers for holding all MSI images, the specified channel images, and the sum of all values
    def __init__(self, sampleType, readAllMSI, mzNum, chanNum, yDim, xDim, allImagesDataPath, allImagesPath, overwriteAllChanFiles):
        
        setupLogging()
        
        self.readAllMSI = readAllMSI
        self.sampleType = sampleType
        self.yDim = yDim
        self.xDim = xDim
        self.allImagesDataPath = allImagesDataPath
        self.allImagesPath = allImagesPath
        self.overwriteAllChanFiles = overwriteAllChanFiles
        
        if self.readAllMSI:
            if self.overwriteAllChanFiles: 
                self.allImages = np.zeros((yDim, xDim, mzNum))
            else:
                self.allImagesMin = np.load(self.allImagesDataPath + 'allImagesMin.npy', allow_pickle=True)
                self.allImagesMax = np.load(self.allImagesDataPath + 'allImagesMax.npy', allow_pickle=True)
        
        self.sumImage = np.zeros((yDim, xDim))
        self.chanImages = np.zeros((yDim, xDim, chanNum))
        if sampleType == 'DESI': self.mzDataComplete, self.origTimesComplete, self.lineNumComplete, self.readScanFiles = [], [], [], []
    
    #Set internal reference for indexing different coordinates
    def setCoordinates(self, coordinates):
        self.coordinates = coordinates
    
    #Assign m/z data to specified locations; for DESI, if the m/z line data could not be interpolated in parallel, then setup for sequential fallback in shared memory space
    def setValues(self, indexData, mzDataTotal, chanDataTotal, sumDataTotal, origTimesTotal=None, lineNumTotal=None, newReadScanFiles=None, allDataInterpFailTotal=None):
        if self.sampleType == 'MALDI':
            yLocations, xLocations = self.coordinates[indexData,1], self.coordinates[indexData,0]
            self.sumImage[yLocations, xLocations] = sumDataTotal
            self.chanImages[yLocations, xLocations, :] = chanDataTotal
            if self.overwriteAllChanFiles and self.readAllMSI: self.allImages[yLocations, xLocations, :] = mzDataTotal
        elif self.sampleType == 'DESI':
            self.sumImage[lineNumTotal, :] = sumDataTotal
            self.chanImages[lineNumTotal, :, :] = np.moveaxis(chanDataTotal, 1, -1)
            self.readScanFiles += newReadScanFiles
            for indexNum in range(0, len(mzDataTotal)):
                if allDataInterpFailTotal[indexNum]: 
                    self.mzDataComplete.append(mzDataTotal[indexNum])
                    self.origTimesComplete.append(origTimesTotal[indexNum])
                    self.lineNumComplete.append(lineNumTotal[indexNum])
                elif self.overwriteAllChanFiles and self.readAllMSI: 
                    self.allImages[lineNumTotal[indexNum], :, :] = np.moveaxis(mzDataTotal[indexNum], 0, -1)
    
    #Write data to a .hdf5 file at the prespecified location on disk and return the min/max values for each m/z
    #With DESI data, save square versions for future reconstructions
    def writeToDisk(self, squareDim):
        if self.overwriteAllChanFiles: 
            self.allImagesMin, self.allImagesMax = np.min(self.allImages, axis=(0,1)), np.max(self.allImages, axis=(0,1))
            np.save(self.allImagesDataPath + 'allImagesMin', self.allImagesMin)
            np.save(self.allImagesDataPath + 'allImagesMax', self.allImagesMax)
            allImagesFile = h5py.File(self.allImagesPath, 'a')
            _ = allImagesFile.create_dataset(name='allImages', data=np.moveaxis(self.allImages, -1, 0), chunks=(1, self.yDim, self.xDim))
        return self.allImagesMin, self.allImagesMax
        
    #Return list of scan files that have already been read
    def getReadScanFiles(self):
        return self.readScanFiles
    
    #Place all images data in shared memory with a persistent reference to survive this actor
    def shareAllImages(self): 
        return ray.put(self.allImages, _owner=None)
    
    #Return all images data; Warning: Performing any operations beyond just the return will induce significant copy overhead!
    def getAllImages(self): 
        return self.allImages
    
    #Return channel images data; Warning: Performing any operations beyond just the return will induce significant copy overhead!
    def getChanImages(self): 
        return self.chanImages
        
    #Return channel images data for specified locations
    def getChanImagesNewIdxs(self, row, cols):
        return self.chanImages[row, cols, :]
    
    #Return sum image data; Warning: Performing any operations beyond just the return will induce significant copy overhead!
    def getSumImage(self): 
        return self.sumImage
        
    #Return sum image data for specified locations
    def getSumImageNewIdxs(self, row, cols):
        return self.sumImage[row, cols]
    
    #Return a list of the read scan files for DESI MSI
    def getReadScanFiles(self):
        return self.readScanFiles
    
    #Align missing DESI m/z line data to a regular grid and store in shared memory space
    def interpolateDESI(self, newTimes):
        for index in range(0, len(self.lineNumComplete)): self.allImages[self.lineNumComplete[index], :, :] = np.moveaxis(interp1d(self.origTimesComplete[index], self.mzDataComplete[index], axis=-1, bounds_error=False, kind='linear', fill_value=0)(newTimes), 0, -1)

#Read in the sample MSI data for a set of indexes and set those values in shared memory location; must use blocking call (ray.get) to prevent data corruption
@ray.remote
def msi_parhelper(allImagesActor, format, readAllMSI, scanFileNames, indexData, chanValues, mzFinalBinEdges, mzRanges, sampleType, mzLowerBound, mzUpperBound, mask, newTimes, finalDim, sampleWidth, scanRate, overwriteAllChanFiles, impFlag=False, postFlag=False, impOffset=None, scanMethod=None, lineMethod=None, physicalLineNums=None, ignoreMissingLines=None, missingLines=None, unorderedNames=None):
    
    setupLogging()
    
    mzDataTotal, chanDataTotal, sumDataTotal = [], [], []
    if sampleType == 'MALDI':
        
        #Load the single imzML file expected for MALDI
        data = ImzMLParser(scanFileNames[0])
        
        #Process each of the files assigned to this helper 
        #Initially load data as strings (avoid accuracy loss from direct 32-to-64-bit casting)
        #CWT choice/parameters - https://pmc.ncbi.nlm.nih.gov/articles/PMC9865071/ and #https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-4
        for index in indexData:
            mzs, ints = data.getspectrum(index)
            mzs, ints = np.asarray(mzs, dtype='str').astype(np.float64), np.asarray(ints, dtype='str').astype(np.float64)
            if data.spectrum_mode == 'profile':
                peakLocations = find_peaks_cwt(ints, np.arange(1,30), min_snr=3.0)
                mzs, ints = mzs[peakLocations], ints[peakLocations]
            sumDataTotal.append(np.sum(ints))
            chanDataTotal.append(np.asarray([np.sum(ints[bisect_left(mzs, mzRange[0]):bisect_right(mzs, mzRange[1])]) for mzRange in mzRanges]))      
            if overwriteAllChanFiles and readAllMSI: mzDataTotal.append(binned_statistic(mzs, ints, statistic='sum', bins=mzFinalBinEdges, range=(mzLowerBound, mzUpperBound))[0])
        
        #Remove any invalid values and transfer information to shared memory actor
        sumDataTotal = np.nan_to_num(sumDataTotal, nan=0, posinf=0, neginf=0)
        chanDataTotal = np.nan_to_num(chanDataTotal, nan=0, posinf=0, neginf=0)
        mzDataTotal = np.nan_to_num(mzDataTotal, nan=0, posinf=0, neginf=0)
        _ = ray.get(allImagesActor.setValues.remote(indexData, mzDataTotal, chanDataTotal, sumDataTotal))
        
        #Close the file
        del data
        _ = cleanup()
    
    elif sampleType == 'DESI':
        
        #For each of the available files assigned to this helper
        origTimesTotal, readScanFiles, lineNumTotal, allDataInterpFailTotal = [], [], [], []
        for index in indexData:
            
            #Read and process the file
            data = readDESI(scanFileNames[index], format, chanValues, mzRanges, mzLowerBound, mzUpperBound, mzFinalBinEdges, readAllMSI, overwriteAllChanFiles, impFlag, postFlag, physicalLineNums, ignoreMissingLines, missingLines, unorderedNames)
            
            #If the file could be handled
            if data: 
                
                #Extract and store obtained data, interpolating as needed
                scanFileName, lineNum, origTimes, chanDataLine, sumImageLine, mzDataLine = data
                readScanFiles.append(scanFileName)
                origTimesTotal.append(origTimes)
                lineNumTotal.append(lineNum)
                chanDataTotal.append(interp1d(origTimes, chanDataLine, axis=-1, bounds_error=False, kind='linear', fill_value=0)(newTimes))
                sumDataTotal.append(interp1d(origTimes, sumImageLine, axis=-1, bounds_error=False, kind='linear', fill_value=0)(newTimes))
                
                #If reading all MSI data and m/z line data cannot be interpolated in set time, then try to initiate a fallback method to do so sequentially later
                #Note that if system is sufficiently memory bottlenecked for this to happen, this method may crash from OOM in transmission to shared memory actor anyways
                if overwriteAllChanFiles and readAllMSI: 
                    timeOutCounter, allDataInterpFail = 0, False
                    while True:
                        try: 
                            mzDataLine = interp1d(origTimes, mzDataLine, axis=-1, bounds_error=False, kind='linear', fill_value=0)(newTimes)
                            break
                        except: 
                            if timeOutCounter==10: print('\nWarning - Interpolation of m/z line data in parallel failed, due to memory limit. This process will retry, but if this warning occurs repeatedly, better perfomance might be achieved by decreasing the number of threads, or disabling parallelization.')
                            time.sleep(1)
                            timeOutCounter += 1
                            if timeOutCounter == 20: 
                                print('\nWarning - Interpolation of m/z line data in parallel failed multiple times due to memory limit, initiating fallback for file import. If this warning occurs repeatedly, better perfomance might be achieved by decreasing the number of threads, or disabling parallelization.')
                                allDataInterpFail = True
                    allDataInterpFailTotal.append(allDataInterpFail)
                    mzDataTotal.append(mzDataLine)
        
        #If there was new data, transfer information to shared memory actor
        if len(readScanFiles) > 0: 
            _ = ray.get(allImagesActor.setValues.remote(indexData, mzDataTotal, chanDataTotal, sumDataTotal, origTimesTotal, np.array(lineNumTotal), readScanFiles, allDataInterpFailTotal))

#If version > 0.10.1, delete this comment block; left for documentation 
#================================================================================================================================
#Tracer()
#Parallel visualization has a race condition producing an occasional exception
#matplotlib\cbook\__init__.py -> _remove_proxy -> del self.callbacks[signal][cid] -> KeyError: 'changed'
#Exception ignored -> Python311\lib\weakref.py -> in _cb
#Tried temporarily redirecting output to prevent spurious output(s), but message still appears in console...
#Think this results from matplotlib not being threadsafe
#Might now be fixed having removed plt.close() statements
#================================================================================================================================

#Visualize sampling steps in parallel
@ray.remote
def visualizeStep_parhelper(samples, indexes, sampleData, dir_progression, dir_chanProgressions, samplingProgress_Actor):
    if not debugMode: setupLogging()
    for index in indexes: 
        visualizeStep(samples[index], sampleData, dir_progression, dir_chanProgressions)
        _ = ray.get(samplingProgress_Actor.update.remote(1))
