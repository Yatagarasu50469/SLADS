#==================================================================
#REMOTE METHODS FOR RAY
#==================================================================

#Define deployment for trained models
@serve.deployment(route_prefix="/ModelServer", ray_actor_options={"num_gpus": numGPUs})
class ModelServer:
    def __init__(self, erdModel, modelPath):
        warnings.filterwarnings("ignore")
        loggerServe.setLevel(logging.ERROR)
        self.erdModel = erdModel
        if self.erdModel == 'SLADS-LS' or self.erdModel == 'SLADS-Net': self.model = np.load(modelPath+'.npy', allow_pickle=True).item()
        elif self.erdModel == 'DLADS' or self.erdModel == 'GLANDS': self.model = tf.function(tf.keras.models.load_model(modelPath, compile=False), experimental_relax_shapes=True)

    def __call__(self, data):
        if self.erdModel == 'SLADS-LS' or self.erdModel == 'SLADS-Net': return self.model.predict(data)
        elif self.erdModel == 'DLADS' or self.erdModel == 'GLANDS': return self.model(data, training=False)[:,:,:,0].numpy()

#Define actor for utilizing trained models
@ray.remote
class Model_Actor:
    def __init__(self, erdModel, modelPath):
        warnings.filterwarnings("ignore")
        loggerServe.setLevel(logging.ERROR)
        self.erdModel = erdModel
        if self.erdModel == 'SLADS-LS' or self.erdModel == 'SLADS-Net': self.model = np.load(modelPath+'.npy', allow_pickle=True).item()
        elif self.erdModel == 'DLADS' or self.erdModel == 'GLANDS': self.model = tf.function(tf.keras.models.load_model(modelPath, compile=False), experimental_relax_shapes=True)
    def generateERD(self, data):
        if self.erdModel == 'SLADS-LS' or self.erdModel == 'SLADS-Net': return self.model.predict(data)
        elif self.erdModel == 'DLADS' or self.erdModel == 'GLANDS': return self.model(data, training=False)[:,:,:,0].numpy()

#Ray actor for holding global progress in parallel sampling operations
@ray.remote
class SamplingProgress_Actor:
    def __init__(self): 
        warnings.filterwarnings("ignore")
        self.current = 0.0
    def update(self, amount): self.current += amount
    def getCurrent(self): return self.current

#Ray actor for computing reconstructions of all images in parallel
@ray.remote
class Recon_Actor:
    
    #Set internal parameters for handling image reconst ruction process
    def __init__(self, indexes, sampleType, squareDim, finalDim, allImagesMax):
        warnings.filterwarnings("ignore")
        self.indexes = indexes
        self.sampleType = sampleType
        self.squareDim = squareDim
        self.finalDim = finalDim
        self.allImagesMax = allImagesMax
        
    #Load in reference for all images file and setup for performing reconstructions
    def setup(self, allImagesPath, squareAllImagesPath):
        self.allImagesFile = h5py.File(allImagesPath, 'r')
        self.allImages = self.allImagesFile['allImages']
        if self.sampleType == 'DESI': 
            self.squareAllImagesFile = h5py.File(squareAllImagesPath, 'r')
            self.squareAllImages = self.squareAllImagesFile['squareAllImages']
        
    #Apply mask to the complete data to obtain sparse images (needs to be float32 after the multiplication to prevent memory issues, mask must therefore be float32!)
    def applyMask(self, mask):
        if self.sampleType == 'MALDI': self.reconImages = self.allImages[self.indexes]*mask
        elif self.sampleType == 'DESI': self.reconImages = self.squareAllImages[self.indexes]*mask
    
    #Compute reconstructions, resizing back to original dimensionality if DESI sample
    def computeRecon(self, unMeasuredIdxs, measuredIdxs, neighborIndices, neighborWeights):
        if len(self.reconImages.shape) == 3: self.reconImages[:, unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]] = [np.sum(self.reconImages[index, measuredIdxs[:,0], measuredIdxs[:,1]][neighborIndices]*neighborWeights, axis=-1) for index in range(0, len(self.indexes))]
        else: self.reconImages[unMeasuredIdxs[:,0], unMeasuredIdxs[:,1]] = np.sum(self.reconImages[measuredIdxs[:,0], measuredIdxs[:,1]][neighborIndices]*neighborWeights, axis=-1)
        if self.sampleType == 'DESI': self.reconImages = np.moveaxis(resize(np.moveaxis(self.reconImages, 0, -1), tuple(self.finalDim), order=0), -1, 0)
    
    #Return allImages already loaded; Warning: Performing any operations beyond just the return will induce significant copy overhead!
    def getAllImages(self):
        return self.allImages
    
    #Compute PSNR and SSIM between reconstructions and original/complete data
    def computeMetrics(self):
        self.imagesPSNRList, self.imagesSSIMList = [], []
        for index in range(0, len(self.indexes)):
            try:
                image = self.allImages[self.indexes[index]]
                self.imagesPSNRList.append(compare_psnr(image, self.reconImages[index], data_range=self.allImagesMax[self.indexes[index]]))
                self.imagesSSIMList.append(compare_ssim(image, self.reconImages[index], data_range=self.allImagesMax[self.indexes[index]]))
            except:
                Tracer()
        
    def getPSNR(self):
        return self.imagesPSNRList
    
    def getSSIM(self):
        return self.imagesSSIMList
    
    #Obtain reconstructions; Warning: Performing any operations beyond just the return will induce significant copy overhead!
    def getReconImages(self):
        return self.reconImages
    
    #Close any open files
    def closeAllImages(self):
        self.allImagesFile.close()
        del self.allImagesFile
        if self.sampleType == 'DESI':
            self.squareAllImagesFile.close()
            del self.squareAllImagesFile

#Ray actor for holding MSI data in a shared memory location for parallel operations
@ray.remote
class Reader_MSI_Actor:
    
    #Create buffers for holding all MSI images, the specified channel images, and the sum of all values
    def __init__(self, sampleType, mzNum, chanNum, yDim, xDim, allImagesPath, squareAllImagesPath):
        warnings.filterwarnings("ignore")
        self.sampleType = sampleType
        self.yDim = yDim
        self.xDim = xDim
        self.allImagesPath = allImagesPath
        self.squareAllImagesPath = squareAllImagesPath
        self.allImages = np.zeros((yDim, xDim, mzNum), dtype=np.float32)
        self.chanImages = np.zeros((yDim, xDim, chanNum), dtype=np.float32)
        self.sumImage = np.zeros((yDim, xDim), dtype=np.float32)
        if sampleType == 'DESI': self.mzDataComplete, self.origTimesComplete, self.lineNumComplete, self.readScanFiles = [], [], [], []
    
    #Set internal reference for indexing different coordinates
    def setCoordinates(self, coordinates):
        self.coordinates = coordinates
    
    #Assign m/z data to specified locations; for DESI, if the m/z line data could not be interpolated in parallel, then setup for sequential fallback in shared memory space
    def setValues(self, indexData, mzDataTotal, chanDataTotal, sumDataTotal, origTimesTotal=None, lineNumTotal=None, newReadScanFiles=None, allDataInterpFailTotal=None):
        if self.sampleType == 'MALDI':
            yLocations, xLocations = self.coordinates[indexData,1], self.coordinates[indexData,0]
            self.allImages[yLocations, xLocations, :] = mzDataTotal
            self.chanImages[yLocations, xLocations, :] = chanDataTotal
            self.sumImage[yLocations, xLocations] = sumDataTotal
        elif self.sampleType == 'DESI':
            self.sumImage[lineNumTotal, :] = sumDataTotal
            self.chanImages[lineNumTotal, :, :] = chanDataTotal
            for indexNum in range(0, len(indexData)):
                if allDataInterpFailTotal[indexNum]: 
                    self.mzDataComplete.append(mzDataTotal[indexNum])
                    self.origTimesComplete.append(copy.deepcopy(origTimesTotal[indexNum]))
                    self.lineNumComplete.append(lineNumTotal[indexNum])
                else: self.allImages[lineNumTotal[indexNum], :, :] = mzDataTotal[indexNum]
                self.readScanFiles.append(newReadScanFiles[indexNum])
    
    #Write data to a .hdf5 file at the prespecified location on disk and return the max value for each m/z; for DESI save square varations as well for later reconstructions
    def writeToDisk(self, squareDim):
        allImagesMax = np.max(self.allImages, axis=(0,1))
        allImagesFile = h5py.File(self.allImagesPath, 'a')
        _ = allImagesFile.create_dataset(name='allImages', data=np.moveaxis(self.allImages, -1, 0), chunks=(1, self.yDim, self.xDim), dtype=np.float32)
        allImagesFile.close()
        if self.sampleType=='DESI':
            squareAllImagesFile = h5py.File(self.squareAllImagesPath, 'a')
            _ = squareAllImagesFile.create_dataset(name='squareAllImages', data=np.moveaxis(resize(self.allImages, tuple(squareDim), order=0), -1, 0), chunks=(1, squareDim[0], squareDim[1]), dtype=np.float32)
            squareAllImagesFile.close()
        return allImagesMax
        
    #Place all images data in shared memory with a persistent reference to survive this actor
    def shareAllImages(self): 
        return ray.put(self.allImages, _owner=None)
    
    #Return all images data; Warning: Performing any operations beyond just the return will induce significant copy overhead!
    def getAllImages(self): 
        return self.allImages
    
    #Return channel images data; Warning: Performing any operations beyond just the return will induce significant copy overhead!
    def getChanImages(self): 
        return self.chanImages
    
    #Return sum image data; Warning: Performing any operations beyond just the return will induce significant copy overhead!
    def getSumImage(self): 
        return self.sumImage
    
    #Return a list of the read scan files for DESI MSI
    def getReadScanFiles(self):
        return self.readScanFiles
    
    #Align missing DESI m/z line data to a regular grid and store in shared memory space
    def interpolateDESI(self, mzFinal, mzFinalGrid):
        mzFinal = copy.deepcopy(mzFinal)
        for index in range(0, len(self.lineNumComplete)): self.allImages[self.lineNumComplete[index], :, :] = scipy.interpolate.RegularGridInterpolator((self.origTimesComplete[index], mzFinal), np.asarray(self.mzDataComplete[index], dtype='float64'), bounds_error=False, fill_value=0)(mzFinalGrid).astype('float32').T
    
#Read in the sample MSI data for a set of indexes and set those values in shared memory location; must use blocking call (ray.get) to prevent data corruption
@ray.remote
def msi_parhelper(allImagesActor, scanFileNames, indexData, mzOriginalIndices, mzRanges, sampleType, mzLowerBound, mzUpperBound, mzLowerIndex, mzPrecision, mzRound, mzInitialCount, mask, newTimes, finalDim, sampleWidth, scanRate, mzFinal=None, mzFinalGrid=None, chanValues=None, chanFinalGrid=None, impModel=None, postModel=None, impOffset=None, scanMethod=None, lineMethod=None, physicalLineNums=None, ignoreMissingLines=None, missingLines=None, unorderedNames=None):
    warnings.filterwarnings("ignore")
    mzDataTotal, chanDataTotal, sumDataTotal = [], [], []
    if sampleType == 'MALDI':
    
        #Load the sinlge imzML file expected for MALDI
        data = ImzMLParser(scanFileNames[0]) 
        
        #Process each of the files assigned to this helper and transmit to shared memory actor
        for index in indexData:
            mzs, ints = data.getspectrum(index)
            sumDataTotal.append(np.sum(ints))
            filtIndexLow, filtIndexHigh = bisect_left(mzs, mzLowerBound), bisect_right(mzs, mzUpperBound)
            mzDataTotal.append(np.add.reduceat(mzFastIndex(mzs[filtIndexLow:filtIndexHigh], ints[filtIndexLow:filtIndexHigh], mzLowerIndex, mzPrecision, mzRound, mzInitialCount), mzOriginalIndices))
            chanDataTotal.append(np.asarray([np.sum(ints[bisect_left(mzs, mzRange[0]):bisect_right(mzs, mzRange[1])]) for mzRange in mzRanges]))      
        _ = ray.get(allImagesActor.setValues.remote(indexData, mzDataTotal, chanDataTotal, sumDataTotal))

        #Close the MSI file
        del data
    
    elif sampleType == 'DESI':
        
        #Duplicate input in this space so as to be writable for regular grid interpolation
        chanValues = copy.deepcopy(chanValues) 
        
        #Keep for trying all mz interp in parallel
        mzFinal = copy.deepcopy(mzFinal)
        
        #Process each of the files assigned to this helper
        origTimesTotal, newReadScanFiles, lineNumTotal, allDataInterpFailTotal = [], [], [], []
        for index in indexData:
            
            #Extract the filename to be processed
            scanFileName = scanFileNames[index]
        
            #Load the line data and flag errors during the process (primarily checking for files without data)
            errorFlag = False
            try: data = mzFile(scanFileName)
            except: errorFlag = True
            
            #Extract the file number and if unordered find corresponding line number in LUT, otherwise line number is the file number
            if not errorFlag:
                fileNum = int(scanFileName.split('line-')[1].split('.')[0].lstrip('0'))-1
                if unorderedNames: 
                    try: lineNum = physicalLineNums[fileNum+1]
                    except: errorFlag = True #print('\nWarning - Attempt to find the physical line number for the file: ' + scanFileName + ' has failed; the file will therefore be ignored this iteration.')
                else: lineNum = fileNum
            
            #If error still has not occurred 
            if not errorFlag:
            
                #If ignoring missing lines, then determine the offset for correct indexing
                if ignoreMissingLines and len(missingLines) > 0: lineNum -= int(np.sum(lineNum > missingLines))
                    
                #Add file name to those that will have been already scanned (when this process finishes)
                newReadScanFiles.append(scanFileName)
            
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
                
                #Process data for each measured location
                chanDataLine, mzDataLine = [[] for _ in range(0, len(mzRanges))], []
                
                #If the data is being sparesly acquired in lines, then the listed times in the file need to be shifted
                if (impModel or postModel) and impOffset and scanMethod == 'linewise' and (lineMethod == 'segLine' or lineMethod == 'fullLine'): origTimes += (np.argwhere(mask[lineNum]==1).min()/finalDim[1])*(((sampleWidth*1e3)/scanRate)/60)
                elif (impModel or postModel) and impOffset: sys.exit('Error - Using implementation or post-process modes with an offset but not segmented-linewise operation is not currently a supported configuration.')
            
                #Seup storage locations for each measured location
                chanDataLine, mzDataLine = [[] for _ in range(0, len(mzRanges))], []
                
                #Set positions to be scanned for the line
                if data.format == 'Bruker': positions = range(1, len(origTimes)+1)
                else: positions = range(data.scan_range()[0], data.scan_range()[1]+1)
                
                #Read in and process spectrum data for each position, storing for later analysis
                for pos in positions:
                    if data.format == 'Bruker':
                        frameData = np.asarray(data.frame(pos))
                        mzs, ints = frameData[:,0], frameData[:,2]
                        sumImageLine.append(np.sum(ints))
                    else: 
                        scanData = np.array(data.scan(pos, 'profile'))
                        mzs, ints = scanData[:,0], scanData[:,1]
                    filtIndexLow, filtIndexHigh = bisect_left(mzs, mzLowerBound), bisect_right(mzs, mzUpperBound)
                    mzDataLine.append(np.add.reduceat(mzFastIndex(mzs[filtIndexLow:filtIndexHigh], ints[filtIndexLow:filtIndexHigh], mzLowerIndex, mzPrecision, mzRound, mzInitialCount), mzOriginalIndices))
                    for mzRangeNum in range(0, len(mzRanges)):
                        mzRange = mzRanges[mzRangeNum]
                        chanDataLine[mzRangeNum].append(np.sum(ints[bisect_left(mzs, mzRange[0]):bisect_right(mzs, mzRange[1])]))
                
                #Store obtained line data interpolating where possible
                origTimesTotal.append(origTimes)
                lineNumTotal.append(lineNum)
                chanDataTotal.append(scipy.interpolate.RegularGridInterpolator((origTimes, chanValues), np.asarray(chanDataLine, dtype='float64').T, bounds_error=False, fill_value=0)(chanFinalGrid).astype('float32').T)
                sumDataTotal.append(np.interp(newTimes, origTimes, np.nan_to_num(sumImageLine, nan=0, posinf=0, neginf=0), left=0, right=0))
                
                #If mz line data cannot be interpolated in set time, then try to initiate a fallback method to do so sequentially later
                #Note that if system is sufficiently memory bottlenecked for this to happen, this method may crash from OOM in transmission to shared memory actor anyways
                timeOutCounter, allDataInterpFail = 0, False
                while True:
                    try: 
                        mzDataLine = scipy.interpolate.RegularGridInterpolator((origTimes, mzFinal), np.asarray(mzDataLine, dtype='float64'), bounds_error=False, fill_value=0)(mzFinalGrid).astype('float32').T
                        break
                    except: 
                        if timeOutCounter==10: print('Warning - Interpolation of m/z line data in parallel failed, due to memory limit. This process will retry, but if this warning occurs repeatedly, better perfomance might be achieved by increasing the number of reserved threads, or disabling parallelization.')
                        time.sleep(1)
                        timeOutCounter += 1
                        if timeOutCounter == 20: 
                            print('Warning - Interpolation of m/z line data in parallel failed multiple times due to memory limit, initiating fallback for file import. If this warning occurs repeatedly, better perfomance might be achieved by increasing the number of reserved threads, or disabling parallelization.')
                            allDataInterpFail = True
                allDataInterpFailTotal.append(allDataInterpFail)
                mzDataTotal.append(mzDataLine)
        
            #Close the mzFile
            data.close()
        
        #If there was new data read, transfer such to shared memory actor
        if len(newReadScanFiles) > 0: _ = ray.get(allImagesActor.setValues.remote(indexData, mzDataTotal, chanDataTotal, sumDataTotal, origTimesTotal, np.array(lineNumTotal), newReadScanFiles, allDataInterpFailTotal))
        