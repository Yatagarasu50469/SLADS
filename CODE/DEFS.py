#==================================================================
#SLADS DEFINITIONS GENERAL
#==================================================================

#Determine measured and unmeasured locations given a binary mask
@jit(nopython=True)
def indexMask(mask):
    return np.transpose(np.vstack(np.where(mask==1))), np.transpose(np.vstack(np.where(mask==0)))

#Iterate object ids in ray, so as to provide progress feedback
def rayIterator(ids):
        while ids:
            done, ids = ray.wait(ids)
            yield ray.get(done[0])

#Perform summation and alignment over an mz range for a given line; mzFile cannot be pickled and passed into function
@ray.remote
def mzrange_parhelper(mzRange, scanFileName, mzMethod, ignoreMissingLines, missingLines):
    return mzrange_serial(mzRange, scanFileName, mzMethod, ignoreMissingLines, missingLines)

def mzrange_serial(mzRange, scanFileName, mzMethod, ignoreMissingLines, missingLines):
    data = mzFile(scanFileName)
    lineNum = int(scanFileName.split('line')[1].split('.')[0])-1
    
    #If ignoring missing lines, then determine the offset for correct indexing
    if ignoreMissingLines: lineNum -= int(np.sum([lineNum > missingLine for missingLine in missingLines]))
    
    if mzMethod == 'sum':
        lineData = np.asarray([np.sum(mz_range(data.scan(px, 'profile'), mzRange)) for px in range(data.scan_range()[0], data.scan_range()[1]+1)])
    elif mzMethod =='xic':
        lineData = np.asarray(list(map(list, data.xic(data.time_range()[0], data.time_range()[1], mzRange[0], mzRange[1]))))[:,1]
    else:
        sys.exit('Error! - mzMethod: ' + mzMethod + 'for: ' + scanFileName + 'has not been implemented')
 
    return lineNum, lineData

#Visualize single sample progression step
@ray.remote
def visualize_parhelper(sample, simulationFlag, dir_avgProgression, dir_mzProgressions):
    return visualize_serial(sample, simulationFlag, dir_avgProgression, dir_mzProgressions)
    
def visualize_serial(sample, simulationFlag, dir_avgProgression, dir_mzProgressions):
    #Re-import libraries inside of thread to set plotting backend as non-interactive
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    #Turn percent measured into a string
    percMeasured = "{:.2f}".format(sample.percMeasured)
    
    #Turn metrics into strings
    if simulationFlag: avgmzImagePSNR = "{:.2f}".format(sample.avgmzImagePSNR)
    if simulationFlag: erdPSNR = "{:.2f}".format(sample.ERDPSNR)

    #For each of the mz ranges, generate visuals
    for mzNum in range(0, len(sample.mzRanges)):
        
        mzMinValue, mzMaxValue = np.min(sample.mzImages[mzNum]), np.max(sample.mzImages[mzNum])
        
        #Turn metrics into strings
        massRange = str(sample.mzRanges[mzNum][0]) + '-' + str(sample.mzRanges[mzNum][1])
        if simulationFlag: mzImagePSNR = "{:.2f}".format(sample.mzImagePSNRList[mzNum])
        if simulationFlag: avgImagePSNR = "{:.2f}".format(sample.avgImagePSNR)
        
        #Measured mz image
        f = plt.figure(figsize=(20,5.3865))
        if simulationFlag: plt.suptitle(r"$\bf{Sample:\ }$" + sample.name + r"$\bf{\ \ mz:\ }$" + massRange + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured + '\n' + r"$\bf{PSNR - mz\ Recon:\ }$" + mzImagePSNR + r"$\bf{\ \ Average\ mz\ Recon:\ }$" + avgmzImagePSNR)
        else: plt.suptitle(r"$\bf{Sample:\ }$" + sample.name + r"$\bf{\ \ mz:\ }$" + massRange + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured)

        if simulationFlag: ax = plt.subplot2grid(shape=(1,3), loc=(0,0))
        else: ax = plt.subplot2grid(shape=(1,3), loc=(0,0))
        im = ax.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Sampled Mask')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

        if simulationFlag:
            ax = plt.subplot2grid(shape=(1,3), loc=(0,1))
            im = ax.imshow(sample.mzImages[mzNum], cmap='hot', aspect='auto', vmin=mzMinValue, vmax=mzMaxValue)
            ax.set_title('Ground-Truth')
            cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        if simulationFlag: ax = plt.subplot2grid(shape=(1,3), loc=(0,2))
        else: ax = plt.subplot2grid(shape=(1,3), loc=(0,1))
        im = ax.imshow(sample.mzReconImages[mzNum], cmap='hot', aspect='auto', vmin=mzMinValue, vmax=mzMaxValue)
        ax.set_title('Reconstruction')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        
        #Save
        f.tight_layout()
        saveLocation = dir_mzProgressions[mzNum] + 'progression_mz_' + massRange + '_perc_' + str(sample.percMeasured) +'.png'
        plt.savefig(saveLocation, bbox_inches='tight')
        plt.close()
        
        #Do borderless saves for each mz image here; mask will be the same as produced in the average output
        saveLocation = dir_mzProgressions[mzNum] + 'reconstruction_mz_' + massRange + '_perc_' + str(sample.percMeasured) + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(sample.mzReconImages[mzNum], cmap='hot', aspect='auto', vmin=mzMinValue, vmax=mzMaxValue)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()
    
    #For the average, generate visual
    avgMinValue, avgMaxValue = np.min(sample.avgGroundTruthImage), np.max(sample.avgGroundTruthImage)
    f = plt.figure(figsize=(20,10))
    if simulationFlag: plt.suptitle(r"$\bf{Sample:\ }$" + sample.name + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured + '\n' + r"$\bf{PSNR - Average\ Recon: }$" + avgImagePSNR + r"$\bf{\ \ Average\ mz\ Recon:\ }$" + avgmzImagePSNR + r"$\bf{\ \ ERD:\ }$" + erdPSNR)
    else:  plt.suptitle(r"$\bf{Sample:\ }$" + sample.name + r"$\bf{\ \ Percent\ Sampled:\ }$" + percMeasured)
    
    if simulationFlag: 
        ax = plt.subplot2grid(shape=(2,3), loc=(0,0))
        im = ax.imshow(sample.avgGroundTruthImage, cmap='hot', aspect='auto', vmin=avgMinValue, vmax=avgMaxValue)
        ax.set_title('Ground-Truth')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if simulationFlag: ax = plt.subplot2grid((2,3), (0,1))
    else: ax = plt.subplot2grid((1,3), (0,0))
    im = ax.imshow(sample.avgReconImage, cmap='hot', aspect='auto', vmin=avgMinValue, vmax=avgMaxValue)
    ax.set_title('Reconstruction')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if simulationFlag: 
        ax = plt.subplot2grid((2,3), (0,2))
        im = ax.imshow(abs(sample.avgGroundTruthImage-sample.avgReconImage), cmap='hot', aspect='auto', vmin=avgMinValue, vmax=avgMaxValue)
        ax.set_title('Absolute Difference')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if simulationFlag: ax = plt.subplot2grid((2,3), (1,0))
    else: ax = plt.subplot2grid((1,3), (0,1))
    im = ax.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Measurement Mask')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    if simulationFlag: ax = plt.subplot2grid((2,3), (1,1))
    else: ax = plt.subplot2grid((1,3), (0,2))
    im = ax.imshow(sample.ERD, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_title('ERD')
    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    if simulationFlag: 
        ax = plt.subplot2grid((2,3), (1,2))
        im = ax.imshow(sample.RDImage, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_title('RD')
        cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    
    #Save
    f.tight_layout()
    saveLocation = dir_avgProgression + 'progression_perc_' + str(sample.percMeasured) + '_avg.png'
    plt.savefig(saveLocation, bbox_inches='tight')
    plt.close()

    #Borderless saves
    saveLocation = dir_avgProgression + 'reconstruction_perc_' + str(sample.percMeasured) + '.png'
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(sample.avgReconImage, cmap='hot', aspect='auto', vmin=avgMinValue, vmax=avgMaxValue)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(saveLocation, bbox_inches=extent)
    plt.close()
    
    saveLocation = dir_avgProgression + 'mask_perc_' + str(sample.percMeasured) + '.png'
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(sample.mask, cmap='gray', aspect='auto', vmin=0, vmax=1)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(saveLocation, bbox_inches=extent)
    plt.close()
    
    saveLocation = dir_avgProgression + 'ERD_perc_' + str(sample.percMeasured) + '.png'
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(sample.ERD, aspect='auto', vmin=0, vmax=1)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(saveLocation, bbox_inches=extent)
    plt.close()
    
    saveLocation = dir_avgProgression + 'measured_perc_' + str(sample.percMeasured) + '.png'
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.imshow(sample.avgReconImage*sample.mask, cmap='hot', aspect='auto', vmin=avgMinValue, vmax=avgMaxValue)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(saveLocation, bbox_inches=extent)
    plt.close()
    
    if simulationFlag:
        saveLocation = dir_avgProgression + 'RD_perc_' + str(sample.percMeasured) + '.png'
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(sample.RDImage, aspect='auto', vmin=0, vmax=1)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(saveLocation, bbox_inches=extent)
        plt.close()

#All information pertaining to a sample
class Sample:
    def __init__(self, sampleFolder, initialPercToScan, scanMethod, ignoreMissingLines=False):
    
        #Should missing lines be ignored (to only be used in training and testing)
        self.ignoreMissingLines = ignoreMissingLines
    
        #Location of MSI data and sample name
        self.sampleFolder = sampleFolder
        self.name = os.path.basename(sampleFolder)
    
        #Which files have already been read
        self.readScanFiles = []
        self.readLines = []

        #Read in data from sampleInfo.txt
        lineIndex = 0
        sampleInfo = open(self.sampleFolder+os.path.sep+'sampleInfo.txt').readlines()
        
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
            mzLocations = np.loadtxt(self.sampleFolder+os.path.sep+'mz.csv', delimiter=',')
            self.mzRanges = [[mzLocation-self.mzTolerance, mzLocation+self.mzTolerance] for mzLocation in mzLocations]
        elif self.mzSpec == 'range':
            self.mzRanges = np.loadtxt(self.sampleFolder+os.path.sep+'mz.csv', delimiter=',')
        
        #Setup averages of the mz ranges for visualizations
        self.mzAverageRanges = np.average(self.mzRanges, axis=1)
        
        #Prepare variables to hold sample data
        self.mzWeights = np.ones(len(self.mzRanges))/len(self.mzRanges)

        #If using a standard, then also read in standards locations (.csv)
        if self.normMethod == 'standard': 
            mzStandardLocations = np.loadtxt(self.sampleFolder+os.path.sep+'mzStandards.csv', delimiter=',')
            if mzStandardLocations.shape != ():
                print('Warning! - Untested functionality; Will sum the formed visualizations of the specified m/z ranges for the normalization')
                self.mzStandardRanges = [[mzStandardLocation-self.mzTolerance, mzStandardLocation+self.mzTolerance] for mzStandardLocation in mzStandardLocations]
            else:
                self.mzStandardRanges = [[mzStandardLocations-self.mzTolerance, mzStandardLocations+self.mzTolerance]]
        
        #Variables used in training/testing implementation
        self.resultsPath = None
        self.percMeasured = 0
        self.polyFeatures = None
        self.avgGroundTruthImage = None
        self.normArray = None
        self.avgMeasuredImage = None
        self.squareRDValues = None
        self.origNormArray = None
        self.RDImage = None
        self.mzReconImages = []
        self.mzImagePSNRList = None
        self.avgmzImagePSNR = None
        
        #Get the MSI file extension automatically      
        extensions = list(map(lambda x:x.lower(), np.unique([filename.split('.')[-1] for filename in natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'), reverse=False)])))
        if 'd' in extensions: self.lineExt = '.d'
        elif 'raw' in extensions: self.lineExt = '.raw'
        else: sys.exit('Error! - Unknown MSI filetype being used for sample: ' + self.name)
        
        #Store final dimensions for physical domain, determining the number of columns for row-alignment interpolations
        self.finalDim = [numLines, int(round((self.sampleWidth*1e3)/self.scanRate))]
        
        #If missing lines are to be ignored, then setup to do so (only possible to perform in simulated operation)
        if self.ignoreMissingLines: 
            scanFiles = natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'+self.lineExt), reverse=False)
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
        
        #Initialize map for most important mz found for RDPP
        self.mzRDPPMap = np.zeros(self.squareDim)
        self.mzMap = np.empty(self.finalDim)
        self.mzMap[:] = np.NaN
        
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
            lineIndexes = [int((self.finalDim[0]-1)*0.50)]
            
            #Obtain points in the specified lines and add them to the initial scan list
            for lineIndex in lineIndexes:
                
                #If only a percentage should be scanned, then randomly select points, otherwise select all
                if lineMethod == 'percLine':
                    newIdxs = copy.deepcopy(self.linesToScan[lineIndex])
                    np.random.shuffle(newIdxs)
                    newIdxs = newIdxs[:int((stopPerc/100)*self.finalDim[1])]
                else: 
                    newIdxs = [pt for pt in self.linesToScan[lineIndex]]
                
                #Add positions to initial list
                self.initialSets.append(newIdxs)
                
                #Add the points to the initial mask
                for pt in [tuple(pt) for pt in newIdxs]: self.initialMask[pt] = 1
                
            #Delete the lines/points specified from remaining potentials; in case of overlapping 'line' geometries
            for lineIndexNum in range(0, len(lineIndexes)): 
                if lineRevistMethod:
                    self.delPoints(newIdxs)
                else:
                    self.delLine(lineIndexes[lineIndexNum]-lineIndexNum)
            
        elif scanMethod == 'pointwise':
        
            #Randomly select points to initially scan
            newIdxs = np.transpose(np.where(self.initialMask==0))
            np.random.shuffle(newIdxs)
            newIdxs = newIdxs[:int((initialPercToScan/100)*self.area)]
            
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
        
        #Find neighbor information if neccessary
        #self.neighborIndices, self.neighborWeights, self.neighborDistances = findNeighbors(self.squareMeasuredIdxs, self.squareUnMeasuredIdxs)
        
    #Update mzImages, TIC, origTIC, and origTimes by information in the present line files
    def readScanData(self, lineRevistMethod):

        #Obtain and sort the available line files pertaining to the current scan
        scanFiles = natsort.natsorted(glob.glob(self.sampleFolder+os.path.sep+'*'+self.lineExt), reverse=False)

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
            if self.ignoreMissingLines: lineNum -= int(np.sum([lineNum > missingLine for missingLine in self.missingLines]))
            
            #Obtain the total ion chromatogram over all mz available
            xicData = np.asarray(data.xic(data.time_range()[0], data.time_range()[1]))
            
            #Save for future reference
            self.origTimes[lineNum] = xicData[:,0]
            self.origTIC[lineNum] = xicData[:,1]
        
        #Check and setup for normalization
        if self.normMethod == 'tic':
            self.origNormArray = self.origTIC
        elif self.normMethod == 'standard':
            for mzRangeNum in range(0, len(self.mzStandardRanges)): 
                if parallelization:
                    results = ray.get([mzrange_parhelper.remote(self.mzStandardRanges[mzRangeNum], scanFileName, self.mzMethod, self.ignoreMissingLines, self.missingLines) for scanFileName in scanFiles])
                else:
                    results = [mzrange_serial(self.mzStandardRanges[mzRangeNum], scanFileName, self.mzMethod, self.ignoreMissingLines, self.missingLines) for scanFileName in scanFiles]
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
            if parallelization:
                results = ray.get([mzrange_parhelper.remote(self.mzRanges[mzRangeNum], scanFileName, self.mzMethod, self.ignoreMissingLines, self.missingLines) for scanFileName in scanFiles])
            else: 
                results = [mzrange_serial(self.mzRanges[mzRangeNum], scanFileName, self.mzMethod, self.ignoreMissingLines, self.missingLines) for scanFileName in scanFiles]
            for result in results: 
                if self.normMethod != 'none': 
                    self.mzImages[mzRangeNum][result[0]] = np.interp(self.newTimes, self.origTimes[result[0]], 
                    np.nan_to_num(result[1]/self.origNormArray[result[0]], nan=0, posinf=0, neginf=0))
                else:
                    self.mzImages[mzRangeNum][result[0]] = np.interp(self.newTimes, self.origTimes[result[0]], result[1])
        
        #Determine ground-truth average
        self.avgGroundTruthImage = np.average(np.asarray(self.mzImages), axis=0, weights=self.mzWeights)
        self.avgSquareGroundTruthImage = resize(self.avgGroundTruthImage, (self.squareDim[0], self.squareDim[1]), order=0)
        
        #Create corresponding square variable to mzImages
        self.squaremzImages = np.asarray([resize(mzImage, (self.squareDim[0], self.squareDim[1]), order=0) for mzImage in self.mzImages])
        
    #Scan new locations in the sample
    def performMeasurements(self, newIdxs, percToScan, simulationFlag, fromRecon, bestCFlag, neighborCalcFlag):
    
        #Ensure newIdxs are indexible in 2 dimensions
        newIdxs = np.atleast_2d(newIdxs)
        
        #Update mask and indexes
        #t0 = time.time()
        for pt in newIdxs: self.mask[tuple(pt)] = 1
        self.measuredIdxs, self.unMeasuredIdxs = indexMask(self.mask)
        self.squareMask = resize(self.mask, (self.squareDim[0], self.squareDim[1]), order=0)
        self.squareMeasuredIdxs, self.squareUnMeasuredIdxs = indexMask(self.squareMask)
        #print('     Index and Mask: ' + str(time.time()-t0))
        
        #If the measurements are not to be taken from the reconstruction image(s) (as is done for groupwise percToScan acquisition)
        if not fromRecon:
            #If this is not a simulation then inform equipment, wait/read/update; otherwise used stored information
            if not simulationFlag:
                with open(dir_ImpResults + 'UNLOCK', 'w') as filehandle: filehandle.writelines(str(sample.initialSets[setNum][0]) + ', ' + str(sample.initialSets[setNum][1]))
                equipWait()
                self.readScanData()
                self.measuredmzImages = np.asarray(self.mzImages)
            else:
                self.measuredmzImages[:, newIdxs[:,0], newIdxs[:,1]] = np.asarray(self.mzImages)[:, newIdxs[:,0], newIdxs[:,1]]
                
            #t0 = time.time()
            #Update the square mz images
            self.squareMeasuredmzImages = resize(self.measuredmzImages, (self.measuredmzImages.shape[0], self.squareDim[0], self.squareDim[1]), order=0)
            #print('     Update Square: ' + str(time.time()-t0))
            
            #t0 = time.time()
            #Perform averaging of the multiple channels
            self.avgMeasuredImage = np.average(np.asarray(self.measuredmzImages), axis=0, weights=self.mzWeights)
            #print('     Average Image: ' + str(time.time()-t0))
            
            #If not groupwise, then update the mzMap here, otherwise updated as indexes are chosen in groupwise
            if percToScan == None and bestCFlag and len(self.mzReconImages)>0: 
                self.mzRDPPMap = np.argmax(abs(self.mzImages-self.mzReconImages), axis=0).astype(int)
                self.mzMap[newIdxs[:,0], newIdxs[:,1]] = self.mzAverageRanges[self.mzRDPPMap[newIdxs[:,0], newIdxs[:,1]]]

        else: #When fromRecon (groupwise), only a single value is presented at a time in a list, set measured as recon value
            
            #If groupwise, the mzMap should updated as indexes are chosen, not when actual measurements are made
            if percToScan != None and bestCFlag and len(self.mzReconImages)>0: 
                self.mzRDPPMap = np.argmax(abs(self.mzImages-self.mzReconImages), axis=0).astype(int)
                self.mzMap[newIdxs[:,0], newIdxs[:,1]] = self.mzAverageRanges[self.mzRDPPMap[newIdxs[:,0], newIdxs[:,1]]]
            
            #If reliant on the average image then set that, otherwise set for all of the mz images
            if averageReconInput or erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net': 
                self.avgMeasuredImage[newIdxs[:,0], newIdxs[:,1]] = self.avgReconImage[newIdxs[:,0], newIdxs[:,1]]
            else:
                self.measuredmzImages[:, newIdxs[:,0], newIdxs[:,1]] = self.mzReconImages[:, newIdxs[:,0], newIdxs[:,1]]
                self.squareMeasuredmzImages = resize(self.measuredmzImages, (self.measuredmzImages.shape[0], self.squareDim[0], self.squareDim[1]), order=0)
                self.avgMeasuredImage = np.average(np.asarray(self.measuredmzImages), axis=0, weights=self.mzWeights)
        
        #Update unmeasured locations' neighbor information if neccessary
        if neighborCalcFlag: self.neighborIndices, self.neighborWeights, self.neighborDistances = findNeighbors(self.squareMeasuredIdxs, self.squareUnMeasuredIdxs)

        self.avgSquareMeasuredImage = resize(self.avgMeasuredImage, (self.squareDim), order=0)
        
        #Update percentage pixels measured; only when not fromRecon
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
    def __init__(self, sample, avgGroundTruthImage, cValue, bestCFlag, oracleFlag, simulationFlag, animationFlag, neighborCalcFlag):
        self.sample = copy.deepcopy(sample)
        self.avgGroundTruthImage = copy.deepcopy(avgGroundTruthImage)
        self.cValue = cValue
        self.simulationFlag = copy.deepcopy(simulationFlag)
        self.animationFlag = copy.deepcopy(animationFlag)
        self.bestCFlag = copy.deepcopy(bestCFlag)
        self.oracleFlag = copy.deepcopy(oracleFlag)
        self.neighborCalcFlag = copy.deepcopy(neighborCalcFlag)
        self.samples = []
        self.mzMap = None
        self.percMeasuredList = []
        self.avgmzImagePSNRList = []
    
    #Save the model development
    def update(self, sample, completedRunFlag):
    
        #If optimizing c, then store mzMap at last step, and save average PSNR of the mz reconstructions, otherwise save duplicate of sample at this step
        if self.bestCFlag:
            if completedRunFlag: self.mzMap = copy.deepcopy(sample.mzMap)
            self.avgmzImagePSNRList.append(np.average([compare_psnr(sample.mzImages[index], sample.mzReconImages[index], data_range=1) for index in range(0, len(sample.mzReconImages))]))
        else:
            self.samples.append(copy.deepcopy(sample))
        
        #Update list with percentage measured this step
        self.percMeasuredList.append(copy.deepcopy(sample.percMeasured))
        
    #Generate visualiations/metrics as needed at the end of scanning
    def complete(self, optimalC): 
        
        #If a simulation, then can do 
        if self.simulationFlag:
            
            #If neighbor information was not calculated live, then calculate it now for computation of comparative RD
            if not self.neighborCalcFlag: 
                for sample in tqdm(self.samples, desc='Neighbor Info', leave=False, ascii=True): 
                    sample.neighborIndices, sample.neighborWeights, sample.neighborDistances = findNeighbors(sample.squareMeasuredIdxs, sample.squareUnMeasuredIdxs)
            
            #Some pre-computation of values for normalization of mz Images/reconstructions
            minMzImageValue, maxMzImageValue = np.min(self.samples[-1].mzImages), np.max(self.samples[-1].mzImages)
            diffMzImageValue = maxMzImageValue-minMzImageValue
        
        #Compute/compare reconstructions for all mz ranges at each step of the progression
        for sample in self.samples:
            if len(sample.mzReconImages) == 0:
                if parallelization:
                    results = list(chain.from_iterable(ray.get([computeRecon_parhelper.remote(sample.squareMeasuredmzImages, sample, indexes) for indexes in np.array_split(np.arange(0, len(sample.squareMeasuredmzImages)), multiprocessing.cpu_count())])))
                else:
                    results = [computeRecon(squareMeasuredmzImage, sample) for squareMeasuredmzImage in sample.squareMeasuredmzImages]
                sample.squaremzReconImages = np.asarray(results)
                sample.mzReconImages = np.asarray([resize(result, tuple(self.samples[-1].finalDim), order=0) for result in results])
            
            #If a simulation, then normalize by the ground truth values and evaluate reconstructions
            if self.simulationFlag: 
                sample.mzImages = (sample.mzImages-minMzImageValue)/diffMzImageValue
                sample.mzReconImages = (sample.mzReconImages-minMzImageValue)/diffMzImageValue
                sample.mzImagePSNRList = [compare_psnr(sample.mzImages[index], sample.mzReconImages[index], data_range=1) for index in range(0, len(sample.mzReconImages))]
                sample.avgmzImagePSNR = np.average(sample.mzImagePSNRList)
        
        #If a simulation, then can do some pre-computation of values for normalization of average images/reconstructions
        if self.simulationFlag: 
            minAvgGroundTruthValue, maxAvgGroundTruthValue = np.min(self.samples[-1].avgGroundTruthImage), np.max(self.samples[-1].avgGroundTruthImage)
            diffAvgReconImageValue = maxAvgGroundTruthValue-minAvgGroundTruthValue
        
        #Compute/compare average reconstructions for all of the percentages
        results = [computeRecon(sample.avgSquareMeasuredImage, sample) for sample in self.samples]
        
        #Resize average reconstructions, and if a simulation, then normalize by ground truth values and evaluate reconstructions
        for resultNum in range(0, len(results)): 
            self.samples[resultNum].avgReconImage = resize(results[resultNum], tuple(sample.finalDim), order=0)
            
            #If a simulation, then normalize by the ground truth values and evaluate reconstructions
            if self.simulationFlag: 
                self.samples[resultNum].avgReconImage = (self.samples[resultNum].avgReconImage-minAvgGroundTruthValue)/diffAvgReconImageValue
                self.samples[resultNum].avgGroundTruthImage = (self.samples[resultNum].avgGroundTruthImage-minAvgGroundTruthValue)/diffAvgReconImageValue
                self.samples[resultNum].avgImagePSNR = compare_psnr(self.samples[resultNum].avgGroundTruthImage, self.samples[resultNum].avgReconImage, data_range=1)
            
            #Legacy SLADS(-Net) measured; for optimize c
            #if self.simulationFlag: self.samples[resultNum].avgImageTD = np.sum(computeDifference(self.samples[resultNum].avgGroundTruthImage, self.samples[resultNum].avgReconImage))/self.samples[resultNum].maskObject.area

        #Calculate the actual RD Image for each percent; bestCFlag data should be returned before this
        for sample in tqdm(self.samples, desc='RD Calc', leave = False, ascii=True):
            sample.RDImage = computeRD(sample, optimalC, True, self.bestCFlag)
            sample.ERDPSNR = compare_psnr(sample.RDImage, sample.ERD, data_range=1)
        
        #For testing printout
        self.mzAvgPSNRList = [sample.avgmzImagePSNR for sample in self.samples]
        self.avgPSNRList = [sample.avgImagePSNR for sample in self.samples]
        self.ERDPSNRList = [sample.ERDPSNR for sample in self.samples]
        
        #If an animation will be produced and the run has completed
        if self.animationFlag:

            #Setup/clean base sample directory
            dir_sampleResults = self.samples[-1].resultsPath + self.samples[-1].name + os.path.sep
            if os.path.exists(dir_sampleResults): shutil.rmtree(dir_sampleResults)
            os.makedirs(dir_sampleResults)
            
            #Prepare subdirectories; for frames and videos of mz progressions
            dir_mzProgression = dir_sampleResults + 'mz' + os.path.sep
            os.makedirs(dir_mzProgression)
            dir_mzProgressions = [dir_mzProgression + str(sample.mzRanges[mzNum][0]) + '-' + str(sample.mzRanges[mzNum][1]) + os.path.sep for mzNum in range(0, len(sample.mzRanges))]
            for dir_mzProgression in dir_mzProgressions: os.makedirs(dir_mzProgression)
            dir_avgProgression = dir_sampleResults + 'Average' + os.path.sep
            os.makedirs(dir_avgProgression)
            dir_videos= dir_sampleResults + 'Videos' + os.path.sep
            os.makedirs(dir_videos)
            
            #Perform visualizations in parallel
            if parallelization:
                futures = [visualize_parhelper.remote(sample, self.simulationFlag, dir_avgProgression, dir_mzProgressions) for sample in self.samples]
                results = [x for x in tqdm(rayIterator(futures), total=len(futures), desc='Visualizations', leave=False, ascii=True)]
            else:
                results = [visualize_serial(sample, self.simulationFlag, dir_avgProgression, dir_mzProgressions) for sample in tqdm(self.samples, desc='Visualizations', leave=False, ascii=True)]

            #Ground truth borderless avg image
            if self.simulationFlag:
                saveLocation = dir_avgProgression + 'avgGroundTruth.png'
                fig=plt.figure()
                ax=fig.add_subplot(1,1,1)
                plt.axis('off')
                plt.imshow(self.samples[-1].avgGroundTruthImage, cmap='hot', aspect='auto', vmin=0, vmax=1)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(saveLocation, bbox_inches=extent)
                plt.close()
                             
            #Ground truth borderless mz images
            if self.simulationFlag:
                for mzNum in range(0, len(sample.mzRanges)):
                    saveLocation = dir_mzProgressions[mzNum] + 'groundTruth.png'
                    fig=plt.figure()
                    ax=fig.add_subplot(1,1,1)
                    plt.axis('off')
                    plt.imshow(self.samples[-1].mzImages[mzNum], cmap='hot', aspect='auto', vmin=0, vmax=1)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig(saveLocation, bbox_inches=extent)
                    plt.close()
            
            #Combine mz images into animations
            for mzNum in tqdm(range(0, len(sample.mzRanges)), desc='mz Videos', leave = False, ascii=True): 
                dataFileNames = natsort.natsorted(glob.glob(dir_mzProgressions[mzNum] + 'progression_*.png'))
                height, width, layers = cv2.imread(dataFileNames[0]).shape
                animation = cv2.VideoWriter(dir_videos + str(sample.mzRanges[mzNum][0]) + '-' + str(sample.mzRanges[mzNum][1]) + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
                for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
                animation.release()
                animation = None

            #Combine average images into animation
            dataFileNames = natsort.natsorted(glob.glob(dir_avgProgression + 'progression_*.png'))
            height, width, layers = cv2.imread(dataFileNames[0]).shape
            animation = cv2.VideoWriter(dir_videos + 'average.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (width, height))
            for specFileName in dataFileNames: animation.write(cv2.imread(specFileName))
            animation.release()
            animation = None
            
#Perform gaussianGenerator for a set of sigma values
@ray.remote
def computeRecon_parhelper(images, sample, indexes):
    return [computeRecon(images[index], sample) for index in indexes]

#Section of computeRDValue that is supported for Numba acceleration
@jit(nopython=True)
def secondComputeRDValue(image, location, radius, gaussianValues, update):
    
    #Initiate the window with zeros (in case of edge overlap)
    window = np.zeros((radius*2, radius*2))

    #Determine indexing locations for the window and image, considering possible edge overlap
    if location[1]-radius < 0:
        windowXStart = -(location[1]-radius)
        imageXStart = 0
    else:
        windowXStart = 0
        imageXStart = location[1]-radius
    if location[1]+radius > image.shape[1]:
        windowXStop = (radius*2)-(location[1]+radius-image.shape[1])
        imageXStop = image.shape[1]
    else:
        windowXStop = radius*2
        imageXStop = location[1]+radius
    if location[0]-radius < 0:
        windowYStart = -(location[0]-radius)
        imageYStart = 0
    else:
        windowYStart = 0
        imageYStart = location[0]-radius
    if location[0]+radius > image.shape[0]:
        windowYStop = (radius*2)-(location[0]+radius-image.shape[0])
        imageYStop = image.shape[0]
    else:
        windowYStop = radius*2
        imageYStop = location[0]+radius
        
    #Extract window from image
    window[windowYStart:windowYStop, windowXStart:windowXStop] = image[imageYStart:imageYStop, imageXStart:imageXStop]
    
    return np.sum(window*np.outer(gaussianValues, gaussianValues))
    #, [imageXStart, imageXStop, imageYStart, imageYStop]
    
#Compute RD values around location; takes ~0.3 seconds
def computeRDValue(image, location, sigma, update):

    #Determine radius 3 times the given sigma values (area of pixel influence)
    radius = int(np.ceil(sigma*3))
    
    return secondComputeRDValue(image, location, radius, signal.gaussian(radius*2, sigma), update)
    #np.asarray((), dtype=object)
   
#Perform gaussianGenerator for a set of sigma values
@ray.remote
def gaussian_parhelper(RDPP, idxs, sigmaValues, update, indexes):
    return [computeRDValue(RDPP, idxs[index], sigmaValues[index], update) for index in indexes]

#Perform Reduction in Distortion computation for a given sample and c value
def computeRD(sample, cValue, finalDimRD, bestCFlag, update=False, RDImage=None):
    
    #If a full calculation of RD then use the squareUnMeasured locations, otherwise find those that should be updated
    if not update: 
        unMeasuredLocations = sample.squareUnMeasuredIdxs
    else:
        unMeasuredLocations = np.empty((0,2)).astype(int)
        updateLocations = np.argwhere(sample.prevSquareMask-sample.squareMask)
        
        #Prepare variables for indexing
        updateLocations_list = updateLocations.tolist()
        squareMeasuredIdxs_list = sample.squareMeasuredIdxs.tolist()
        neighborIndices = sample.neighborIndices[:,0].ravel()
        
        #Find indices of updateLocations and then the indices of neighboring unMeasuredLocations 
        indices = [squareMeasuredIdxs_list.index(updateLocations_list[index]) for index in range(0, len(updateLocations))]
        indices = np.concatenate([np.argwhere(neighborIndices==index) for index in indices]).flatten()

        #If there are no locations that need updating, then just return the existing RDImage
        if len(indices)==0: return RDImage
        
        #Extract unMeasuredLocations to be updated and their relevant neighbor information (to avoid recalculation)
        neighborDistances = sample.neighborDistances[:,0][indices]
        unMeasuredLocations = sample.squareUnMeasuredIdxs[indices]
    
    #If not performing just an update
    if not update:
        
        #For novel variations, collapse mz difference stack, for original compute difference of collapsed mz stack
        if RDMethod == 'var': sample.RDPP = np.var(abs(sample.squaremzImages-sample.squaremzReconImages), axis=0)
        elif RDMethod == 'max': 
            sample.RDPP = np.max(abs(sample.squaremzImages-sample.squaremzReconImages), axis=0)
            
            #If optimizing c/mz then identify which mz ranges were used for each pixel (in the physical domain)
            #if bestCFlag: sample.mzRDPPMap = np.argmax(resize(abs(sample.squaremzImages-sample.squaremzReconImages), (len(sample.mzRanges), sample.finalDim[0], sample.finalDim[1]), order=0), axis=0)
            
        elif RDMethod == 'avg': sample.RDPP = np.mean(abs(sample.squaremzImages-sample.squaremzReconImages), axis=0)
        elif RDMethod == 'original': sample.RDPP = computeDifference(sample.avgSquareGroundTruthImage, sample.avgSquareReconImage)
        else: sys.exit('Error! - Unknown RD Method specified in configuration: ' + RDMethod)
        
        #Neighbor information for all unmeasured locations is already in the sample
        neighborDistances = sample.neighborDistances[:,0]
    
    #Calculate the sigma values for chosen c value
    sigmaValues = neighborDistances/cValue
    
    #Determine RDValues, parallelizing if not done so at a higher level
    if not bestCFlag and parallelization:
        RDPP_id = ray.put(sample.RDPP)
        sigmaValues_id = ray.put(sigmaValues)
        idxs_id = ray.put(unMeasuredLocations)
        results = np.asarray(list(chain.from_iterable(ray.get([gaussian_parhelper.remote(RDPP_id, idxs_id, sigmaValues_id, update, indexes) for indexes in np.array_split(np.arange(0, len(unMeasuredLocations)), multiprocessing.cpu_count())]))))
    else:
        results = np.asarray([computeRDValue(sample.RDPP, unMeasuredLocations[index], sigmaValues[index], update) for index in range(0, len(unMeasuredLocations))])

    #Update or compute 2D RD; save pre-Normalization image in case of future updates
    if not update:
        RDImage = np.zeros((sample.squareMask.shape))
        RDImage[tuple(unMeasuredLocations.T)] = results
        sample.preNormRDImage = copy.deepcopy(RDImage)
    else:
        sample.preNormRDImage[tuple(unMeasuredLocations.T)] = results
        sample.preNormRDImage[tuple(updateLocations.T)] = 0
        
    #Normalize and resize if neccessary
    RDImage = (sample.preNormRDImage-np.min(sample.preNormRDImage))/(np.max(sample.preNormRDImage)-np.min(sample.preNormRDImage))
    if finalDimRD: RDImage = resize(RDImage, tuple(sample.finalDim), order=0)
    
    #Update the previous mask, so measurement locations can be isolated in future updates
    sample.prevSquareMask = copy.deepcopy(sample.squareMask)
    
    return RDImage

#Extract features of the reconstruction to use as inputs to SLADS(-Net) models
def computePolyFeatures(sample, reconImage):
    
    #Retreive recon values
    inputValues = reconImage[sample.squareUnMeasuredIdxs[:,0], sample.squareUnMeasuredIdxs[:,1]]
    
    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in sample.squareMeasuredIdxs]))
    measuredValues = reconImage[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Find neighbor information
    neighborValues = measuredValues[sample.neighborIndices]
    
    #Create array to hold features
    feature = np.zeros((np.shape(sample.squareUnMeasuredIdxs)[0],6))
    
    #Compute std div features
    diffVect = computeDifference(neighborValues, np.transpose(np.matlib.repmat(inputValues, np.shape(neighborValues)[1],1)))
    feature[:,0] = np.sum(sample.neighborWeights*diffVect, axis=1)
    feature[:,1] = np.sqrt(np.sum(np.power(diffVect,2),axis=1))
    
    #Compute distance/density features
    cutoffDist = np.ceil(np.sqrt((featDistCutoff/100)*(sample.area/np.pi)))
    feature[:,2] = sample.neighborDistances[:,0]
    neighborsInCircle = np.sum(sample.neighborDistances<=cutoffDist,axis=1)
    feature[:,3] = (1+(np.pi*(np.square(cutoffDist))))/(1+neighborsInCircle)
    
    #Compute gradient features; assume continuous features
    gradientImageX, gradientImageY = np.gradient(reconImage)
    feature[:,4] = abs(gradientImageY)[sample.squareUnMeasuredIdxs[:,0], sample.squareUnMeasuredIdxs[:,1]]
    feature[:,5] = abs(gradientImageX)[sample.squareUnMeasuredIdxs[:,0], sample.squareUnMeasuredIdxs[:,1]]
    
    #Fit polynomial features to the determined array
    polyFeatures = PolynomialFeatures(degree=2).fit_transform(feature)
    
    return polyFeatures

#Determine the Expected Reduction in Distortion for uneasured points in a sample
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
        
        #Normalize for PSNR comparisons
        ERD = (ERD-np.min(ERD))/(np.max(ERD)-np.min(ERD))
        
        #Resize to physical domain dimensionality
        ERD = resize(ERD, tuple(sample.finalDim), order=0)
    
    elif erdModel == 'DLADS':
        
        #t0=time.time()
        #Get and normalize input data
        if averageReconInput: inputImage = featureExtractor(sample, sample.avgSquareMeasuredImage, sample.avgSquareReconImage)
        else: inputImage = np.stack(sample.squareMeasuredmzImages, axis=-1)
        inputImage = (inputImage-np.min(inputImage))/(np.max(inputImage)-np.min(inputImage))
        inputImage, topBottomPad, leftRightPad = makeCompatible(inputImage)
        #print('Compatability: ' + str(time.time()-t0))
        
        #Send input through trained model
        #t0=time.time()
        ERD = model(inputImage, training=False)[0,:,:,0].numpy()
        #print('Model ERD (GPU): ' + str(time.time()-t0))
        
        #t0=time.time()
        ERD = ERD[topBottomPad:, leftRightPad:]
        ERD = (ERD-np.min(ERD))/(np.max(ERD)-np.min(ERD))
        ERD = resize(ERD, tuple(sample.finalDim), order=0)
        #print('Normalize/Resize Output: ' + str(time.time()-t0))

    return ERD

#Dynamically scan a sample so as to maximize reconstruction PSNR
def runSLADS(sample, model, scanMethod, cValue, percToScan, percToViz, stopPerc, simulationFlag, trainPlotFlag, animationFlag, tqdmHide, oracleFlag, bestCFlag):
    
    #Setup flags for whether reconstructions need to be performed of averages and/or each mz range
    avgReconstructionFlag, mzReconstructionFlag = False, False
    if averageReconInput or erdModel == 'SLADS-LS' or erdModel == 'SLADS-Net' or RDMethod == 'original': avgReconstructionFlag = True
    if bestCFlag or (percToScan != None and erdModel == 'DLADS' and ((lineMethod != 'segLine' and scanMethod=='linewise') or scanMethod=='pointwise')): mzReconstructionFlag = True
    
    #Determine whether neighbor information needs to be calculated each iteration
    if avgReconstructionFlag or mzReconstructionFlag or oracleFlag: neighborCalcFlag = True
    else: neighborCalcFlag = False

    #If bestC (parallel run) then need to make sample writable
    if bestCFlag: sample = copy.deepcopy(sample)
    
    #If not a simulation then sample.mzImages contains the initially measured images, mask is already populated with initial points
    if not simulationFlag: 
        sample.measuredmzImages = sample.mzImages
        sample.squareMeasuredmzImages = sample.squaremzImages
        
    #Reinitialize the mask to starting state
    sample.maskReset(simulationFlag)

    #Indicate that the stopping condition has not yet been met
    completedRunFlag = False
    
    #Perform the initial measurements
    if simulationFlag: sample.performMeasurements(np.transpose(np.where(sample.initialMask == 1)), percToScan, simulationFlag, False, bestCFlag, neighborCalcFlag)

    #Calculate the reconstruction(s)
    if avgReconstructionFlag: 
        sample.avgSquareReconImage = computeRecon(sample.avgSquareMeasuredImage, sample)
        sample.avgReconImage = resize(sample.avgSquareReconImage, tuple(sample.finalDim), order=0)
    if mzReconstructionFlag:
        if not bestCFlag and parallelization: sample.squaremzReconImages = np.asarray(list(chain.from_iterable(ray.get([computeRecon_parhelper.remote(sample.squareMeasuredmzImages, sample, indexes) for indexes in np.array_split(np.arange(0, len(sample.squareMeasuredmzImages)), multiprocessing.cpu_count())]))))
        else: sample.squaremzReconImages = np.asarray([computeRecon(squareMeasuredmzImage, sample) for squareMeasuredmzImage in sample.squareMeasuredmzImages])
        sample.mzReconImages = np.asarray([resize(squaremzReconImage, tuple(sample.finalDim), order=0) for squaremzReconImage in sample.squaremzReconImages])

    #Determine ERD or use the RD as the ERD if the oracleFlag or bestCFlag is enabled
    if oracleFlag or bestCFlag: sample.ERD = computeRD(sample, cValue, True, bestCFlag)
    else: sample.ERD = computeERD(sample, model)

    #Initialize a result object
    result = Result(sample, sample.avgGroundTruthImage, cValue, bestCFlag, oracleFlag, simulationFlag, animationFlag, neighborCalcFlag)
    
    #Check stopping criteria, just in case of a bad input
    if (scanMethod == 'pointwise' or not lineVisitAll) and (sample.percMeasured >= stopPerc): completedRunFlag = True
    if scanMethod == 'linewise' and len(sample.linesToScan) == 0: completedRunFlag = True
    
    #Perform the first update for the result
    result.update(sample, completedRunFlag)
    
    if not lineVisitAll or scanMethod != 'linewise': maxProgress = stopPerc
    else: maxProgress = 100
    #Until the stopping criteria has been met
    with tqdm(total = float(maxProgress), desc = '% Sampled', leave = False, ascii=True, disable=tqdmHide) as pbar:

        #Initialize progress bar state according to % measured
        pbar.n = round(sample.percMeasured,2)
        pbar.refresh()

        #Until the program has completed
        while not completedRunFlag:
            
            #print('\n')
            
            #t0 = time.time()
            #Find next measurement locations
            newIdxs = findNewMeasurementIdxs(sample, result, model, scanMethod, cValue, simulationFlag, oracleFlag, bestCFlag, neighborCalcFlag, percToScan)
            #print('Find Time: ' + str(time.time()-t0))
            
            #t0 = time.time()
            #Perform measurements
            sample.performMeasurements(newIdxs, percToScan, simulationFlag, False, bestCFlag, neighborCalcFlag)
            #print('Scan Time: ' + str(time.time()-t0))
            
            #t0 = time.time()
            #Calculate the reconstruction(s)
            if avgReconstructionFlag:
                sample.avgSquareReconImage = computeRecon(sample.avgSquareMeasuredImage, sample)
                sample.avgReconImage = resize(sample.avgSquareReconImage, tuple(sample.finalDim), order=0)
            if mzReconstructionFlag:
                if not bestCFlag and parallelization: sample.squaremzReconImages = np.asarray(list(chain.from_iterable(ray.get([computeRecon_parhelper.remote(sample.squareMeasuredmzImages, sample, indexes) for indexes in np.array_split(np.arange(0, len(sample.squareMeasuredmzImages)), multiprocessing.cpu_count())]))))
                else: sample.squaremzReconImages = np.asarray([computeRecon(squareMeasuredmzImage, sample) for squareMeasuredmzImage in sample.squareMeasuredmzImages])
                sample.mzReconImages = np.asarray([resize(squaremzReconImage, tuple(sample.finalDim), order=0) for squaremzReconImage in sample.squaremzReconImages])
            #print('Reconstructions: ' + str(time.time()-t0))
            
            #t0 = time.time()
            #Determine ERD or use the RD as the ERD if the oracleFlag or bestCFlag is enabled
            if oracleFlag or bestCFlag: sample.ERD = computeRD(sample, cValue, True, bestCFlag)
            else: sample.ERD = computeERD(sample, model)
            #print('E/RD Computation: ' + str(time.time()-t0))
            
            #t0 = time.time()
            #Check stopping conditions
            if (scanMethod == 'pointwise' or not lineVisitAll) and (sample.percMeasured >= stopPerc): completedRunFlag = True
            if scanMethod == 'linewise' and len(sample.linesToScan) == 0: completedRunFlag = True
            #print('Stopping Cond: ' + str(time.time()-t0))
            
            #t0 = time.time()
            #If viz limit, only update when percToViz has been met; otherwise update every iteration
            if ((scanMethod =='pointwise') and (percToViz != None) and ((sample.percMeasured - result.percMeasuredList[-1]) >= percToViz)) or (percToViz == None) or completedRunFlag: result.update(sample, completedRunFlag)
            #print('Result Update: ' + str(time.time()-t0))

            #t0 = time.time()
            #Update the progress bar
            pbar.n = round(sample.percMeasured,2)
            pbar.refresh()
            #print('Progress Bar: ' + str(time.time()-t0))
            
            #print('\n')
            
    return result

#Determine which unmeasured points of a sample should be scanned given the current E/RD
def findNewMeasurementIdxs(sample, result, model, scanMethod, cValue, simulationFlag, oracleFlag, bestCFlag, neighborCalcFlag, percToScan):

    if scanMethod == 'random':
        newIdxs = sample.unMeasuredIdxs.tolist()
        np.random.shuffle(newIdxs)
        newIdxs = newIdxs[:int((percToScan/100)*sample.area)]
    else:
        #Make sure ERD is in np array
        ERD = np.asarray(sample.ERD)
    if scanMethod == 'pointwise':
        
        #If performing a groupwise scan, use reconstruction as the measurement value, until reaching target number of points to scan
        if percToScan != None:
        
            #Create a list to hold the chosen scanning locations
            newIdxs = []
            
            #Until the percToScan has been reached, substitute reconstruction values for actual measurements
            while True:
                
                #Find next measurement location and store the chosen scanning location for later, actual measurement
                newIdxs.append(sample.unMeasuredIdxs[np.argmax(ERD[sample.mask==0])])
                
                #Perform the measurement, using values from reconstruction 
                sample.performMeasurements(newIdxs[-1], percToScan, simulationFlag, True, bestCFlag, neighborCalcFlag)
                
                #When enough new locations have been determined, break from loop
                if (sample.percMeasured-result.percMeasuredList[-1]) >= percToScan: break
                
                #Re-compute/update ERD/RD; ensure in an array
                if oracleFlag or bestCFlag: ERD = np.asarray(computeRD(sample, cValue, True, bestCFlag, True, ERD))
                else: ERD = np.asarray(computeERD(sample, model))
                
        else:
            #Identify the unmeasured location with the highest ERD value; return in a list to ensure it is iterable
            newIdxs = [sample.unMeasuredIdxs[np.argmax(ERD[sample.mask==0])]]

    elif scanMethod == 'linewise':
        #==========================================
        #OPTIMAL LINE DETERMINATION
        #==========================================

        #Choose the line with maximum ERD and extract the actual indices
        lineERDSums = [np.nansum(ERD[tuple([x[0] for x in line]), tuple([y[1] for y in line])]) for line in sample.linesToScan]
        lineToScanIdx = np.nanargmax(lineERDSums)
        lineToScanIdxs = sample.linesToScan[lineToScanIdx]
        
        #Obtain the ERD values in the chosen line
        lineERDValues = [ERD[tuple(pt)] for pt in lineToScanIdxs]
        
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
                lineToScanIdxs = np.delete(lineToScanIdxs, nextIndex, 0)
                
                #Perform the measurement, using values from reconstruction 
                sample.performMeasurements(newIdxs[-1], percToScan, simulationFlag, True, bestCFlag, neighborCalcFlag)
                
                #When enough new locations have been determined, break from loop
                if (len(newIdxs)/len(sample.linesToScan[lineToScanIdx]))*100 >= stopPerc: break
                
                #Re-compute/update ERD/RD; ensure in an array
                if oracleFlag or bestCFlag: ERD = np.asarray(computeRD(sample, cValue, True, bestCFlag, True, ERD))
                else: ERD = np.asarray(computeERD(sample, model))
                
                #Obtain the ERD values in the chosen line
                lineERDValues = [ERD[tuple(pt)] for pt in lineToScanIdxs]

        #==========================================
        
        #==========================================
        #PARTIAL LINE BY START/END POINTS
        #==========================================
        #Choose segment to scan on line which contains at least stopPerc locations with maximal ERD
        elif lineMethod == 'segLine':
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
        #Remove the selected points from further consideration, allows revisting lines, otherwise remove the line entirely
        if lineRevistMethod: sample.delPoints(newIdxs)
        else: sample.delLine(lineToScanIdx)
        #==========================================

    return np.asarray(newIdxs)

def findNeighbors(measuredIdxs, unMeasuredIdxs):

    #Calculate knn
    neigh = NearestNeighbors(n_neighbors=numNeighbors).fit(measuredIdxs)
    #neigh.fit(measuredIdxs)
    neighborDistances, neighborIndices = neigh.kneighbors(unMeasuredIdxs)

    #Determine inverse distance weights
    unNormNeighborWeights = 1.0/np.square(neighborDistances)
    neighborWeights = unNormNeighborWeights/(np.sum(unNormNeighborWeights, axis=1))[:, np.newaxis]

    return neighborIndices, neighborWeights, neighborDistances

#Perform the reconstruction without 0-padding
def computeRecon(inputImage, sample):

    #Create a blank image for the reconstruction
    reconImage = np.zeros(sample.squareDim)

    #Retreive measured values
    measuredValues = inputImage[tuple(sample.squareMeasuredIdxs.T)]
    
    #Compute reconstruction values using IDW (inverse distance weighting)
    reconImage[tuple(sample.squareUnMeasuredIdxs.T)] = np.sum(measuredValues[sample.neighborIndices]*sample.neighborWeights, axis=1)

    #Combine measured values back into the reconstruction image
    reconImage[tuple(sample.squareMeasuredIdxs.T)] = measuredValues
    reconImage[sample.squareMeasuredIdxs[:,0], sample.squareMeasuredIdxs[:,1]] = measuredValues

    return reconImage

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
    #output = tfp.math.clip_by_value_preserve_gradient(output, 0, 1)

    return tf.keras.Model(inputs=inputs, outputs=output)

def mlp(numFilters, numChannels):
    inputs = Input(shape=(None,None,numChannels))
    dense_1 = Dense(numFilters, activation='relu', kernel_initializer='he_normal')(inputs)
    dense_2 = Dense(numFilters, activation='relu', kernel_initializer='he_normal')(dense_1)
    dense_3 = Dense(numFilters, activation='relu', kernel_initializer='he_normal')(dense_2)
    dense_4 = Dense(numFilters, activation='relu', kernel_initializer='he_normal')(dense_3)
    dense_5 = Dense(numFilters, activation='relu', kernel_initializer='he_normal')(dense_4)
    output = Dense(1, activation='linear', kernel_initializer='he_normal')(dense_5)
    #output = tfp.math.clip_by_value_preserve_gradient(output, 0, 1)

    return tf.keras.Model(inputs=inputs, outputs=output)

def flatunet(numFilters, numChannels):

    inputs = Input(shape=(None,None,numChannels))
    
    conv1 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = LayerNormalization()(conv1)
    down1 = Conv2D(numFilters, 2, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    
    conv2 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down1)
    conv2 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = LayerNormalization()(conv2)
    down2 = Conv2D(numFilters, 2, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    
    conv3 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down2)
    conv3 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = LayerNormalization()(conv3)
    conv3 = Dropout(rate=0.5)(conv3)
    down3 = Conv2D(numFilters, 2, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    
    conv4 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down3)
    conv4 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = LayerNormalization()(conv4)
    conv4 = Dropout(rate=0.5)(conv4)

    up7 = Conv2D(numFilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LayerNormalization()(conv7)

    up8 = Conv2D(numFilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LayerNormalization()(conv8)

    up9 = Conv2D(numFilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LayerNormalization()(conv9)
    
    output = Conv2D(1, (1,1), activation='linear', padding='same')(conv9)
    
    return tf.keras.Model(inputs=inputs, outputs=output)
    
    
def unet(numFilters, numChannels):

    inputs = Input(shape=(None,None,numChannels))
    
    conv1 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = LayerNormalization()(conv1)
    down1 = Conv2D(numFilters, 2, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    
    conv2 = Conv2D(numFilters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down1)
    conv2 = Conv2D(numFilters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = LayerNormalization()(conv2)
    down2 = Conv2D(numFilters*2, 2, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    
    conv3 = Conv2D(numFilters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down2)
    conv3 = Conv2D(numFilters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = LayerNormalization()(conv3)
    conv3 = Dropout(rate=0.5)(conv3)
    down3 = Conv2D(numFilters*4, 2, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    
    conv4 = Conv2D(numFilters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(down3)
    conv4 = Conv2D(numFilters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = LayerNormalization()(conv4)
    conv4 = Dropout(rate=0.5)(conv4)

    up7 = Conv2D(numFilters*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(numFilters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(numFilters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LayerNormalization()(conv7)

    up8 = Conv2D(numFilters*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(numFilters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(numFilters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LayerNormalization()(conv8)

    up9 = Conv2D(numFilters*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(numFilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LayerNormalization()(conv9)
    
    output = Conv2D(1, (1,1), activation='linear', padding='same')(conv9)
    
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
    
    topBottomPad = int(np.ceil(image.shape[0]/depthFactor)*depthFactor)-image.shape[0]
    leftRightPad = int(np.ceil(image.shape[1]/depthFactor)*depthFactor)-image.shape[1]
    
    #Reshape for tensor transition, as needed by number of channels
    if len(image.shape) > 2: 
        image = np.pad(image, [(topBottomPad, 0), (leftRightPad, 0), (0,0) ], mode='constant', constant_values=0)
        image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    else: 
        image = np.pad(image, [(topBottomPad, 0), (leftRightPad, 0)], mode='constant', constant_values=0)
        image = image.reshape((1,image.shape[0],image.shape[1],1))

    return image, topBottomPad, leftRightPad

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

#Metric for model training; compute PSNR between two tensors; normalize prediction first (ground-truth is expected as [0, 1] range
def PSNR(imageTrue, imagePred): 
    imagePred = tf.divide(tf.subtract(imagePred,tf.reduce_min(imagePred)), tf.subtract(tf.reduce_max(imagePred),tf.reduce_min(imagePred)))
    return tf.reduce_mean(tf.image.psnr(imageTrue, imagePred, max_val=1.0))

#Unused intersection over union metric
def iou(groundTruth, prediction):
    return np.sum(np.logical_and(groundTruth, prediction)) / np.sum(np.logical_or(groundTruth, prediction))
