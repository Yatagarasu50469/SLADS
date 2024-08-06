#==================================================================
#MODEL: SLADS(-LS and -Net)
#==================================================================

#Compute RDPPs
def computeRDPPs(labels, recons):
    return abs(labels-recons)

#Extract features of the reconstruction to use as inputs to SLADS(-Net) models
def computePolyFeatures(sampleData, tempScanData, reconImages):
    
    #Retreive recon values
    inputValues = reconImages[tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]]
    
    #Retrieve the measured values
    idxsX, idxsY = map(list, zip(*[tuple(idx) for idx in tempScanData.squareMeasuredIdxs]))
    measuredValues = reconImages[np.asarray(idxsX), np.asarray(idxsY)]
    
    #Find neighbor information
    neighborValues = measuredValues[tempScanData.neighborIndices]
    
    #Create array to hold features
    feature = np.zeros((np.shape(tempScanData.squareUnMeasuredIdxs)[0],6), dtype=np.float32)
    
    #Compute std div features
    diffVect = abs(neighborValues-np.transpose(np.matlib.repmat(inputValues, np.shape(neighborValues)[1],1)))
    feature[:,0] = np.sum(tempScanData.neighborWeights*diffVect, axis=1)
    feature[:,1] = np.sqrt(np.sum(np.power(diffVect,2),axis=1))
    
    #Compute distance/density features
    cutoffDist = np.ceil(np.sqrt((featDistCutoff/100)*(sampleData.area/np.pi)))
    feature[:,2] = tempScanData.neighborDistances[:,0]
    neighborsInCircle = np.sum(tempScanData.neighborDistances<=cutoffDist,axis=1)
    feature[:,3] = (1+(np.pi*(np.square(cutoffDist))))/(1+neighborsInCircle)
    
    #Compute gradient features; assume continuous features
    gradientImageX, gradientImageY = np.gradient(reconImages)
    feature[:,4] = abs(gradientImageY)[tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]]
    feature[:,5] = abs(gradientImageX)[tempScanData.squareUnMeasuredIdxs[:,0], tempScanData.squareUnMeasuredIdxs[:,1]]
    
    #Fit polynomial features to the determined array
    polyFeatures = PolynomialFeatures(degree=2).fit_transform(feature)
    
    return polyFeatures
    
class SLADS:
    def __init__(self, trainFlag, local_gpus=None, modelDirectory=None, modelName=None):
        pass
    
    def loadData(self, trainingDatabase, validationDatabase):
    
        #Extract polyFeatures and squareRDValues for each input channel in the sample
        polyFeatureStack, squareRDValueStack = [], []
        for sample in trainingDatabase:
            for channelNum in range(0, len(sample.polyFeatures)):
                polyFeatureStack.append(sample.polyFeatures[channelNum])
                squareRDValueStack.append(sample.squareRDValues[channelNum])
        
        #Collapse the stacks for regression
        self.bigPolyFeatures = np.row_stack(polyFeatureStack)
        self.bigRD = np.concatenate(squareRDValueStack)
        
    def train(self):
        
        #Create and save regression model based on user selection
        if erdModel == 'SLADS-LS': model = linear_model.LinearRegression()
        elif erdModel == 'SLADS-Net': model = nnr(activation='identity', solver='adam', alpha=1e-4, hidden_layer_sizes=(50, 5), random_state=1, max_iter=500)
        model.fit(self.bigPolyFeatures, self.bigRD)
        np.save(dir_TrainingResults + modelName, model)
