#==================================================================
#MODEL: DLADS-PY
#
#PyTorch equivalent implementation to MODEL_DLADS_TF.py
#
#NOTE(S): DIFFERENCES WITH RESPECT TO ORIGINAL DLADS IMPLEMENTATION
#
# 1) DLADS was originally implemented using nearest neighbor upsampling followed by a convolution layer
#    PyTorch has a deterministic nearest upsampling, but TensorFlow does not.
#    Coincidently, where TensorFlow does have a deterministic bilinear upsampling, PyTorch does not.
#
#    An equivalent bilinear method was created for PyTorch for deterministic comparison. 
#    However, this implementation could not be optimized sufficiently for comparable performance.
#
#    Another deterministic alternative in PyTorch is to use exact nearest-neighbor sampling with a subsequent binomial filter. 
#    To match with bilinear upsampling behavior this requires even spatial dimensionality at all layers of depth.
#    Therefore, zero padding has been added at the top of the network, proportional to the input data dimensionality. 
#    There should be an incidental benefit in reduction of shifting artifacts during the downsampling process.
#    The padding has been done in both TensorFlow and PyTorch models to employ identical NN architectures.
#
# 2) Augmentation now only uses random rotations in 90 degree increments and horizonal/vertical flips.
#
# 3) DLADS originally allowed training data to be shuffled across epochs, this has now been prevented.
#
#==================================================================

#Random rotation transform using a discrete set of angles
class RandomDiscreteRotate:
    def __init__(self): pass
    def __call__(self, inputs): 
        rots = torch.randint(4, (1,)).item()
        return torch.rot90(inputs, k=rots, dims=(-2, -1))
    
#Perform augmentation and setup for DLADS data processing
class DataPreprocessing_DLADS(Dataset):
    def __init__(self, inputs, labels, device, augmentFlag):
        super().__init__()
        self.noAugmentFlag = not augmentFlag
        
        if storeOnDevice:
            self.data_Inputs = [torch.from_numpy(item).float().to(device) for item in inputs]
            self.data_Labels = [torch.from_numpy(item).float().to(device) for item in labels]
        else: 
            self.data_Inputs = [torch.from_numpy(item).float() for item in inputs]
            self.data_Labels = [torch.from_numpy(item).float() for item in labels]
        self.channelSplit = [self.data_Inputs[0].shape[0]]
        
        if augmentFlag: 
            self.transform = transforms.Compose([
                v2.RandomHorizontalFlip(p=0.5), 
                v2.RandomVerticalFlip(p=0.5),
                RandomDiscreteRotate()
            ])
            self.data_Merged = [torch.cat([self.data_Inputs[index], self.data_Labels[index]], 0) for index in range(0, len(self.data_Inputs))]
        
    def __getitem__(self, index):
        if self.noAugmentFlag: data, label = self.data_Inputs[index], self.data_Labels[index]
        else: data, label = torch.tensor_split(self.transform(self.data_Merged[index]), self.channelSplit, 0)
        return data, label

    def __len__(self):
        return len(self.data_Inputs)

#Compute RDPPs
def computeRDPPs(labels, recons):
    return abs(labels-recons)

#Prepare data for DLADS/GLANDS model input
def prepareInput(reconImages, squareMask, squareOpticalImage=None):
    if reconImages.ndim==2: reconImages = np.expand_dims(reconImages, 0)
    inputStack = []
    if 'measureData' in inputChannels: inputStack.append(reconImages*squareMask)
    if 'reconData' in inputChannels: inputStack.append(reconImages*(1-squareMask))
    if 'combinedData' in inputChannels: inputStack.append(reconImages)
    if 'opticalData' in inputChannels: inputStack.append(np.repeat(np.expand_dims(squareOpticalImage, 0), len(reconImages), axis=0))
    if 'mask' in inputChannels: inputStack.append(np.repeat(np.expand_dims(squareMask, 0), len(reconImages), axis=0))
    if len(inputStack) != len(inputChannels): sys.exit('\nError - The number of intended input channels did not match with the number added to the input stack. Please verify that the specified inputChannels are valid.')
    return np.moveaxis(np.stack(inputStack, axis=-1), -1, 1)

#Custom definition of bilinear interpolation to force deterministic backpropogation
#Default behavior follows the default Bilinear arguments used by TensorFlow (tf.image.resize) and PyTorch (torch.nn.functional.interpolate)
#Reference: https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html
#Reference: https://github.com/pytorch/pytorch/blob/59d71b9664b57b0ea0de0d87cea87b21daa4dd7b/aten/src/THNN/generic/upsampling.h#L32
class Bilinear(nn.Module):
    def __init__(self, align_corners=False, half_pixel_centers=True):
        super().__init__()
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers
        if align_corners and half_pixel_centers: sys.exit("Error - align_corners and half_pixel_centers may not be enabled simultaneously")
    
    def forward(self, data, outSize):
        
        with data.device as device: 
        
            #Extract data dimensions and flatten spatial information
            batches, channels, height, width = torch.tensor(data.shape, device=device)
            data = torch.flatten(data, start_dim=2)

            #Compute horizontal and vertical scaling factors
            if self.align_corners:
                x_scale = torch.clip((width-1)/(outSize[0]-1), min=0) 
                y_scale = torch.clip((height-1)/(outSize[1]-1), min=0) 
            else: 
                x_scale = torch.clip((width)/(outSize[0]), min=0) 
                y_scale = torch.clip((height)/(outSize[1]), min=0) 
            
            #Get indexes for the flattened data and as needed to compute mapping coordinates
            indexes = torch.arange(outSize[1] * outSize[0], device=device)
            x, y = torch.remainder(indexes, outSize[0]), indexes//outSize[0]
            
            #Apply half-pixel offset
            if self.half_pixel_centers:
                x_in = ((x+0.5)*x_scale)-0.5
                y_in = ((y+0.5)*y_scale)-0.5
            else: 
                x_in = x*x_scale
                y_in = y*y_scale
            
            #Floor the half-pixel offset for weight and coordinate computations
            x_in_floor = torch.floor(x_in)
            y_in_floor = torch.floor(y_in)
            
            #Compute weights needed to scale location values during interpolation
            x_weight = x_in-x_in_floor
            y_weight = y_in-y_in_floor

            #Get upper and lower coordinates that will be mapped to the output
            x_lower = torch.clip(x_in_floor, min=0).to(torch.int)
            y_lower = torch.clip(y_in_floor, min=0).to(torch.int)
            x_upper = torch.clip(torch.ceil(x_in), max=width-1).to(torch.int)
            y_upper = torch.clip(torch.ceil(y_in), max=height-1).to(torch.int)
            
            #Get contributions for destination locations from four relevant locations in the origin data
            a = data[:, :, y_lower*width + x_lower] * (1.-x_weight) * (1.-y_weight)
            b = data[:, :, y_lower*width + x_upper] * x_weight * (1.-y_weight)
            c = data[:, :, y_upper*width + x_lower] * y_weight * (1.-x_weight)
            d = data[:, :, y_upper*width + x_upper] * x_weight * y_weight
            
            #Combine the contributions and reshape the data
            return torch.reshape(a+b+c+d, (batches, channels, outSize[0], outSize[1]))

#Deterministic reflection padding; modified from https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L322
class PadRef(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.register_buffer('indices_LeftTop', torch.tensor([i for i in range(padding, 0, -1)]))
        self.register_buffer('indices_RightBottom', torch.tensor([-(i + 2) for i in range(padding)]))
        
    def forward(self, data):
        x_indices = torch.cat([self.indices_LeftTop, torch.arange(data.shape[-1], device=data.device), self.indices_RightBottom])
        y_indices = torch.cat([self.indices_LeftTop, torch.arange(data.shape[-2], device=data.device), self.indices_RightBottom])
        return data[:, :, y_indices[:, None], x_indices[None, :]]

#Convolution using a 1-sigma low pass binomial filter; modified from https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
#If dimensions are even and applied after nearest-exact upsampling this (with reflection padding) can be used to replicate a deterministic bilinear upsampling
class Conv_BF_DLADS(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        filter = np.array([1., 2., 1.])
        filter = torch.Tensor(filter[:,None]*filter[None,:])
        filter = filter/torch.sum(filter)
        self.register_buffer('filter', filter[None,None,:,:].repeat((in_channels,1,1,1)))
        self.pad = PadRef(1)
        
    def forward(self, data):
        return F.conv2d(self.pad(data), self.filter, stride=self.stride, groups=self.in_channels)

#Convolutional block for DLADS
class Conv_PC_DLADS(nn.Module):
    def __init__(self, numIn, numOut, act):
        super().__init__()
        if act=='LeakyReLU': self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act=='ReLU': self.act = nn.ReLU(inplace=True)
        else: sys.exit('\nError - Unexpected activation function specified.')
        self.conv0 = nn.Conv2d(in_channels=numIn, out_channels=numOut, kernel_size=1, stride=1, padding='same', bias=useBias)
        self.conv1 = nn.Conv2d(in_channels=numOut, out_channels=numOut, kernel_size=3, stride=1, padding='same', bias=useBias)
        nn.init.xavier_uniform_(self.conv0.weight, gain=1.0)
        nn.init.xavier_uniform_(self.conv1.weight, gain=1.0)

    def forward(self, data):
        return self.act(self.conv1(self.act(self.conv0(data))))

#Downsampling convolutional block for DLADS
class Conv_DN_DLADS(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0 = Conv_PC_DLADS(numIn, numOut, 'LeakyReLU')
    
    def forward(self, data):
        return self.conv0(self.pool(data))

#Upsampling convolutional block for DLADS
class Conv_UP_DLADS(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.blur = Conv_BF_DLADS(numIn, stride=1)
        self.conv0 = nn.Conv2d(in_channels=numIn, out_channels=numOut, kernel_size=2, stride=1, padding='same', bias=useBias) 
        self.conv1 = Conv_PC_DLADS(numIn, numOut, 'ReLU')
        nn.init.xavier_uniform_(self.conv0.weight, gain=1.0)
        
    def forward(self, data, skip):
        data = self.blur(F.interpolate(data, size=skip.shape[2:], mode='nearest-exact'))
        data = self.act(self.conv0(data))
        return self.conv1(torch.cat([data, skip], 1))

#DLADS model
class Model_DLADS(nn.Module):
    def __init__(self, numFilt, numChan):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.convIn0 = Conv_PC_DLADS(numChan, numFilt, 'LeakyReLU')
        self.convDn0 = Conv_DN_DLADS(numFilt, numFilt*2)
        self.convDn1 = Conv_DN_DLADS(numFilt*2, numFilt*4)
        self.convDn2 = Conv_DN_DLADS(numFilt*4, numFilt*8)
        self.convDn3 = Conv_DN_DLADS(numFilt*8, numFilt*16)
        self.convUp3 = Conv_UP_DLADS(numFilt*16, numFilt*8)
        self.convUp2 = Conv_UP_DLADS(numFilt*8, numFilt*4)
        self.convUp1 = Conv_UP_DLADS(numFilt*4, numFilt*2)
        self.convUp0 = Conv_UP_DLADS(numFilt*2, numFilt)
        self.convFn0 = nn.Conv2d(in_channels=numFilt, out_channels=1, kernel_size=1, stride=1, padding='same', bias=useBias)
        
        nn.init.xavier_uniform_(self.convFn0.weight, gain=1.0)
        
    def forward(self, data):
        if padInputData:
            padHeight, padWidth = (-(-data.shape[-2]//16)*16)-data.shape[-2], (-(-data.shape[-1]//16)*16)-data.shape[-1]
            data = F.pad(data, (padWidth, 0, padHeight, 0), mode='constant')
        
        convDn0 = self.convIn0(data)
        convDn1 = self.convDn0(convDn0)
        convDn2 = self.convDn1(convDn1)
        convDn3 = self.convDn2(convDn2)
        convDn4 = self.convDn3(convDn3)
        convUp3 = self.convUp3(convDn4, convDn3)
        convUp2 = self.convUp2(convUp3, convDn2)
        convUp1 = self.convUp1(convUp2, convDn1)
        convUp0 = self.convUp0(convUp1, convDn0)
        convOut = self.act(self.convFn0(convUp0))
        if padInputData: return convOut[:, :, padHeight:, padWidth:]
        else: return convOut

#Define DLADS model
class DLADS_PY:
    def __init__(self, trainFlag, local_gpus, modelDirectory=None, modelName=None):
        
        #Create model
        self.model = Model_DLADS(numStartFilters, len(inputChannels))
        
        #If not training, load parameters (before potential parallelization on multiple GPUs) and setup for inferencing
        if not trainFlag: 
            modelPath = modelDirectory + modelName
            with multivolumefile.open(modelPath + os.path.sep + modelName + '.7z', mode='rb') as modelArchive:
                with py7zr.SevenZipFile(modelArchive, 'r') as archive:
                    archive.extract(modelDirectory)
            _ = self.model.load_state_dict(torch.load(modelPath + '.pt'))
            _ = self.model.train(False)
            os.remove(modelPath + '.pt')
        
        #Configure CPU/GPU computation environment; allocate model location and batch sizes accordingly
        self.device = torch.device(f"cuda:{local_gpus[-1]}" if len(local_gpus) > 0 else "cpu")
        self.model.to(self.device)
        self.batchsize_TRN, self.batchsize_VAL = 1, 1
        
        #Upcoming compile function might improve speed even further; not currently working 2.2.0
        #https://github.com/pytorch/pytorch/pull/119750
        #self.model = torch.compile(self.model, dynamic=True, mode="reduce-overhead")
    
    def loadData(self, trainingDatabase, validationDatabase):
    
        #Extract and prepare training data
        inputs_TRN, labels_TRN = [], []
        for sample in tqdm(trainingDatabase, desc = 'Training Data Setup', leave=True, ascii=asciiFlag):
            inputStack = prepareInput(sample.squareChanReconImages, sample.squareMask, trainingSampleData[sample.sampleDataIndex].squareOpticalImage)
            for chanNum in range(0, len(sample.squareRDs)):
                input = inputStack[chanNum]
                label = np.expand_dims(sample.squareRDs[chanNum], 0)
                inputs_TRN.append(input)
                labels_TRN.append(label)
        data_TRN = DataPreprocessing_DLADS(inputs_TRN, labels_TRN, self.device, augTrainData)
        self.dataloader_TRN = DataLoader(data_TRN, batch_size=self.batchsize_TRN, num_workers=0, shuffle=True)
        self.numTRN = len(self.dataloader_TRN)
        
        #If there is not a validation set then indicate such, otherwise extract and prepare validation data
        if len(validationDatabase)<=0: 
            self.valFlag = False
            vizSamples = None
        else:
            self.valFlag = True
            vizSampleIndices = [0, len(np.arange(initialPercToScanTrain, stopPercTrain))]
            self.numViz = len(vizSampleIndices)
            inputs_VAL, labels_VAL = [], []
            self.inputs_Viz, self.labels_Viz = [], []
            for i, sample in enumerate(tqdm(validationDatabase, desc = 'Validation Data Setup', leave=True, ascii=asciiFlag)):
                inputStack = prepareInput(sample.squareChanReconImages, sample.squareMask, validationSampleData[sample.sampleDataIndex].squareOpticalImage)
                if i in vizSampleIndices:
                    if storeOnDevice: self.inputs_Viz.append(torch.from_numpy(inputStack).float().to(self.device))
                    else: self.inputs_Viz.append(torch.from_numpy(inputStack).float())
                    self.labels_Viz.append(sample.squareRD)
                for chanNum in range(0, len(sample.squareRDs)):
                    input = inputStack[chanNum]
                    label = np.expand_dims(sample.squareRDs[chanNum], 0)
                    inputs_VAL.append(input)
                    labels_VAL.append(label)
            data_VAL = DataPreprocessing_DLADS(inputs_VAL, labels_VAL, self.device, False)
            self.dataloader_VAL = DataLoader(data_VAL, batch_size=self.batchsize_VAL, num_workers=0, shuffle=False)
            self.numVAL = len(self.dataloader_VAL)
    
    #Produce/save visualizations after a training epoch for DLADS
    def visualizeTraining(self, epoch):
        
        if self.valFlag: f = plt.figure(figsize=(15,15))
        else: f = plt.figure(figsize=(15,8))
        f.subplots_adjust(bottom=0.05, left=0.10, right=0.90, top=0.88, wspace=0.20, hspace=0.30)
        plt.rcParams['font.size'] = 10
        
        if self.valFlag: ax = plt.subplot2grid((3,1), (0,0))
        else: ax = plt.subplot2grid((1,1), (0,0))
        if self.sepEpochs != 0: 
            ax.plot(np.ma.masked_invalid(self.meanLoss_HST), label='HST Mean', alpha=0.7, color='black')
            ax.plot(np.ma.masked_invalid(self.meanLoss_CUR), label='CUR Mean', alpha=0.7, color='green')
        ax.plot(self.loss_TRN, label='TRN', alpha=0.7, color='red')
        if self.valFlag: ax.plot(self.loss_VAL, label='VAL', alpha=0.7, color='blue')
        ax.set_yscale('log')
        ax.legend(loc='upper right', fontsize=10)
        
        if self.valFlag: 
            for vizSampleNum in range(0, self.numViz): 
            
                if not storeOnDevice: input = self.inputs_Viz[vizSampleNum].to(self.device)
                else: input = self.inputs_Viz[vizSampleNum]
                
                squareRD = self.labels_Viz[vizSampleNum]
                with torch.inference_mode(): squareERD = torch.mean(self.model(input), 0).detach().cpu().numpy()[0]
                ERD_NRMSE, ERD_SSIM, ERD_PSNR = compareImages(squareRD, squareERD, np.min(squareRD), np.max(squareRD))
                
                ax = plt.subplot2grid((3,2), (vizSampleNum+1,0))
                im = ax.imshow(squareRD, aspect='auto', interpolation='none')
                ax.set_title('RD', fontsize=15)
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                cbar.formatter.set_powerlimits((0, 0))
                
                ax = plt.subplot2grid((3,2), (vizSampleNum+1,1))
                im = ax.imshow(squareERD, aspect='auto', interpolation='none')
                plotTitle = 'ERD - NRMSE: ' + '{:.6f}'.format(round(ERD_NRMSE, 6)) + '\nSSIM: ' + '{:.6f}'.format(round(ERD_SSIM, 6)) + '; PSNR: ' + '{:.6f}'.format(round(ERD_PSNR, 6))
                ax.set_title(plotTitle, fontsize=15)
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                cbar.formatter.set_powerlimits((0, 0))
        
        plotTitle = 'EPOCH: ' + ('{:0'+str(maxEpochsPrecision)+'}').format(epoch) + ', LR: ' + '{:.1e}'.format(self.learningRate)
        plotTitle += '\nPAT: ' + ('{:0'+str(maxPatiencePrecision)+'}').format(self.patienceCounter) + '/' + str(maxPatience) + ", STG: " + ('{:0'+str(maxStagnationPrecision)+'}').format(self.stagnationCounter) + "/" + str(maxStagnation)
        plotTitle += '\nLOSS - TRN: ' + '{:.6f}'.format(round(self.loss_TRN[-1], 6)) + ', '
        if self.valFlag: plotTitle += 'VAL: ' + '{:.6f}'.format(round(self.loss_VAL[-1], 6)) + ', '
        else: plotTitle += 'VAL: N/A,   '
        plotTitle += 'BST: ' + '{:.6f}'.format(round(self.loss_BST, 6)) +' at Epoch: ' + ('{:0'+str(maxEpochsPrecision)+'}').format(self.epoch_BST)
        if (self.sepEpochs != 0) and (len(self.meanLoss_HST) > 0): plotTitle += '\nMean Loss - HST: '+ '{:.6f}'.format(round(self.meanLoss_HST[-1], 6)) +', CUR: ' + '{:.6f}'.format(round(self.meanLoss_CUR[-1], 6))
        else: plotTitle += '\nMean Loss - Historical: N/A,   Current: N/A    '
        plt.suptitle(plotTitle, fontsize=20, fontweight='bold')
        
        #Save resulting plot
        f.savefig(dir_TrainingModelResults + 'epoch_' +str(epoch) + '.tiff')
        plt.close(f)
        
    def computeLoss(self, data, label, trainFlag=False):
        
        #Zero network gradients
        if trainFlag: self.model.zero_grad()
        
        #Move data to device if not already there
        if not storeOnDevice: data, label = data.to(self.device), label.to(self.device)
        
        #Compute loss
        loss = torch.mean(torch.abs(self.model(data)-label))
       
        #Compute loss gradients and update network parameters
        if trainFlag:
            loss.backward()
            self.opt.step()
        
        return loss.item()
        
    def setOptimizer(self):
        if optimizer == 'AdamW': self.opt = optim.AdamW(self.model.parameters(), lr=self.learningRate)
        elif optimizer == 'NAdam': self.opt = optim.NAdam(self.model.parameters(), lr=self.learningRate)
        elif optimizer == 'Adam': self.opt = optim.Adam(self.model.parameters(), lr=self.learningRate)
        elif optimizer == 'RMSprop': self.opt = optim.RMSprop(self.model.parameters(), lr=self.learningRate)
        elif optimizer == 'SGD': self.opt = optim.SGD(self.model.parameters(), lr=self.learningRate)
        else: sys.exit('\nError - Unknown optimizer was specified.')
        
    def train(self):
        
        #Setup learning rate and optimizer
        self.learningRate = copy.deepcopy(learningRate)
        self.setOptimizer()
        
        #Setup storage for losses/events
        self.loss_TRN, self.loss_VAL, self.meanLoss_HST, self.meanLoss_CUR = [], [], [], []
        
        #Setup variables for early stopping critera
        visualizeLastEpoch, self.loss_BST, self.epoch_BST, endTraining, patienceLost, modelStagnation = maxEpochs-1, np.inf, -1, False, False, False
        if sepEpochs>0: self.sepEpochs, self.minEpochs = copy.deepcopy(sepEpochs), sepEpochs*2
        else: self.sepEpochs, self.minEpochs = 0, 0
        
        #Create progress bar
        trainingBar = tqdm(range(maxEpochs), desc='Epochs', leave=True, ascii=asciiFlag)
        
        #Perform model training
        t0 = time.perf_counter()
        for epoch in trainingBar:
            
            #Compute losses over the training dataset
            _ = self.model.train(True)
            self.loss_TRN.append(np.mean([self.computeLoss(data, label, True) for data, label in tqdm(self.dataloader_TRN, total=self.numTRN, desc='TRN', leave=False, ascii=asciiFlag)]))
            
            #Compute losses over the validation dataset and update the current loss used for training status evaluation
            _ = self.model.train(False)
            if self.valFlag: 
                with torch.inference_mode(): 
                    self.loss_VAL.append(np.mean([self.computeLoss(data, label, False) for data, label in tqdm(self.dataloader_VAL, total=self.numVAL, desc='VAL', leave=False, ascii=asciiFlag)]))
                loss_CUR = self.loss_VAL[-1]
            else: loss_CUR = self.loss_TRN[-1]
            
            #If this model is the best performing, update the best model parameters and reset applicable variables; if not an improvement and not using mean early stopping, increase the patience counter
            if (loss_CUR < self.loss_BST): self.model_BST, self.loss_BST, self.epoch_BST, self.patienceCounter, self.stagnationCounter = copy.deepcopy(self.model.state_dict()), loss_CUR, epoch, 0, 0
            elif (self.sepEpochs == 0) and (epoch >= self.minEpochs): self.patienceCounter += 1
            
            #If using mean early stopping, then evaluate/track training status by comparing historical and current moving averages; if below minEpochs, append nan as placeholder
            if (self.sepEpochs != 0):
                if (epoch >= self.minEpochs):
                    if self.valFlag: 
                        meanLoss_HST = np.mean(self.loss_VAL[-self.minEpochs:-self.sepEpochs])
                        meanLoss_CUR = np.mean(self.loss_VAL[-self.sepEpochs:])
                    else: 
                        meanLoss_HST = np.mean(self.loss_TRN[-self.minEpochs:-self.sepEpochs])
                        meanLoss_CUR = np.mean(self.loss_TRN[-self.sepEpochs:])
                    self.meanLoss_HST.append(meanLoss_HST)
                    self.meanLoss_CUR.append(meanLoss_CUR)
                    if meanLoss_CUR >= meanLoss_HST: self.patienceCounter += 1
                    elif self.patienceCounter > 0: self.patienceCounter -= 1
                else:
                    self.meanLoss_HST.append(np.nan)
                    self.meanLoss_CUR.append(np.nan)
            
            #Evaluate/track training stagnation
            if epoch >= 1: 
                if self.valFlag and (self.loss_VAL[-1] == self.loss_VAL[-2]): self.stagnationCounter +=1
                elif not self.valFlag and (self.loss_TRN[-1] == self.loss_TRN[-2]): self.stagnationCounter +=1
                else: self.stagnationCounter = 0
            
            #Evaluate stopping criteria
            if (self.patienceCounter >= maxPatience): patienceLost = True
            if (self.stagnationCounter >= maxStagnation): modelStagnation = True
            if (patienceLost or modelStagnation): endTraining = True
            
            #Update progress bar with epoch data
            progBarString = "PAT: " + ('{:0'+str(maxPatiencePrecision)+'}').format(self.patienceCounter) + "/" + str(maxPatience)
            progBarString += ", STG: " + ('{:0'+str(maxStagnationPrecision)+'}').format(self.stagnationCounter) + "/" + str(maxStagnation)
            progBarString += ", LOSS - TRN: " + '{:.6f}'.format(round(self.loss_TRN[-1], 6))
            if self.valFlag: progBarString += ", VAL: " + '{:.6f}'.format(round(self.loss_VAL[-1], 6))
            trainingBar.set_postfix_str(progBarString)
            trainingBar.refresh()
            
            #Perform visualization(s)
            if trainingProgressionVisuals and ((epoch == 0) or (epoch%trainingVizSteps == 0) or (epoch == self.epoch_BST) or (epoch == visualizeLastEpoch) or endTraining): self.visualizeTraining(epoch)
            
            #If the model has stagnated, exit the program
            if modelStagnation: sys.exit('\nError - The model was unable to be trained; this is most likely due to the learning rate being set too high.')
            
            #As applicable, terminate the training loop
            if endTraining: break
        
        #Compute, store, and print training time
        t1 = time.perf_counter()
        lines = ['Model Training Time: ' + str(datetime.timedelta(seconds=(t1-t0)))]
        lines.append('Epoch Time: ' + str((t1-t0)/(epoch+1)) + ' s/epoch')
        with open(dir_TrainingResults + 'trainingTime.txt', 'w') as f:
            for line in lines: 
                _ = f.write(line+'\n')
                print(line)
        
        #Strip out any parallel 'module' references from the model definition
        self.model_BST = {key.replace("module.", ""): value for key, value in self.model_BST.items()}
        
        #Store the model across multiple 100 Mb files to bypass Github file size limits
        modelPath = dir_TrainingResults + modelName
        torch.save(self.model_BST, modelPath + '.pt')
        if os.path.exists(modelPath): shutil.rmtree(modelPath)
        os.makedirs(modelPath)
        with multivolumefile.open(modelPath + os.path.sep + modelName + '.7z', mode='wb', volume=104857600) as modelArchive:
            with py7zr.SevenZipFile(modelArchive, 'w') as archive:
                archive.writeall(modelPath + '.pt', modelName + '.pt')
        os.remove(modelPath + '.pt')
        
        #Save training history
        history = np.vstack([np.array(range(0, epoch+1)), self.loss_TRN])
        if self.valFlag: history = np.vstack([history, self.loss_VAL])
        pd.DataFrame(history.T, columns=['Epoch','Loss_TRN', 'Loss_VAL']).to_csv(dir_TrainingResults+'trainingHistory.csv', index=False)
        
    def predict(self, input):
        if len(input.shape) == 3: input = np.expand_dims(input, 0)
        input = torch.from_numpy(input).float()
        input = input.to(self.device)
        with torch.inference_mode(): squareERDs = np.moveaxis(self.model(input).detach().cpu().numpy(), 1, 0)[0]
        return squareERDs
        