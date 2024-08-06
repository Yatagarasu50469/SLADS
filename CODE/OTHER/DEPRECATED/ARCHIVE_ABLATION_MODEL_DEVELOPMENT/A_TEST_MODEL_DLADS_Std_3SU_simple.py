#==================================================================
#MODEL: DLADS
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

#Compute RDPPs, standardizing input data as done in model inferencing (must use torch as numpy arithmetic optimization yields slightly different values)
def computeRDPPs(labels, recons):
    if standardizeInputData: 
        recons, labels = torch.from_numpy(recons).float(), torch.from_numpy(labels).float()
        if padInputData: paddedRecons = F.pad(recons.float(), ((-(-recons.shape[-1]//16)*16)-recons.shape[-1], 0, (-(-recons.shape[-2]//16)*16)-recons.shape[-2], 0), mode='constant')
        else: paddedRecons = recons
        meanValue, stdValue = torch.mean(paddedRecons, axis=(-1, -2), keepdims=True), torch.std(paddedRecons, dim=(-1, -2), correction=0, keepdims=True)
        recons, labels = torch.nan_to_num((recons-meanValue)/stdValue).numpy(), torch.nan_to_num((labels-meanValue)/stdValue).numpy()
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

#Custom implementation of instance normalization; removes epsilon addition in the denominator
class CustomInstanceNorm(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, data): return torch.nan_to_num((data-torch.mean(data, axis=(-1, -2), keepdims=True))/torch.std(data, dim=(-1, -2), correction=0, keepdims=True))

#Convolution using a 3-sigma low pass binomial filter; modified from https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
class Conv_BF_DLADS(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        filter = np.array([1., 6., 15., 20., 15., 6., 1.])
        filter = torch.Tensor(filter[:,None]*filter[None,:])
        filter = filter/torch.sum(filter)
        self.register_buffer('filter', filter[None,None,:,:].repeat((in_channels,1,1,1)))
        self.pad = PadRef(3)
        
    def forward(self, data):
        return F.conv2d(self.pad(data), self.filter, stride=self.stride, groups=self.in_channels)

#Simple convolution construct for DLADS
class Conv_SC_DLADS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, non_linearity):
        super().__init__()
        
        #Setup the base convolution layer
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=useBias)
        
        #Setup activation layers/parameters
        if non_linearity=='leaky_relu': self.act, negative_slope = nn.LeakyReLU(0.2, inplace=True), 0.2
        elif non_linearity=='relu': self.act, negative_slope = nn.ReLU(inplace=True), 0.0
        elif non_linearity=='linear': self.act, negative_slope = lambda inputs:inputs, 1.0
        elif non_linearity=='abs': self.act, negative_slope, non_linearity = lambda inputs:torch.abs(inputs), -1.0, 'leaky_relu'
        elif non_linearity=='prelu': self.act, negative_slope, non_linearity = nn.PReLU(init=0.2), 0.2, 'leaky_relu'
        else: sys.exit('\nError - Unexpected activation function specified.')
        
        #Setup initialization for base convolution layer
        if initialization=='xavier_uniform': nn.init.xavier_uniform_(self.conv.weight)
        elif initialization=='xavier_normal': nn.init.xavier_normal_(self.conv.weight)
        elif initialization=='kaiming_uniform': nn.init.kaiming_uniform_(self.conv.weight, a=negative_slope, nonlinearity=non_linearity)
        elif initialization=='kaiming_normal': nn.init.kaiming_normal_(self.conv.weight, a=negative_slope, nonlinearity=non_linearity)
        else: sys.exit('\nError - Unexpected initialization method specified.')
        
        #Setup normalization layers/parameters
        if dataNormalize: self.std = nn.InstanceNorm2d(out_channels, affine=True) #default eps=1e-5 is oddly high...
        else: self.std = lambda inputs:inputs
        
    def forward(self, data):
        return self.std(self.act(self.conv(data)))

#Processing convolutional block for DLADS
class Conv_PC_DLADS(nn.Module):
    def __init__(self, numIn, numOut, non_linearity):
        super().__init__()
        self.conv0 = Conv_SC_DLADS(numIn, numOut, 3, 'same', non_linearity)
        self.conv1 = Conv_SC_DLADS(numOut, numOut, 3, 'same', non_linearity)
    
    def forward(self, data):
        data = self.conv0(data)
        return self.conv1(data)

#Input convolutional block for DLADS
class Conv_IN_DLADS(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()
        if standardizeInputData: self.std = CustomInstanceNorm()
        else: self.std = lambda inputs:inputs
        self.conv0 = Conv_SC_DLADS(numIn, numOut, 1, 'same', emDnAct)
        self.conv1 = Conv_PC_DLADS(numOut, numOut, inAct)
    
    def forward(self, data):
        data = self.std(data)
        data = self.conv0(data)
        return self.conv1(data)
        
#Downsampling convolutional block for DLADS
class Conv_DN_DLADS(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0 = Conv_SC_DLADS(numIn, numOut, 1, 'same', emDnAct)
        self.conv1 = Conv_PC_DLADS(numOut, numOut, dnAct)
    
    def forward(self, data):
        data = self.pool0(data)
        return self.conv1(self.conv0(data))

#Upsampling convolutional block for DLADS
class Conv_UP_DLADS(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.conv0 = Conv_SC_DLADS(numIn, numOut, 1, 'same', emUpAct)
        self.conv1 = Conv_PC_DLADS(numOut, numOut, upAct)
        
    def forward(self, data, skip):
        data = F.interpolate(data, size=skip.shape[2:], mode='nearest-exact')
        data = self.conv0(data)
        return self.conv1(data)

#DLADS model
class Model_DLADS(nn.Module):
    def __init__(self, numFilt, numChan):
        super().__init__()
        self.convIn1 = Conv_IN_DLADS(numChan, numFilt)
        #self.convPC0 = Conv_PC_DLADS(numFilt, numFilt, dnAct)
        self.convFn0 = Conv_SC_DLADS(numFilt, 1, 1, 'same', fnAct)
        
    def forward(self, data, pause=False):
        if padInputData:
            padHeight, padWidth = (-(-data.shape[-2]//16)*16)-data.shape[-2], (-(-data.shape[-1]//16)*16)-data.shape[-1]
            data = F.pad(data, (padWidth, 0, padHeight, 0), mode='constant')
        convDn0 = self.convIn1(data)
        #convPC0 = self.convPC0(convDn0)
        convOut = self.convFn0(convDn0)
        #if pause: Tracer() #If line enabled in compute loss, enables dropping into model; useful for debugging
        if padInputData: return convOut[:, :, padHeight:, padWidth:]
        else: return convOut

#Define DLADS model
class DLADS:
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
                ERD_NRMSE, ERD_SSIM = compareImages(squareRD, squareERD, np.min(squareRD), np.max(squareRD))
                
                ax = plt.subplot2grid((3,2), (vizSampleNum+1,0))
                im = ax.imshow(squareRD, aspect='auto', interpolation='none')
                ax.set_title('RD', fontsize=15)
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                cbar.formatter.set_powerlimits((0, 0))
                
                ax = plt.subplot2grid((3,2), (vizSampleNum+1,1))
                im = ax.imshow(squareERD, aspect='auto', interpolation='none')
                plotTitle = 'ERD\nNRMSE: ' + '{:.6f}'.format(round(ERD_NRMSE, 6)) + '; SSIM: ' + '{:.6f}'.format(round(ERD_SSIM, 6))
                ax.set_title(plotTitle, fontsize=15)
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                cbar.formatter.set_powerlimits((0, 0))
        
        plotTitle = 'EPOCH: ' + ('{:0'+str(maxEpochsPrecision)+'}').format(epoch)
        #plotTitle = 'EPOCH: ' + ('{:0'+str(maxEpochsPrecision)+'}').format(epoch) + ', LR: ' + '{:.1e}'.format(self.scheduler.get_last_lr()[0])
        #plotTitle = 'EPOCH: ' + ('{:0'+str(maxEpochsPrecision)+'}').format(epoch) + ', LR: ' + '{:.1e}'.format(self.learningRate)
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
        
    def computeLoss(self, data, label, trainFlag=False, epoch=None, step=None):
        
        #Zero network gradients
        if trainFlag: self.model.zero_grad()
        
        #Move data to device if not already there
        if not storeOnDevice: data, label = data.to(self.device), label.to(self.device)
        
        #Compute loss
        pred = self.model(data)
        loss = torch.mean(torch.abs(label-pred))
        
        #If network returned all 0s, then can use this line to enter model (in combination with other relevent commented line); useful for debugging
        #if torch.sum(pred).item() == 0: self.model(data, True)
        
        #Compute loss gradients and update network parameters
        if trainFlag: # and torch.sum(pred).item() != 0: #Skip any updates where predictions would crash training
            loss.backward()
            self.opt.step()
            if cosineScheduler == 'batch': self.scheduler.step(epoch+step/self.numTRN)
        
        return loss.item()
        
    def setOptimizer(self):
        if optimizer == 'AdamW': self.opt = optim.AdamW(self.model.parameters(), lr=self.learningRate)
        elif optimizer == 'NAdam': self.opt = optim.NAdam(self.model.parameters(), lr=self.learningRate)
        elif optimizer == 'Adam': self.opt = optim.Adam(self.model.parameters(), lr=self.learningRate)
        elif optimizer == 'RMSprop': self.opt = optim.RMSprop(self.model.parameters(), lr=self.learningRate)
        elif optimizer == 'SGD': self.opt = optim.SGD(self.model.parameters(), lr=self.learningRate)
        else: sys.exit('\nError - Unknown optimizer was specified.')
        
    def train(self):
        
        #Setup initial learning rate, optimizer, and scheduler
        self.learningRate = copy.deepcopy(learningRate)
        self.setOptimizer()
        if cosineScheduler != 'None': 
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, schedPeriod, schedMult, eta_min=schedMinLearningRate)
            if consineScheduler != 'batch' or cosineScheduler != 'epoch': sys.exit('\nError - The specified parameter for cosineScheduler was not recognized')
        
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
            self.loss_TRN.append(np.mean([self.computeLoss(data, label, True, epoch, step) for step, (data, label) in tqdm(enumerate(self.dataloader_TRN), total=self.numTRN, desc='TRN', leave=False, ascii=asciiFlag)]))
            
            #Update LR after complete epoch
            if cosineScheduler == 'epoch': self.scheduler.step()
            
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
        trainingTime = datetime.timedelta(seconds=(t1-t0))
        lines = ['Model Training Time: ' + str(trainingTime)]
        with open(dir_TrainingResults + 'trainingTime.txt', 'w') as f:
            for line in lines: _ = f.write(line+'\n')
        print(lines[0])
        
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
        