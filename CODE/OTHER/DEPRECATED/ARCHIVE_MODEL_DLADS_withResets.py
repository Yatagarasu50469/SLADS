#==================================================================
#TEMPORARY REFERENCE
#Some experimentation was performed with learning rate annealing, requiring the reset of model parameters and dynamic
#learning rate adjustments. As it was later determined (though corrected herein), these experiments were done with 
#a very shallow U-Net and therefore possibly resultant in generating invalid conclusions. Since it may be desired to 
#run similar tests in the future, a copy shall be included with at least one published version of the program. 
#==================================================================

#Downsampling convolutional block for DLADS
class Conv_Dn_DLADS(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.actDn = nn.LeakyReLU(0.2, inplace=True)
        self.convDn0 = nn.Conv2d(in_channels=numIn, out_channels=numOut, kernel_size=3, stride=1, padding='same', bias=False)
        self.convDn1 = nn.Conv2d(in_channels=numOut, out_channels=numOut, kernel_size=3, stride=1, padding='same', bias=False)
        #nn.init.kaiming_normal_(self.convDn0.weight, a=0.2, nonlinearity='leaky_relu')
        #nn.init.kaiming_normal_(self.convDn1.weight, a=0.2, nonlinearity='leaky_relu')
        nn.init.normal_(self.convDn0.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.convDn1.weight, mean=0.0, std=0.02)

    def forward(self, data):
        return self.actDn(self.convDn1(self.actDn(self.convDn0(data))))

#Upsampling convolutional block for DLADS
class Conv_Up_DLADS(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.actUp = nn.ReLU(inplace=True)
        self.convUp0 = nn.Conv2d(in_channels=numIn, out_channels=numIn, kernel_size=2, stride=1, padding='same', bias=False)
        self.convUp1 = nn.Conv2d(in_channels=numIn+numOut, out_channels=numOut, kernel_size=3, stride=1, padding='same', bias=False)
        self.convUp2 = nn.Conv2d(in_channels=numOut, out_channels=numOut, kernel_size=3, stride=1, padding='same', bias=False)
        #nn.init.kaiming_normal_(self.convUp0.weight, nonlinearity='relu')
        #nn.init.kaiming_normal_(self.convUp1.weight, nonlinearity='relu')
        #nn.init.kaiming_normal_(self.convUp2.weight, nonlinearity='relu')
        nn.init.normal_(self.convUp1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.convUp1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.convUp2.weight, mean=0.0, std=0.02)

    def forward(self, data, skip):
        data = functional.interpolate(data, size=skip.size()[2:], mode='nearest')
        data = self.actUp(self.convUp0(data))
        data = torch.cat([data, skip], 1) 
        return self.actUp(self.convUp2(self.actUp(self.convUp1(data))))
    
#DLADS model
class Model_DLADS(nn.Module):
    def __init__(self, numFilt, numChan):
        super().__init__()
        self.poolDn = nn.MaxPool2d(kernel_size=2)
        self.convDn0 = Conv_Dn_DLADS(numChan, numFilt)
        self.convDn1 = Conv_Dn_DLADS(numFilt, numFilt*2)
        self.convDn2 = Conv_Dn_DLADS(numFilt*2, numFilt*4)
        self.convDn3 = Conv_Dn_DLADS(numFilt*4, numFilt*8)
        self.convDn4 = Conv_Dn_DLADS(numFilt*8, numFilt*16)
        self.convUp3 = Conv_Up_DLADS(numFilt*16, numFilt*8)
        self.convUp2 = Conv_Up_DLADS(numFilt*8, numFilt*4)
        self.convUp1 = Conv_Up_DLADS(numFilt*4, numFilt*2)
        self.convUp0 = Conv_Up_DLADS(numFilt*2, numFilt)
        self.convFin = nn.Conv2d(in_channels=numFilt, out_channels=1, kernel_size=3, stride=1, padding='same', bias=False)
        #nn.init.kaiming_normal_(self.convFin.weight, nonlinearity='linear')
        nn.init.normal_(self.convFin.weight, mean=0.0, std=0.02)
        
    def forward(self, data):
        convDn0 = self.convDn0(data)
        convDn1 = self.convDn1(self.poolDn(convDn0))
        convDn2 = self.convDn2(self.poolDn(convDn1))
        convDn3 = self.convDn3(self.poolDn(convDn2))
        convDn4 = self.convDn4(self.poolDn(convDn3))
        convUp3 = self.convUp3(convDn4, convDn3)
        convUp2 = self.convUp2(convUp3, convDn2)
        convUp1 = self.convUp1(convUp2, convDn1)
        convUp0 = self.convUp0(convUp1, convDn0)
        return self.convFin(convUp0)

#Random rotation transform using a discrete set of angles; enusres RandomCrop captures data (and doesn't add zeros) in original input FOV
class RandomDiscreteRotate:
    def __init__(self): pass
    def __call__(self, image): 
        angle = torch.randint(3, (1,))
        if angle == 0: return image.clone()
        elif angle == 1: return torch.rot90(image, k=1, dims=(-2, -1))
        else: return torch.rot90(image, k=3, dims=(-2, -1))
    
#Perform augmentation and setup for DLADS data processing
class DataPreprocessing_DLADS(Dataset):
    def __init__(self, inputs, labels, device, augmentFlag):
        super().__init__()
        self.noAugmentFlag = not augmentFlag
        if augmentFlag: 
            self.transform = transforms.Compose([
                v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5), RandomDiscreteRotate()
                #v2.RandomResizedCrop(size=(128, 128), scale=(0.01, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.NEAREST)
                #v2.RandomCrop(size=(128, 128), pad_if_needed=True)
            ])
        
        if storeOnDevice:
            self.data_Inputs = [torch.from_numpy(item).float().to(device) for item in inputs]
            self.data_Labels = [torch.from_numpy(item).float().to(device) for item in labels]
        else: 
            self.data_Inputs = [torch.from_numpy(item).float() for item in inputs]
            self.data_Labels = [torch.from_numpy(item).float() for item in labels]
        self.channelSplit = [self.data_Inputs[0].size()[0]]
        
    def __getitem__(self, index):
        if self.noAugmentFlag: return self.data_Inputs[index], self.data_Labels[index]
        return torch.tensor_split(self.transform(torch.cat([self.data_Inputs[index], self.data_Labels[index]], 0)), self.channelSplit, 0)

    def __len__(self):
        return len(self.data_Inputs)

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
        if (self.device.type == 'cuda') and (len(local_gpus)>1): 
            self.model = nn.DataParallel(self.model, local_gpus)
            self.batchsize_TRN, self.batchsize_VAL = len(local_gpus), len(local_gpus)
        else:
            self.batchsize_TRN, self.batchsize_VAL = 1, 1
        self.model.to(self.device)
        
        #Upcoming compile function should improve speed even further; not currently working 2.2.0
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
            vizSamples, vizSampleData = None, None
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
    def visualizeTraining(self, epoch_GBL, epoch_LCL):
        
        if self.valFlag: f = plt.figure(figsize=(15,15))
        else: f = plt.figure(figsize=(15,8))
        f.subplots_adjust(bottom=0.05, left=0.10, right=0.90, top=0.88, wspace=0.20, hspace=0.30)
        plt.rcParams['font.size'] = 10
        
        if self.valFlag: ax = plt.subplot2grid((3,1), (0,0))
        else: ax = plt.subplot2grid((1,1), (0,0))
        if earlyStoppingMean: 
            ax.plot(np.ma.masked_invalid(self.meanLoss_HST_GBL), label='HST Mean', alpha=0.7, color='black')
            ax.plot(np.ma.masked_invalid(self.meanLoss_CUR_GBL), label='CUR Mean', alpha=0.7, color='green')
        ax.plot(self.loss_TRN_GBL, label='TRN', alpha=0.7, color='red')
        if self.valFlag: ax.plot(self.loss_VAL_GBL, label='VAL', alpha=0.7, color='blue')
        ax.set_yscale('log')
        ymin, ymax = ax.get_ylim()
        ax.vlines(x=self.restoreEpochs, ymin=ymin, ymax=ymax, label='Model Restore', alpha=0.7, color='purple', ls='--')
        ax.set_ylim(ymin, ymax)
        ax.legend(loc='upper right', fontsize=10)
        
        #Local losses
        #if self.valFlag: ax = plt.subplot2grid((3,1), (0,0))
        #else: ax = plt.subplot2grid((1,1), (0,0))
        #if earlyStoppingMean: 
        #    ax.plot(np.ma.masked_invalid(self.meanLoss_HST_LCL), label='HST Mean', alpha=0.7, color='black')
        #    ax.plot(np.ma.masked_invalid(self.meanLoss_CUR_LCL), label='CUR Mean', alpha=0.7, color='green')
        #ax.plot(self.loss_TRN_LCL, label='TRN', alpha=0.7, color='red')
        #if self.valFlag: ax.plot(self.loss_VAL_LCL, label='VAL', alpha=0.7, color='blue')
        #ax.set_yscale('log')
        #ax.legend(loc='upper right', fontsize=10)
        
        if self.valFlag: 
            for vizSampleNum in range(0, self.numViz): 
            
                if not storeOnDevice: input = self.inputs_Viz[vizSampleNum].to(self.device)
                input = self.inputs_Viz[vizSampleNum]
                
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
        
        plotTitle = 'EPOCH - GBL: ' + ('{:0'+str(maxEpochsPrecision)+'}').format(epoch_GBL) + ', LCL: '  + ('{:0'+str(maxEpochsPrecision)+'}').format(epoch_LCL) + ', LR: ' + '{:.1e}'.format(self.learningRate)
        plotTitle += '\nPAT: ' + ('{:0'+str(maxPatiencePrecision)+'}').format(self.patienceCounter) + '/' + str(maxPatience) + ", STG: " + ('{:0'+str(maxStagnationPrecision)+'}').format(self.stagnationCounter) + "/" + str(maxStagnation)
        plotTitle += '\nLOSS - TRN: ' + '{:.6f}'.format(round(self.loss_TRN_LCL[-1], 6)) + ', '
        if self.valFlag: plotTitle += 'VAL: ' + '{:.6f}'.format(round(self.loss_VAL_LCL[-1], 6)) + ', '
        else: plotTitle += 'VAL: N/A,   '
        plotTitle += 'BST: ' + '{:.6f}'.format(round(self.loss_BST, 6)) +' at Epoch: ' + ('{:0'+str(maxEpochsPrecision)+'}').format(self.epoch_BST)
        if earlyStoppingMean and (len(self.meanLoss_HST_LCL) > 0): plotTitle += '\nMean Loss - HST: '+ '{:.6f}'.format(round(self.meanLoss_HST_LCL[-1], 6)) +', CUR: ' + '{:.6f}'.format(round(self.meanLoss_CUR_LCL[-1], 6))
        else: plotTitle += '\nMean Loss - Historical: N/A,   Current: N/A    '
        plt.suptitle(plotTitle, fontsize=20, fontweight='bold')
        
        #Save resulting plot
        f.savefig(dir_TrainingModelResults + 'epoch_' +str(epoch_GBL) + '.tiff')
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
        
    def train(self):
        
        #Setup optimizer
        learningRateIndex = 0
        self.learningRate = learningRates[learningRateIndex]
        self.setOptimizer()
        
        #Backup initial model/optimizer before training in case a complete restart is needed
        self.model_INT = copy.deepcopy(self.model.state_dict())
        self.opt_INT = copy.deepcopy(self.opt.state_dict())
        self.rng_INT = copy.deepcopy(torch.get_rng_state())
        
        #Setup storage for losses/events
        self.loss_TRN_LCL, self.loss_VAL_LCL, self.meanLoss_HST_LCL, self.meanLoss_CUR_LCL = [], [], [], []
        self.loss_TRN_GBL, self.loss_VAL_GBL, self.meanLoss_HST_GBL, self.meanLoss_CUR_GBL = [], [], [], []
        self.restoreEpochs = []
        
        #Setup variables for stopping critera and learning rate decay
        visualizeLastEpoch, self.loss_BST, self.epoch_BST, endTraining, epoch_LCL, patienceLost, modelStagnation = maxEpochs-1, np.inf, -1, False, 0, False, False
        
        #If using mean early stopping, then the minimum number of epochs is double sepEpochs, otherwise set to zero
        if earlyStoppingMean: minEpochs = sepEpochs*2
        else: minEpochs = 0
        
        #Create progress bar
        trainingBar = tqdm(range(maxEpochs), desc='Epochs', leave=True, ascii=asciiFlag)
        
        #Perform model training
        t0 = time.perf_counter()
        for epoch_GBL in trainingBar:
            
            #Compute losses over the training dataset
            _ = self.model.train(True)
            self.loss_TRN_GBL.append(np.mean([self.computeLoss(data, label, True) for data, label in tqdm(self.dataloader_TRN, total=self.numTRN, desc='TRN Batches', leave=False, ascii=asciiFlag)]))
            self.loss_TRN_LCL.append(self.loss_TRN_GBL[-1])
            
            #Compute losses over the validation dataset
            _ = self.model.train(False)
            if self.valFlag: 
                with torch.inference_mode(): 
                    self.loss_VAL_GBL.append(np.mean([self.computeLoss(data, label, False) for data, label in tqdm(self.dataloader_VAL, total=self.numVAL, desc='VAL Batches', leave=False, ascii=asciiFlag)]))
                    self.loss_VAL_LCL.append(self.loss_VAL_GBL[-1])
            
            #Update the current loss used for training status evaluation
            if self.valFlag: loss_CUR = self.loss_VAL_GBL[-1]
            else: loss_CUR = self.loss_TRN_GBL[-1]
            
            #If this model is the best performing, update the best model parameters and reset applicable variables; if not an improvement and not using mean early stopping, increase the patience counter
            if (loss_CUR < self.loss_BST): self.model_BST, self.opt_BST, self.rng_BST, self.loss_BST, self.epoch_BST, self.patienceCounter, self.stagnationCounter = copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.opt.state_dict()), copy.deepcopy(torch.get_rng_state()), loss_CUR, epoch_GBL, 0, 0
            elif (epoch_LCL >= minEpochs) and (not earlyStoppingMean): self.patienceCounter += 1
            
            #If using mean early stopping, then evaluate/track training status by comparing historical and current moving averages; if below minEpochs, append nan as placeholder
            if (epoch_LCL >= minEpochs) and earlyStoppingMean:
                if self.valFlag: 
                    meanLoss_HST = np.mean(self.loss_VAL_LCL[-minEpochs:-sepEpochs])
                    meanLoss_CUR = np.mean(self.loss_VAL_LCL[-sepEpochs:])
                else: 
                    meanLoss_HST = np.mean(self.loss_TRN_LCL[-minEpochs:-sepEpochs])
                    meanLoss_CUR = np.mean(self.loss_TRN_LCL[-sepEpochs:])
                self.meanLoss_HST_LCL.append(meanLoss_HST)
                self.meanLoss_CUR_LCL.append(meanLoss_CUR)
                self.meanLoss_HST_GBL.append(meanLoss_HST)
                self.meanLoss_CUR_GBL.append(meanLoss_CUR)
                if meanLoss_CUR >= meanLoss_HST: self.patienceCounter += 1
                elif self.patienceCounter > 0: self.patienceCounter -= 1
            elif earlyStoppingMean: 
                self.meanLoss_HST_LCL.append(np.nan)
                self.meanLoss_CUR_LCL.append(np.nan)
                self.meanLoss_HST_GBL.append(np.nan)
                self.meanLoss_CUR_GBL.append(np.nan)
            
            #Evaluate/track training stagnation
            if epoch_LCL >= 1: 
                if self.valFlag and (self.loss_VAL_LCL[-1] == self.loss_VAL_LCL[-2]): self.stagnationCounter +=1
                elif (self.loss_TRN_LCL[-1] == self.loss_TRN_LCL[-2]): self.stagnationCounter +=1
                else: self.stagnationCounter = 0
            
            #Evaluate stopping criteria
            if (self.patienceCounter >= maxPatience): patienceLost = True
            if (self.stagnationCounter >= maxStagnation): modelStagnation = True
            if (patienceLost or modelStagnation) and (learningRateIndex >= len(learningRates)-1): endTraining = True
            
            #Update progress bar with epoch data
            progBarString = "PAT: " + ('{:0'+str(maxPatiencePrecision)+'}').format(self.patienceCounter) + "/" + str(maxPatience)
            progBarString += ", STG: " + ('{:0'+str(maxStagnationPrecision)+'}').format(self.stagnationCounter) + "/" + str(maxStagnation)
            progBarString += ", LOSS - TRN: " + '{:.6f}'.format(round(self.loss_TRN_LCL[-1], 6))
            if self.valFlag: progBarString += ", VAL: " + '{:.6f}'.format(round(self.loss_VAL_LCL[-1], 6))
            trainingBar.set_postfix_str(progBarString)
            trainingBar.refresh()
            
            #Perform visualization(s)
            if trainingProgressionVisuals and ((epoch_LCL == 0) or (epoch_GBL == 0) or (epoch_GBL%trainingVizSteps == 0) or (epoch_GBL == self.epoch_BST) or (epoch_GBL == visualizeLastEpoch) or endTraining or modelStagnation or patienceLost): self.visualizeTraining(epoch_GBL, epoch_LCL)
            
            #As applicable, terminate the training loop, increment counter for epochs conducted with the current learning rate, reset criteria/variables as needed/configured
            if endTraining: 
                break
            elif not (patienceLost or modelStagnation):
                epoch_LCL += 1
            else: 
                self.restoreEpochs.append(epoch_GBL)
                epoch_LCL, self.patienceCounter, self.stagnationCounter, patienceLost, modelStagnation = 0, 0, 0, False, False
                self.loss_TRN_LCL, self.loss_VAL_LCL, self.meanLoss_HST_LCL, self.meanLoss_CUR_LCL = [], [], [], []
                if (self.epoch_BST <= 0): 
                    self.model.load_state_dict(self.model_INT)
                    #self.opt.load_state_dict(self.opt_INT)
                    torch.set_rng_state(self.rng_INT)
                else: 
                    self.model.load_state_dict(self.model_BST)
                    #self.opt.load_state_dict(self.opt_BST)
                    torch.set_rng_state(self.rng_BST)
                learningRateIndex += 1
                self.learningRate = learningRates[learningRateIndex]
                #for param_group in self.opt.param_groups: param_group['lr'] = self.learningRate
                self.setOptimizer()
        
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
        history = np.vstack([np.array(range(0, epoch_GBL+1)), self.loss_TRN_GBL])
        if self.valFlag: history = np.vstack([history, self.loss_VAL_GBL])
        pd.DataFrame(history.T, columns=['Epoch','Loss_TRN', 'Loss_VAL']).to_csv(dir_TrainingResults+'trainingHistory.csv', index=False)
        
    def predict(self, input):
        if len(input.shape) == 3: input = np.expand_dims(input, 0)
        input = torch.from_numpy(input).float()
        input = input.to(self.device)
        with torch.inference_mode(): squareERDs = np.moveaxis(self.model(input).detach().cpu().numpy(), 1, 0)[0]
        return squareERDs
        