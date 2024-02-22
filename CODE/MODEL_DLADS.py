#==================================================================
#MODEL: DLADS
#==================================================================

#Downsampling convolutional block for DLADS
class Conv_Dn_DLADS(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.actDn = nn.LeakyReLU(0.2, inplace=True)
        self.convDn0 = nn.Conv2d(in_channels=numIn, out_channels=numOut, kernel_size=1, stride=1, padding='same', bias=True)
        self.convDn1 = nn.Conv2d(in_channels=numOut, out_channels=numOut, kernel_size=3, stride=1, padding='same', bias=True)
        nn.init.xavier_uniform_(self.convDn0.weight)
        nn.init.xavier_uniform_(self.convDn1.weight)

    def forward(self, data):
        return self.actDn(self.convDn1(self.actDn(self.convDn0(data))))

#Upsampling convolutional block for DLADS
class Conv_Up_DLADS(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()
        self.actUp = nn.ReLU(inplace=True)
        self.convUp0 = nn.Conv2d(in_channels=numIn, out_channels=numIn, kernel_size=2, stride=1, padding='same', bias=True)
        self.convUp1 = nn.Conv2d(in_channels=numIn+numOut, out_channels=numOut, kernel_size=3, stride=1, padding='same', bias=True)
        self.convUp2 = nn.Conv2d(in_channels=numOut, out_channels=numOut, kernel_size=1, stride=1, padding='same', bias=True)
        nn.init.xavier_uniform_(self.convUp0.weight)
        nn.init.xavier_uniform_(self.convUp1.weight)
        nn.init.xavier_uniform_(self.convUp2.weight)

    def forward(self, data, skip):
        data = functional.interpolate(data, size=skip.size()[2:], mode='nearest')
        data = self.actUp(self.convUp0(data))
        data = torch.cat([data, skip], 1) 
        return self.actUp(self.convUp2(self.actUp(self.convUp1(data))))
    
#DLADS model
class Model_DLADS(nn.Module):
    def __init__(self, numFilt, numChan):
        super().__init__()
        self.actFn = nn.ReLU(inplace=True)
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
        self.convFin = nn.Conv2d(in_channels=numFilt, out_channels=1, kernel_size=3, stride=1, padding='same', bias=True)
        nn.init.xavier_uniform_(self.convFin.weight)
        
    def forward(self, data):
        convDn0 = self.convDn0(data)
        convDn1 = self.convDn1(self.poolDn(convDn0))
        convDn2 = self.convDn2(self.poolDn(convDn1))
        convDn3 = self.convDn3(self.poolDn(convDn2))
        convDn4 = self.convDn4(self.poolDn(convDn3))
        convUp3 = self.convUp3(convDn4, convDn3)
        convUp2 = self.convUp2(convDn3, convDn2)
        convUp1 = self.convUp1(convDn2, convDn1)
        convUp0 = self.convUp0(convDn1, convDn0)
        return self.actFn(self.convFin(convUp0))

#Perform augmentation and setup for DLADS data processing
class DataPreprocessing_DLADS(Dataset):
    def __init__(self, inputs, labels, device, augmentFlag):
        super().__init__()
        self.noAugmentFlag = not augmentFlag
        if augmentFlag: 
            self.transform = transforms.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomAffine(degrees=45, translate=(0.25, 0.25), interpolation=transforms.InterpolationMode.NEAREST, fill=0) 
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
        
            #Unpack and load the stored model 
            modelPath = modelDirectory + modelName
            with multivolumefile.open(modelPath + os.path.sep + modelName + '.7z', mode='rb') as modelArchive:
                with py7zr.SevenZipFile(modelArchive, 'r') as archive:
                    archive.extract(modelDirectory)
            _ = self.model.load_state_dict(torch.load(modelPath + '.pt'))
            _ = self.model.train(False)
            os.remove(modelPath + '.pt')
        
        #Configure CPU/GPU computation environment; allocate model location and batch sizes accordingly
        #DLADS augmentation in the last publication didn't use random cropping to synchronize dimensions, meaning batch sizes were fixed to 1
        #However, parallel operations with pytorch neccessitate identical dimensions across any given batch
        #For at least one release that uses pytorch, disable parallel training, to match with tensorflow implementation
        #In v0.10.1+, add randomcrop to augmentation process and evaluate parallel training potential
        self.device = torch.device(f"cuda:{local_gpus[-1]}" if len(local_gpus) > 0 else "cpu")
        #if (self.device.type == 'cuda') and (len(local_gpus)>1): 
        #    self.model = nn.DataParallel(self.model, local_gpus)
        #    self.batchsize_TRN, self.batchsize_VAL = len(local_gpus), len(local_gpus)
        #else:
        #    self.batchsize_TRN, self.batchsize_VAL = 1, 1
        self.batchsize_TRN, self.batchsize_VAL = 1, 1
        self.model.to(self.device)
        
        #Upcoming compile function should improve speed even further; not currently working 2.2.0
        #https://github.com/pytorch/pytorch/pull/119750
        #self.model = torch.compile(self.model, dynamic=True, mode="reduce-overhead")
        
        #If training, setup optimizers
        if trainFlag:
            if optimizer == 'AdamW': self.opt = optim.AdamW(self.model.parameters(), lr=learningRate, betas=(beta1, beta2))
            elif optimizer == 'Adam': self.opt = optim.Adam(self.model.parameters(), lr=learningRate, betas=(beta1, beta2))
            elif optimizer == 'Nadam': self.opt = optim.NAdam(self.model.parameters(), lr=learningRate, betas=(beta1, beta2))
            elif optimizer == 'SGD': self.opt = optim.SGD(self.model.parameters(), lr=learningRate)
            elif optimizer == 'RMSProp': self.opt = optim.RMSprop(self.model.parameters(), lr=learningRate)
    
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
    def visualizeTraining(self, epoch):
        
        if self.valFlag: f = plt.figure(figsize=(15,15))
        else: f = plt.figure(figsize=(15,5))
        f.subplots_adjust(top = 0.90)
        f.subplots_adjust(wspace=0.2, hspace=0.2)
        
        if self.valFlag: ax = plt.subplot2grid((3,1), (0,0))
        else: ax = plt.subplot2grid((1,1), (0,0))
        ax.plot(self.loss_Trn, label='Training')
        if self.valFlag: ax.plot(self.loss_Val, label='Validation')
        ax.legend(loc='upper right', fontsize=14)
        ax.set_yscale('log')
        
        if self.valFlag: 
            for vizSampleNum in range(0, self.numViz): 
            
                if not storeOnDevice: input = self.inputs_Viz[vizSampleNum].to(self.device)
                input = self.inputs_Viz[vizSampleNum]
                
                squareRD = self.labels_Viz[vizSampleNum]
                with torch.inference_mode(): squareERD = torch.mean(self.model(input), 0).detach().cpu().numpy()[0]
                maxRangeValue = np.max([squareRD, squareERD])
                ERD_SSIM = compare_ssim(squareRD, squareERD, data_range=maxRangeValue)
                ERD_PSNR = compare_psnr(squareRD, squareERD, data_range=maxRangeValue)

                ax = plt.subplot2grid((3,2), (vizSampleNum+1,0))
                im = ax.imshow(squareRD, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('RD', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,2), (vizSampleNum+1,1))
                im = ax.imshow(squareERD, aspect='auto', vmin=0, interpolation='none')
                plotTitle = 'ERD - PSNR: ' + '{:.4f}'.format(round(ERD_PSNR, 4)) + ' SSIM: ' + '{:.4f}'.format(round(ERD_SSIM, 4))
                ax.set_title(plotTitle, fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            
            plotTitle = 'Epoch: '+str(epoch)+'     Patience: '+str(self.patience)+'/'+str(maxPatience)
            plotTitle += '\nBest Loss: '+ '{:.5f}'.format(round(self.bestLoss, 5)) +' at Epoch: '+str(self.bestEpoch)
            plotTitle += '\nLoss - TRN: ' + '{:.5f}'.format(round(self.loss_Trn[-1], 5))
            plotTitle += '     VAL: ' + '{:.5f}'.format(round(self.loss_Val[-1], 5))
            plt.suptitle(plotTitle, fontsize=20, fontweight='bold')
            
            #Save resulting plot
            f.savefig(dir_TrainingModelResults + 'epoch_' +str(epoch) + '.tiff', bbox_inches='tight')
            plt.close(f)
        
    def computeLoss(self, data, label, trainFlag=False):
        
        #Zero network gradients
        if trainFlag: self.model.zero_grad()
        
        #Compute MAE
        if not storeOnDevice: data, label = data.to(self.device), label.to(self.device)
        loss = torch.mean(torch.abs(self.model(data)-label))
        
        #Compute loss gradients and update network parameters
        if trainFlag:
            loss.backward()
            self.opt.step()
        
        return loss.item()
        
    def train(self):
        
        #Setup storage for losses
        self.loss_Trn, self.loss_Val = [], []

        #Setup variables for early stopping critera
        bestModel, self.bestLoss, self.bestEpoch, self.patience, endTraining = None, np.inf, -1, 0, False

        #Create progress bar
        trainingBar = tqdm(range(numEpochs), desc='Epochs', leave=True, ascii=asciiFlag)
        
        #Perform model training
        t0 = time.time()
        for epoch in trainingBar:
            
            #Compute losses over the training dataset
            _ = self.model.train(True)
            self.loss_Trn.append(np.mean([self.computeLoss(data, label, True) for data, label in tqdm(self.dataloader_TRN, total=self.numTRN, desc='TRN Batches', leave=False, ascii=asciiFlag)]))
            
            #Compute losses over the validation dataset
            _ = self.model.train(False)
            if self.valFlag: 
                with torch.inference_mode(): 
                    self.loss_Val.append(np.mean([self.computeLoss(data, label, False) for data, label in tqdm(self.dataloader_VAL, total=self.numVAL, desc='VAL Batches', leave=False, ascii=asciiFlag)]))
            
            #If applicable: update best model parameters or increase patience
            if (epoch >= minimumEpochs):
                
                if self.valFlag: currLoss = self.loss_Val[-1]
                else: currLoss = self.loss_Trn[-1]
                
                if (currLoss <= self.bestLoss): bestModel, self.bestLoss, self.bestEpoch, self.patience = copy.deepcopy(self.model.state_dict()), currLoss, epoch, 0
                else: self.patience += 1
            
            #Update progress bar with epoch data
            progBarString = "LOSS -" 
            progBarString += " TRN: " + '{:.5f}'.format(round(self.loss_Trn[-1], 5))
            if self.valFlag: progBarString += " VAL: " + '{:.5f}'.format(round(self.loss_Val[-1], 5))
            
            #Exit training if early stopping criteria is triggered
            if self.patience >= maxPatience: endTraining = True
            
            #Perform visualization(s) if applicable
            if trainingProgressionVisuals and ((epoch == 0) or (epoch % trainingVizSteps == 0) or endTraining or (self.bestEpoch == epoch)): self.visualizeTraining(epoch)
        
            #If training should be terminated, exit the loop
            if endTraining: break
        
        t1 = time.time()
        trainingTime = datetime.timedelta(seconds=(t1-t0))
        
        lines = ['Model Training Time: ' + str(trainingTime)]
        with open(dir_TrainingResults + 'trainingTime.txt', 'w') as f:
            for line in lines: _ = f.write(line+'\n')
        print(lines[0])
        
        #Strip out any parallel 'module' references from the model definition
        bestModel = {key.replace("module.", ""): value for key, value in bestModel.items()}
        
        #Store the model across multiple 100 Mb files to bypass Github file size limits
        modelPath = dir_TrainingResults + modelName
        torch.save(bestModel, modelPath + '.pt')
        if os.path.exists(modelPath): shutil.rmtree(modelPath)
        os.makedirs(modelPath)
        with multivolumefile.open(modelPath + os.path.sep + modelName + '.7z', mode='wb', volume=104857600) as modelArchive:
            with py7zr.SevenZipFile(modelArchive, 'w') as archive:
                archive.writeall(modelPath + '.pt', modelName + '.pt')
        os.remove(modelPath + '.pt')
        
        #Save training history
        history = np.vstack([np.array(range(0, epoch+1)), self.loss_Trn])
        if self.valFlag: history = np.vstack([history, self.loss_Val])
        pd.DataFrame(history.T, columns=['Epoch','Loss_TRN', 'Loss_VAL']).to_csv(dir_TrainingResults+'trainingHistory.csv', index=False)
        
    def predict(self, input):
        if len(input.shape) == 3: input = np.expand_dims(input, 0)
        input = torch.from_numpy(input).float()
        input = input.to(self.device)
        with torch.inference_mode(): squareERDs = np.moveaxis(self.model(input).detach().cpu().numpy(), 1, 0)[0]
        return squareERDs
        