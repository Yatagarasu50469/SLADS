#==================================================================
#MODEL: GLANDS
#==================================================================

#Downsampling convolutional block for GLANDS
class Conv_Dn_GLANDS(nn.Module):

    def __init__(self, numIn, numOut):
        super().__init__()

    def forward(self):
        return 

#Upsampling convolutional block for GLANDS
class Conv_Up_GLANDS(nn.Module):
    def __init__(self, numIn, numOut):
        super().__init__()

    def forward(self):
        return 
    
class Model_GLANDS_REC(nn.Module):
    def __init__(self, numStartFilters, numChannels):
        super().__init__()
        
    def forward(self):
        return 
        
class Model_GLANDS_ACT(nn.Module):
    def __init__(self, numStartFilters, numChannels):
        super().__init__()
        
    def forward(self):
        return 
        
class Model_GLANDS_DIS(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self):
        return 

#Perform augmentation and setup for GLANDS data processing
class DataPreprocessing_GLANDS(Dataset):
    def __init__(self, inputs, labels, augmentFlag=False):
        super().__init__()
        if augmentFlag:
            self.transform = transforms.Compose([
                v2.RandomResizedCrop(size=(64, 64), scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), antialias=True), #https://arxiv.org/pdf/1409.4842.pdf
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=False)
            ])
        else: 
            self.transform = transforms.Compose([
                v2.ToDtype(torch.float32, scale=False)
            ])
            
        #If can call v2.ToDtype(torch.float32, scale=False) first, then can do so here and then only do the remaining transformations in __getitem__....
        self.data_Inputs = [torch.from_numpy(item) for item in inputs]
        self.data_Labels = [torch.from_numpy(item) for item in labels]
        
    def __getitem__(self, index):
        return self.transform(self.data_Inputs[index], self.data_Labels[index])

    def __len__(self):
        return len(self.data_Inputs)
        


#Extract and prepare training data
inputs_TRN = []
for sampleData in tqdm(trainingSampleData, desc = 'Training Data Setup', leave=True, ascii=asciiFlag):
    for chanNum in range(0, sampleData.numChannels):
        inputs_TRN.append(sampleData.squareChanImages[chanNum])
data_TRN = DataPreprocessing_GLANDS(inputs_TRN, augTrainData)
dataloader_TRN = DataLoader(data_TRN, batch_size=batchsize_TRN, num_workers=0, shuffle=True)

#If there is not a validation set then indicate such, otherwise extract and prepare validation data
if len(validationSampleData)<=0: 
    valFlag = False
    vizSampleData = None
else:
    valFlag = True
    inputs_VAL = []
    for sampleData in tqdm(validationSampleData, desc = 'Validation Data Setup', leave=True, ascii=asciiFlag):
        for chanNum in range(0, sampleData.numChannels):
            inputs_VAL.append(sampleData.squareChanImages[chanNum])
    data_VAL = DataPreprocessing_GLANDS(inputs_VAL, False)
    if batchsize_VAL == -1: dataloader_VAL = DataLoader(data_VAL, batch_size=len(inputs_VAL), num_workers=0, shuffle=False)
    else: dataloader_VAL = DataLoader(data_VAL, batch_size=batchsize_VAL, num_workers=0, shuffle=False)
    
#Create networks on specified computational device(s)
device = torch.device("cuda:"+str(gpus[0]) if numGPUs > 0 else "cpu")
REC = Model_GLANDS_REC(numStartFilters, len(inputChannels)).to(device)
ACT = Model_GLANDS_ACT(numStartFilters, len(inputChannels)+1).to(device)
DIS = Model_GLANDS_DIS(numStartFilters, len(inputChannels)+1).to(device)

#Configure for multiple GPUs as specified
if (device.type == 'cuda') and (numGPUs > 1): 
    REC = nn.DataParallel(REC, gpus)
    ACT = nn.DataParallel(ACT, gpus)
    DIS = nn.DataParallel(DIS, gpus)

#Setup optimizers
if optimizer == 'AdamW': 
    opt_REC = optim.AdamW(REC.parameters(), lr=learningRate, betas=(beta1, beta2))
    opt_ACT = optim.AdamW(ACT.parameters(), lr=learningRate, betas=(beta1, beta2))
    opt_DIS = optim.AdamW(DIS.parameters(), lr=learningRate, betas=(beta1, beta2))
elif optimizer == 'Adam': 
    opt_REC = optim.Adam(REC.parameters(), lr=learningRate, betas=(beta1, beta2))
    opt_ACT = optim.Adam(ACT.parameters(), lr=learningRate, betas=(beta1, beta2))
    opt_DIS = optim.Adam(DIS.parameters(), lr=learningRate, betas=(beta1, beta2))
elif optimizer == 'Nadam': 
    opt_REC = optim.NAdam(REC.parameters(), lr=learningRate, betas=(beta1, beta2))
    opt_ACT = optim.NAdam(ACT.parameters(), lr=learningRate, betas=(beta1, beta2))
    opt_DIS = optim.NAdam(DIS.parameters(), lr=learningRate, betas=(beta1, beta2))
elif optimizer == 'SGD': 
    opt_REC = optim.SGD(REC.parameters(), lr=learningRate)
    opt_ACT = optim.SGD(ACT.parameters(), lr=learningRate)
    opt_DIS = optim.SGD(DIS.parameters(), lr=learningRate)
elif optimizer == 'RMSProp': 
    opt_REC = optim.RMSprop(REC.parameters(), lr=learningRate)
    opt_ACT = optim.RMSprop(ACT.parameters(), lr=learningRate)
    opt_DIS = optim.RMSprop(DIS.parameters(), lr=learningRate)

#Training loop for GLANDS
def train_GLANDS():
    t0 = time.time()
    
    #Perform training/validation processes here
    
    t1 = time.time()
    trainingTime = datetime.timedelta(seconds=(t1-t0))
    print('Model Training Time: ' + str(trainingTime))
    
    #Save parameters of best models
    torch.save(best_REC, dir_TrainingResults + modelName + 'Reconstructor')
    torch.save(best_ACT, dir_TrainingResults + modelName + 'Actor')
    torch.save(best_DIS, dir_TrainingResults + modelName + 'Discriminator')
    
    #Save training history...
    #pd.DataFrame(history.history).to_csv(dir_TrainingResults+'history.csv')
    
    
    
#Training loop for GLANDS
def train_GLANDS():
    t0 = time.time()
    
    #Perform training/validation processes here
    
    t1 = time.time()
    trainingTime = datetime.timedelta(seconds=(t1-t0))
    print('Model Training Time: ' + str(trainingTime))
    
    #Save parameters of best models
    torch.save(best_REC, dir_TrainingResults + modelName + 'Reconstructor')
    torch.save(best_ACT, dir_TrainingResults + modelName + 'Actor')
    torch.save(best_DIS, dir_TrainingResults + modelName + 'Discriminator')
    
    #Save training history...
    #pd.DataFrame(history.history).to_csv(dir_TrainingResults+'history.csv')