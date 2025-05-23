#==================================================================
#MODEL: DLADS-TF-Deprecated
#TensorFlow DLADS implementation, modified from v0.9.5 - Please do not use unless you know what you are doing.
#Need to set manualSeed to -1 in configuration, as a deterministic implementation of ResizeNearestNeighborGrad is not available. 
#
#NOTE(S): DIFFERENCES WITH RESPECT TO ORIGINAL v0.9.5 IMPLEMENTATION
#
# NOW DISABLED for training only: 1) DLADS originally allowed training and validation data to be shuffled across epochs, this has now been prevented.
#
# NOW DISABLED: 2) Training option to reshuffle_each_iteration has been enabled; this has increased regularization and results in smoother convergence.
#
# NOW DISABLED: 3) Augmentation now only uses random rotations in 90 degree increments and horizonal/vertical flips.
#
#==================================================================

#Random rotation transform using a discrete set of angles; enusres RandomCrop captures data (and doesn't add zeros) in original input FOV
#Reference: https://stackoverflow.com/questions/66368576/is-there-a-way-to-build-a-keras-preprocessing-layer-that-randomly-rotates-at-spe
class RandomDiscreteRotate(PreprocessingLayer):
    def __init__(self): 
        super().__init__()
    def __call__(self, inputs): 
        rots = tf.random.stateless_uniform((1,1), (manualSeedValue, manualSeedValue), 0, 4, dtype=tf.int32)[0,0]
        return tf.image.rot90(inputs, k=rots)

#Perform identical data augmentation steps on an a set of inputs with num channels and an output with one channel
class DataAugmentation(Layer):
    def __init__(self, numChannels=None):
        super().__init__()
        self.numChannels = numChannels
        #self.augmentLayer = tf.keras.Sequential([
        #    RandomFlip('horizontal_and_vertical'),
        #    RandomDiscreteRotate()
        #])
        self.augmentLayer = tf.keras.Sequential([
            RandomFlip('horizontal_and_vertical'),
            RandomRotation(factor = (-0.125, 0.125), fill_mode='constant', interpolation='nearest', fill_value=0.0),
            RandomTranslation(height_factor=(-0.25, 0.25), width_factor=(-0.25, 0.25), fill_mode = 'constant', interpolation='nearest', fill_value=0.0)
        ])
        
    #Convert training/validation sample(s) in ragged tensors to regular tensors and perform augmentation; MUST set training=True for functionality
    def __call__(self, inputs, outputs): return tf.split(self.augmentLayer(tf.concat([inputs.to_tensor(), outputs.to_tensor()], -1), training=True), [self.numChannels,1], axis=-1)

#Convert training/validation sample(s) in ragged tensors to regular tensors
class RaggedPassthrough(Layer):
    def __init__(self): super().__init__()
    def call(self, inputs, outputs): return inputs.to_tensor(), outputs.to_tensor()

#Compute RDPPs
def computeRDPPs(labels, recons):
    return abs(labels-recons)

#Prepare data for DLADS model input
def prepareInput(reconImages, squareMask, squareOpticalImage=None):
    if reconImages.ndim==2: reconImages = np.expand_dims(reconImages, 0)
    inputStack = []
    if 'measureData' in inputChannels: inputStack.append(reconImages*squareMask)
    if 'reconData' in inputChannels: inputStack.append(reconImages*(1-squareMask))
    if 'combinedData' in inputChannels: inputStack.append(reconImages)
    if 'opticalData' in inputChannels: inputStack.append(np.repeat(np.expand_dims(squareOpticalImage, 0), len(reconImages), axis=0))
    if 'mask' in inputChannels: inputStack.append(np.repeat(np.expand_dims(squareMask, 0), len(reconImages), axis=0))
    if len(inputStack) != len(inputChannels): sys.exit('\nError - The number of intended input channels did not match with the number added to the input stack. Please verify that the specified inputChannels are valid.')
    return np.stack(inputStack, axis=-1)
    
#Rescale spatial dimensions of tensor x to match to those of tensor y
def customResize(x, y):
    x = image_ops.resize_images_v2(x, array_ops.shape(y)[1:3], method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
    nshape = tuple(y.shape.as_list())
    x.set_shape((None, nshape[1], nshape[2], None))
    return x
    
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
    outputs = Conv2D(1, 1, activation='relu', padding='same')(conv8)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
    
#Define DLADS model using TensorFlow
class DLADS_TF:
    def __init__(self, trainFlag, local_gpus, modelDirectory=None, modelName=None):
        
        #Configure CPU/GPU computation environment; allocate model location and batch sizes accordingly
        if len(local_gpus) > 0: self.device = tf.device('/device:GPU:'+str(local_gpus[-1]))
        else: self.device = tf.device('/device:CPU:0')
        self.batchsize_TRN, self.batchsize_VAL = 1, 1
        
        #If not training, load model for inferencing, running it once to initialize model on computational device, otherwise the first call would affect inferencing performance
        #If training, use a dummy optimizer run a silenced backpropogation call to prevent superfluous text output from future calls, then load the actual model, running it once to initialize
        #For TensorFlow, must manually enforce device assignment throughout; otherwise defaults to GPU:0 if such is available and then crashes due to unavailable resources!
        with self.device: 
            data = tf.ones((1,512,512,len(inputChannels)))
            if not trainFlag: 
                self.model = tf.function(tf.keras.models.load_model(modelDirectory + modelName, compile=False), experimental_relax_shapes=True)
                _ = self.model(data, training=False)
            else: 
                model = tf.keras.Sequential([Conv1D(1, 1)])
                with tf.GradientTape() as tape: loss = tf.math.reduce_mean(tf.math.abs(model(tf.zeros((1,1,1)), training=True)-tf.zeros((1,1,1))))
                grad = tape.gradient(loss, model.trainable_variables)
                with suppressSTD() if not debugMode else nullcontext(): _ = SGD().apply_gradients(zip(grad, model.trainable_variables))
                self.model = unet(numStartFilters, len(inputChannels))
                _ = self.model(data, training=False)
        
    def loadData(self, trainingDatabase, validationDatabase):
        
        #Force device assignment
        with self.device: 
        
            #Create training and validation datasets compatible with tensorflow models
            inputs_TRN, labels_TRN = [], []
            for sample in tqdm(trainingDatabase, desc = 'Training Data Setup', leave=True, ascii=asciiFlag):
                if sample.squareRD.sum() > 0:
                    inputStack = tf.convert_to_tensor(prepareInput(sample.squareChanReconImages, sample.squareMask, trainingSampleData[sample.sampleDataIndex].squareOpticalImage).astype(np.float32))
                    for chanNum in range(0, len(sample.squareRDs)):
                        input = inputStack[chanNum]
                        label = tf.convert_to_tensor(np.expand_dims(sample.squareRDs[chanNum], -1).astype(np.float32))
                        inputs_TRN.append(input)
                        labels_TRN.append(label)
            trainCount = len(inputs_TRN)
            self.numTRN = trainCount//self.batchsize_TRN
            self.trainData = tf.data.Dataset.from_tensor_slices((tf.ragged.stack(inputs_TRN), tf.ragged.stack(labels_TRN)))
            
            #If there is not a validation set then indicate such, otherwise create required lists
            if len(validationDatabase)<=0: 
                self.valFlag = False
                vizSamples = None
            else:
                self.valFlag = True
                
                #Find the index of the first sample with maximum percent measured
                lastPercMeasured, lastSampleVizIndex  = 0, 0
                for i, sample in enumerate(validationDatabase):
                    if (sample.percMeasured < lastPercMeasured) or (i == len(validationDatabase)-1): 
                        lastSampleVizIndex = i-1
                        break
                    else: 
                        lastPercMeasured = sample.percMeasured
                
                #Move back through the validation sample stack until finding a sample with non-zero RD
                while (True):
                    if (validationDatabase[lastSampleVizIndex].squareRD.sum() > 0) or (lastSampleVizIndex == 0): break
                    else: lastSampleVizIndex -= 1
                
                #Store indicies of validation sample data to visualize during training
                vizSampleIndices = [0]
                if lastSampleVizIndex != 0: vizSampleIndices.append(lastSampleVizIndex)
                
                self.numViz = len(vizSampleIndices)
                inputs_VAL, labels_VAL = [], []
                self.inputs_Viz, self.labels_Viz = [], []
                for i, sample in enumerate(tqdm(validationDatabase, desc = 'Validation Data Setup', leave=True, ascii=asciiFlag)):
                    if sample.squareRD.sum() > 0:
                        inputStack = tf.convert_to_tensor(prepareInput(sample.squareChanReconImages, sample.squareMask, validationSampleData[sample.sampleDataIndex].squareOpticalImage).astype(np.float32))
                        if i in vizSampleIndices:
                            self.inputs_Viz.append(tf.convert_to_tensor(inputStack))
                            self.labels_Viz.append(sample.squareRD)
                        for chanNum in range(0, len(sample.squareRDs)):
                            input = inputStack[chanNum]
                            label = tf.convert_to_tensor(np.expand_dims(sample.squareRDs[chanNum], -1).astype(np.float32))
                            inputs_VAL.append(input)
                            labels_VAL.append(label)
                valCount = len(inputs_VAL)
                self.numVAL = valCount//self.batchsize_VAL
                self.valData = tf.data.Dataset.from_tensor_slices((tf.ragged.stack(inputs_VAL), tf.ragged.stack(labels_VAL)))
            
            #Set dynamic resource tuning option
            AUTOTUNE = tf.data.AUTOTUNE
            
            #Setup training dataset for model
            self.trainData = self.trainData.repeat()
            self.trainData = self.trainData.shuffle(trainCount, seed=0, reshuffle_each_iteration=False)
            self.trainData = self.trainData.batch(self.batchsize_TRN)
            if augTrainData: self.trainData = self.trainData.map(DataAugmentation(len(inputChannels)), num_parallel_calls=AUTOTUNE, deterministic=True)
            else: self.trainData = self.trainData.map(RaggedPassthrough(), num_parallel_calls=AUTOTUNE, deterministic=True)
            self.trainData = self.trainData.prefetch(AUTOTUNE)
            
            #Setup validation dataset for model if applicable
            #Note that shuffle and repeat operations were removed here to synchronize inter-model scoring
            if self.valFlag: 
                self.valData = self.valData.batch(self.batchsize_VAL)
                self.valData = self.valData.map(RaggedPassthrough(), num_parallel_calls=AUTOTUNE, deterministic=True)
                self.valData = self.valData.prefetch(AUTOTUNE)
    
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
                vizSample = self.inputs_Viz[vizSampleNum]
                squareERD = np.mean(self.model(vizSample, training=False)[:,:,:,0].numpy(), axis=0)
                squareRD = self.labels_Viz[vizSampleNum]
                ERD_PSNR, ERD_SSIM, ERD_NRMSE = compareImages(squareRD, squareERD, np.min(squareRD), np.max(squareRD))
                
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
    
    @tf.function
    def computeLoss(self, data, label, trainFlag=False):
        if trainFlag:
            with tf.GradientTape() as tape: loss = tf.math.reduce_mean(tf.math.abs(self.model(data, training=True)-label))
            grad = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        else: 
            loss = tf.math.reduce_mean(tf.math.abs(self.model(data, training=False)-label))
        return loss
    
    def setOptimizer(self):
            if optimizer == 'NAdam': self.opt = Nadam(learning_rate=self.learningRate)
            elif optimizer == 'Adam': self.opt = Adam(learning_rate=self.learningRate)
            elif optimizer == 'RMSprop': self.opt = RMSprop(learning_rate=self.learningRate)
            elif optimizer == 'SGD': self.opt = SGD(learning_rate=self.learningRate)
            else: sys.exit('\nError - Unknown optimizer was specified.')

    def train(self):
    
        #Force device assignment
        with self.device: 
        
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
            
            #Create iterator for the training dataset; replicates behavior of prior implementation, where the data can bleed between epochs
            iterator_TRN = iter(self.trainData)
            
            #Perform model training
            t0 = time.perf_counter()
            for epoch in trainingBar:
                
                #Compute losses over the training dataset
                #self.loss_TRN.append(np.mean([self.computeLoss(data, label, True) for data, label in tqdm(self.trainData, total=self.numTRN, desc='TRN', leave=False, ascii=asciiFlag)]))
                
                #Enabling repeat() before shuffle() and using an iterator and the get_next() loop below can be used to replicate the inter-epoch data bleeding behavior from prior versions
                losses = []
                for _ in tqdm(range(self.numTRN), total=self.numTRN, desc='TRN', leave=False, ascii=asciiFlag):
                    data, label = iterator_TRN.get_next()
                    losses.append(self.computeLoss(data, label, True))
                self.loss_TRN.append(np.mean(losses))
                
                #Compute losses over the validation dataset and update the current loss used for training status evaluation
                if self.valFlag: 
                    self.loss_VAL.append(np.mean([self.computeLoss(data, label, False) for data, label in tqdm(self.valData, total=self.numVAL, desc='VAL', leave=False, ascii=asciiFlag)]))
                    loss_CUR = self.loss_VAL[-1]
                else: loss_CUR = self.loss_TRN[-1]
                
                #If this model is the best performing, update the best model parameters and reset applicable variables; if not an improvement and not using mean early stopping, increase the patience counter
                if (loss_CUR < self.loss_BST): self.model_BST, self.loss_BST, self.epoch_BST, self.patienceCounter, self.stagnationCounter = copy.deepcopy(self.model.get_weights()), loss_CUR, epoch, 0, 0
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
            
            #Save the final model and weights; do not include optimizer to save space
            self.model.stop_training = True
            self.model.set_weights(self.model_BST)
            self.model.save(dir_TrainingResults + modelName, include_optimizer=False)

            #Save training history
            history = np.vstack([np.array(range(0, epoch+1)), self.loss_TRN])
            if self.valFlag: history = np.vstack([history, self.loss_VAL])
            pd.DataFrame(history.T, columns=['Epoch','Loss_TRN', 'Loss_VAL']).to_csv(dir_TrainingResults+'trainingHistory.csv', index=False)
    
    def predict(self, input):
        with self.device: 
            if len(input.shape) == 3: input = np.expand_dims(input, 0)
            return self.model(input, training=False)[:,:,:,0].numpy()
