#==================================================================
#TEMPORARY REFERENCE
#v0.9.5 TensorFlow DLADS implementation - DO NOT USE!
#Set manualSeedValue to -1, as this implementation is nondeterministic
#
#During change to the new training loop showed an issue with inter-epoch data bleeding not previously noticed.
#This leads to much more jagged convergence of the training losses and increased variance in resulting model performance.
#The unfortunate behavior was maintained here, but removed in MODEL_DLADS_TF_Deprecated.py, where the data handling procedures 
#were altered to match those used in MODEL_DLADS_TF.py and MODEL_DLADS_PY.py. As this code may be useful in understanding 
#the transition between models and machine learning packages, a copy shall be included with at least one published version of the program. 
#
#=================================================================

#Additional import(s)
from tensorflow.keras.callbacks import Callback
from tqdm import keras as kerasTQDM

#Perform identical data augmentation steps on an a set of inputs with num channels and an output with one channel
class DataAugmentation(Layer):
    def __init__(self, numChannels=None):
        super().__init__()
        self.numChannels = numChannels
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
    if len(inputStack) != len(inputChannels): sys.exit('Error - The number of intended input channels did not match with the number added to the input stack. Please verify that the specified inputChannels are valid.')
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

#Tensorflow callback object to check early stopping criteria and visualize the network's current training progression/status
class EpochEnd(Callback):
    def __init__(self, valFlag, inputs_Viz, labels_Viz, numViz, dir_TrainingModelResults):
        self.valFlag = valFlag
        self.inputs_Viz = inputs_Viz
        self.labels_Viz = labels_Viz
        self.numViz = numViz
        self.dir_TrainingModelResults = dir_TrainingModelResults
        self.loss_TRN = []
        if self.valFlag: self.loss_VAL = []
        self.patience = 0
        self.bestWeights = None
        self.loss_BST = np.inf
        self.epoch_BST = -1
        self.epoch_STP =-1
        self.nanValue = False
        
    def on_epoch_end(self, epoch, logs=None):
        
        if np.isnan(logs.get('loss')): 
            self.model.stop_training = True
            self.nanValue = True
        
        #Store model convergence progress
        self.loss_TRN.append(logs.get('loss'))
        if self.valFlag: 
            currentLoss = logs.get('val_loss')
            self.loss_VAL.append(logs.get('val_loss'))
        else: 
            currentLoss = logs.get('loss')
        
        #Update early stopping criteria
        if (currentLoss < self.loss_BST):
            self.patience = 0
            self.loss_BST = currentLoss
            self.epoch_BST = epoch
            self.bestWeights = copy.deepcopy(self.model.get_weights())
        else:
            self.patience += 1
            if self.patience >= maxPatience: self.epoch_STP = epoch
        
        #Perform visualization as needed/specified
        if trainingProgressionVisuals and ((epoch == 0) or (epoch % trainingVizSteps == 0) or (self.epoch_STP == epoch) or (self.epoch_BST == epoch)):

            if self.valFlag: f = plt.figure(figsize=(15,15))
            else: f = plt.figure(figsize=(15,8))
            f.subplots_adjust(bottom=0.05, left=0.10, right=0.90, top=0.88, wspace=0.20, hspace=0.30)
            plt.rcParams['font.size'] = 10
            
            if self.valFlag: ax = plt.subplot2grid((3,1), (0,0))
            else: ax = plt.subplot2grid((1,1), (0,0))
            ax.plot(self.loss_TRN, label='TRN', alpha=0.7, color='red')
            if self.valFlag: ax.plot(self.loss_VAL, label='VAL', alpha=0.7, color='blue')
            ax.set_yscale('log')
            ax.legend(loc='upper right', fontsize=10)
            
            if self.valFlag: 
                for vizSampleNum in range(0, self.numViz): 
                    vizSample = self.inputs_Viz[vizSampleNum]
                    squareERD = np.mean(self.model(vizSample, training=False)[:,:,:,0].numpy(), axis=0)
                    squareRD = self.labels_Viz[vizSampleNum]
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
            
            plotTitle = 'EPOCH: ' + ('{:0'+str(maxEpochsPrecision)+'}').format(epoch) + ', LR: ' + '{:.1e}'.format(learningRate)
            plotTitle += '\nPAT: ' + ('{:0'+str(maxPatiencePrecision)+'}').format(self.patience) + '/' + str(maxPatience)
            plotTitle += '\nLOSS - TRN: ' + '{:.6f}'.format(round(self.loss_TRN[-1], 6)) + ', '
            if self.valFlag: plotTitle += 'VAL: ' + '{:.6f}'.format(round(self.loss_VAL[-1], 6)) + ', '
            else: plotTitle += 'VAL: N/A,   '
            plotTitle += 'BST: ' + '{:.6f}'.format(round(self.loss_BST, 6)) +' at Epoch: ' + ('{:0'+str(maxEpochsPrecision)+'}').format(self.epoch_BST)
            plotTitle += '\nMean Loss - Historical: N/A,   Current: N/A    '
            plt.suptitle(plotTitle, fontsize=20, fontweight='bold')
            
            #Save resulting plot
            f.savefig(dir_TrainingModelResults + 'epoch_' +str(epoch) + '.tiff')
            plt.close(f)
        
        #Apply early stopping criteria (after visualization)
        if self.epoch_STP == epoch: 
            self.model.stop_training = True
            self.model.set_weights(self.bestWeights)
    
#Define DLADS model using TensorFlow
class DLADS_TF:
    def __init__(self, trainFlag, local_gpus, modelDirectory=None, modelName=None):
    
        #Configure CPU/GPU computation environment; allocate model location and batch sizes accordingly
        if len(local_gpus) >=0: self.device = tf.device('/device:GPU:'+str(local_gpus[-1]))
        else: self.device = nullcontext()
        self.batchsize_TRN, self.batchsize_VAL = 1, 1
        
        #If not training, load model for inferencing, running it once to initialize model on computational device, otherwise the first call would affect inferencing performance
        #If training, use a dummy optimizer run a silenced backpropogation call to prevent superfluous text output from future calls, then load the actual model
        self.blankData = tf.ones((1,512,512,len(inputChannels)))
        if not trainFlag: 
            with self.device: self.model = tf.function(tf.keras.models.load_model(modelDirectory + modelName, compile=False), experimental_relax_shapes=True)
            _ = self.model(self.blankData, training=False)
        else: 
            with self.device: model = tf.keras.Sequential([Conv1D(1, 1)])
            with tf.GradientTape() as tape: loss = tf.math.reduce_mean(tf.math.abs(model(tf.zeros((1,1,1)), training=True)-tf.zeros((1,1,1))))
            grad = tape.gradient(loss, model.trainable_variables)
            with suppressSTD() if not debugMode else nullcontext(): _ = SGD().apply_gradients(zip(grad, model.trainable_variables))
            with self.device: self.model = unet(numStartFilters, len(inputChannels))
            
    def loadData(self, trainingDatabase, validationDatabase):
    
        #Create training and validation datasets compatible with tensorflow models
        inputs_TRN, labels_TRN = [], []
        for sample in tqdm(trainingDatabase, desc = 'Training Data Setup', leave=True, ascii=asciiFlag):
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
            vizSampleIndices = [0, len(np.arange(initialPercToScanTrain, stopPercTrain))]
            self.numViz = len(vizSampleIndices)
            inputs_VAL, labels_VAL = [], []
            self.inputs_Viz, self.labels_Viz = [], []
            for i, sample in enumerate(tqdm(validationDatabase, desc = 'Validation Data Setup', leave=True, ascii=asciiFlag)):
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
        
        #Setup training dataset for model; repeat before shuffle allows data to 'bleed' between epochs (not technically desirable, but is part of the deprecated behavior)
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
            
    def setOptimizer(self):
        if optimizer == 'AdamW': self.opt = AdamW(learning_rate=self.learningRate)
        elif optimizer == 'NAdam': self.opt = Nadam(learning_rate=self.learningRate)
        elif optimizer == 'Adam': self.opt = Adam(learning_rate=self.learningRate)
        elif optimizer == 'RMSprop': self.opt = RMSprop(learning_rate=self.learningRate)
        elif optimizer == 'SGD': self.opt = SGD(learning_rate=self.learningRate)
        else: sys.exit('Error - Unknown optimizer was specified')

    def train(self):
    
        #Setup learning rate and optimizer
        self.learningRate = copy.deepcopy(learningRate)
        self.setOptimizer()
        
        #Compile the optimizer/model
        with self.device: self.model.compile(optimizer=self.opt, loss='mean_absolute_error')
        
        #Setup callback object for visualizing training convergence
        epochEndCallback = EpochEnd(self.valFlag, self.inputs_Viz, self.labels_Viz, self.numViz, dir_TrainingModelResults)
        tqdmCallback = kerasTQDM.TqdmCallback()
        tqdmCallback.epoch_bar.desc='Epoch'
        tqdmCallback.batch_bar.desc='Batch'
        
        #Perform model training
        t0 = time.perf_counter()
        if self.valFlag: history = self.model.fit(self.trainData, epochs=maxEpochs, callbacks=[epochEndCallback, tqdmCallback], validation_data=self.valData, steps_per_epoch=self.numTRN, validation_steps=self.numVAL, verbose=0)
        else: history = self.model.fit(self.trainData, epochs=maxEpochs, callbacks=[epochEndCallback, tqdmCallback], steps_per_epoch=self.numTRN, verbose=0)
        t1 = time.perf_counter()
        trainingTime = datetime.timedelta(seconds=(t1-t0))
        lines = ['Model Training Time: ' + str(trainingTime)]
        with open(dir_TrainingResults + 'trainingTime.txt', 'w') as f:
            for line in lines: _ = f.write(line+'\n')
        print(lines[0])
        
        #Save the final model and weights; do not include optimizer to save space
        self.model.save(dir_TrainingResults + modelName, include_optimizer=False)
        
        #Save training history
        pd.DataFrame(history.history).to_csv(dir_TrainingResults+'trainingHistory.csv')
    
    def predict(self, input):
        return self.model(input, training=False)[:,:,:,0].numpy()
