#==================================================================
#DEPRECATED - CODE TO BE ARCHIVED AT LEAST ONCE BEFORE REMOVAL
#==================================================================

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

#Convert image into TF model compatible shapes/tensors
def makeCompatible(image):
    
    #Turn into an array before processings; will produce an error in the event of dimensional incompatability
    image = np.asarray(image)

    #Reshape for tensor transition, as needed by number of channels
    if len(image.shape) > 3: return image
    elif len(image.shape) > 2: return image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    else: return image.reshape((1,image.shape[0],image.shape[1],1))


def convLayer(numFilters, kernSize, inputs, activation):
    if activation == 'swish': return swish(Conv2D(numFilters, kernSize, kernel_initializer='he_normal', padding='same')(inputs))
    elif activation == 'relu': return Conv2D(numFilters, kernSize, kernel_initializer='he_normal', activation='relu', padding='same')(inputs)
    else: return Conv2D(numFilters, kernSize, kernel_initializer='he_normal', padding='same')(inputs)
    
def modelUnified(numFilters, numChannelsIn):
    
    inputs = Input(shape=(None,None,numChannelsIn), batch_size=None)
    
    conv0 = convLayer(numFilters, 1, inputs, 'swish')
    conv0 = convLayer(numFilters, 3, conv0, 'swish')
    
    conv1 = MaxPool2D(pool_size=(2,2))(conv0)
    conv1 = convLayer(numFilters*2, 1, conv1, 'swish')
    conv1 = convLayer(numFilters*2, 3, conv1, 'swish')
    
    conv2 = MaxPool2D(pool_size=(2,2))(conv1)
    conv2 = convLayer(numFilters*4, 1, conv2, 'swish')
    conv2 = convLayer(numFilters*4, 3, conv2, 'swish')
    
    conv3 = MaxPool2D(pool_size=(2,2))(conv2)
    conv3 = convLayer(numFilters*8, 1, conv3, 'swish')
    conv3 = convLayer(numFilters*8, 3, conv3, 'swish')
    
    conv4 = MaxPool2D(pool_size=(2,2))(conv3)
    conv4 = convLayer(numFilters*8, 1, conv4, 'swish')
    conv4 = convLayer(numFilters*8, 3, conv4, 'swish')
    
    conv5 = convLayer(numFilters*8, 2, customResize(conv4, conv3), 'swish')
    conv5 = concatenate([conv3, conv5])
    conv5 = convLayer(numFilters*8, 1, conv5, 'swish')
    conv5 = convLayer(numFilters*8, 3, conv5, 'swish')
    
    conv6 = convLayer(numFilters*8, 2, customResize(conv5, conv2), 'swish')
    conv6 = concatenate([conv2, conv6])
    conv6 = convLayer(numFilters*4, 1, conv6, 'swish')
    conv6 = convLayer(numFilters*4, 3, conv6, 'swish')
    
    conv7 = convLayer(numFilters*4, 2, customResize(conv6, conv1), 'swish')
    conv7 = concatenate([conv1, conv7])
    conv7 = convLayer(numFilters*2, 1, conv7, 'swish')
    conv7 = convLayer(numFilters*2, 3, conv7, 'swish')
    
    conv8 = convLayer(numFilters*2, 2, customResize(conv7, conv0), 'swish')
    conv8 = concatenate([conv0, conv8])
    conv8 = convLayer(numFilters, 1, conv8, 'swish')
    conv8 = convLayer(numFilters, 3, conv8, 'swish')
    
    outputs = convLayer(1, 1, conv8, 'relu')
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

class GLANDS(keras.Model):
    
    def __init__(self, modelRecon, modelCritic, modelERRE):
        super(GLANDS, self).__init__()
        self.modelRecon = modelRecon
        self.modelCritic = modelCritic
        self.modelERRE = modelERRE
        
    def compile(self, opt_Recon, opt_Critic, opt_ERRE):
        super(GLANDS, self).compile()
        self.opt_Recon = opt_Recon
        self.opt_Critic = opt_Critic
        self.opt_ERRE = opt_ERRE
    
    #Compute losses for input data for either training or inference
    def calcLosses(self, data_GT, trainFlag):
        
        #Ensure the data is a regular and not ragged tensor before processing
        #data_GT = data_GT.to_tensor()
        
        #Get shape of data_GT and create corresponding zero, one, and noise arrays
        arr_Shape = tf.shape(data_GT)
        numPositions = tf.cast(tf.math.reduce_prod(arr_Shape), tf.float32)
        arr_Ones = tf.ones(arr_Shape)
        arr_Zeros = tf.zeros(arr_Shape)
        arr_Random = tf.reshape(tf.random.shuffle(tf.range(numPositions)), arr_Shape)
        arr_Noise = tf.random.uniform(arr_Shape)
        
        #Generate a random 1-25% measurement mask and apply to the GT data
        num_Initial = tf.floor(numPositions*tf.random.uniform([], minval=0.75, maxval=0.99))
        mask_Initial_Bool = tf.greater_equal(arr_Random, num_Initial)
        mask_Initial = tf.cast(mask_Initial_Bool, dtype=tf.float32)
        data_Initial = data_GT*mask_Initial
        
        #Update the initial mask with up to 25% more random measurements
        num_New = tf.floor(numPositions*tf.random.uniform([], minval=0.50, maxval=0.75))
        mask_New = tf.cast(tf.greater_equal(arr_Random, num_New), dtype=tf.float32)
        data_New = data_GT*mask_New
        
        #Inference reconstruction model
        recon_Initial = self.modelRecon(tf.concat([data_Initial, mask_Initial], -1), training=True)
        recon_New = self.modelRecon(tf.concat([data_New, mask_New], -1), training=True)
        recon_GT = self.modelRecon(tf.concat([data_GT, arr_Ones], -1), training=False)
        
        #Inference critic model
        critic_Initial = self.modelCritic(tf.concat([tf.stop_gradient(recon_Initial), data_Initial, mask_Initial], -1), training=True)
        critic_New = self.modelCritic(tf.concat([tf.stop_gradient(recon_New), data_New, mask_New], -1), training=True)
        #critic_GT = self.modelCritic(tf.concat([tf.stop_gradient(recon_GT), data_GT, arr_Ones], -1), training=True)
        
        #Inference ERRE model
        ERRE_Initial = self.modelERRE(tf.concat([tf.stop_gradient(recon_Initial), data_Initial, mask_Initial], -1), training=True)
        ERRE_New = self.modelERRE(tf.concat([tf.stop_gradient(recon_New), data_New, mask_New], -1), training=True)
        #ERRE_GT = self.modelERRE(tf.concat([tf.stop_gradient(recon_GT), data_GT, arr_Ones], -1), training=True)
        
        actual_Recon_Error_Initial = tf.abs(data_GT-recon_Initial)
        actual_Recon_Error_New = tf.abs(data_GT-recon_New)
        actual_Recon_Error_GT = tf.abs(data_GT-recon_GT)
        
        RRE_Initial = tf.reduce_mean(tf.abs(actual_Recon_Error_Initial-actual_Recon_Error_New))
        RRE_New = tf.reduce_mean(tf.abs(actual_Recon_Error_New-recon_GT))
        #RRE_GT = arr_Zeros
        
        #Compute and combine losses for reconstruction model        
        loss_Recon_Initial = tf.reduce_mean(actual_Recon_Error_Initial)+tf.reduce_mean(critic_Initial)
        loss_Recon_New = tf.reduce_mean(actual_Recon_Error_New)+tf.reduce_mean(critic_New)
        #loss_Recon_GT = tf.reduce_mean(actual_Recon_Error_GT)+tf.reduce_mean(critic_GT)
        loss_Recon_Total = tf.reduce_mean([loss_Recon_Initial, loss_Recon_New])
        
        #Compute and combine losses for the critic model
        loss_Critic_Initial = tf.reduce_mean(tf.abs(actual_Recon_Error_Initial-critic_Initial))
        loss_Critic_New = tf.reduce_mean(tf.abs(actual_Recon_Error_New-critic_New))
        #loss_Critic_GT = tf.reduce_mean(tf.abs(actual_Recon_Error_GT-critic_GT))
        loss_Critic_Total = tf.reduce_mean([loss_Critic_Initial, loss_Critic_New])
        
        #Compute and combine losses for ERRE model
        loss_ERRE_Initial = tf.abs(RRE_Initial-tf.reduce_mean(ERRE_Initial*(mask_New-mask_Initial)))+tf.reduce_mean(ERRE_Initial*mask_Initial)
        loss_ERRE_New = tf.abs(RRE_New-tf.reduce_mean(ERRE_New*(arr_Ones-mask_New)))+tf.reduce_mean(ERRE_New*mask_New)
        #loss_ERRE_GT = tf.reduce_mean(ERRE_GT)
        loss_ERRE_Total = tf.reduce_mean([loss_ERRE_Initial, loss_ERRE_New])
        
        return loss_Recon_Total, loss_Critic_Total, loss_ERRE_Total
    
    #Compute and return inference losses
    def test_step(self, data_GT):
        loss_Recon_Total, loss_Critic_Total, loss_ERRE_Total = self.calcLosses(data_GT, False)
        return {"Recon": loss_Recon_Total, "Critic": loss_Critic_Total, "ERRE": loss_ERRE_Total}
    
    #Compute losses, resulting gradients, and apply to model weights
    def train_step(self, data_GT):
        
        #Create temporary gradient tapes
        with tf.GradientTape() as tape_Recon, tf.GradientTape() as tape_Critic, tf.GradientTape() as tape_ERRE:
            
            #Compute training losses
            loss_Recon_Total, loss_Critic_Total, loss_ERRE_Total = self.calcLosses(data_GT, True)
            
            #Compute losses and resultant gradients
            grad_Recon = tape_Recon.gradient(loss_Recon_Total, self.modelRecon.trainable_variables)
            grad_Critic = tape_Critic.gradient(loss_Critic_Total, self.modelCritic.trainable_variables)
            grad_ERRE = tape_ERRE.gradient(loss_ERRE_Total, self.modelERRE.trainable_variables)
            
        #Apply gradients
        self.opt_Recon.apply_gradients(zip(grad_Recon, self.modelRecon.trainable_variables))
        self.opt_Critic.apply_gradients(zip(grad_Critic, self.modelCritic.trainable_variables))
        self.opt_ERRE.apply_gradients(zip(grad_ERRE, self.modelERRE.trainable_variables))
        
        #Return loss dictionary
        return {"Recon": loss_Recon_Total, "Critic": loss_Critic_Total, "ERRE": loss_ERRE_Total}

#Convert training/validation sample(s) in ragged tensors to regular tensors
class RaggedPassthrough(Layer):
    def __init__(self): super().__init__()
    def call(self, inputs, outputs=[]): 
        if len(outputs)==0: return inputs.to_tensor()
        else: return inputs.to_tensor(), outputs.to_tensor()

#DLADS; Tensorflow callback object to check early stopping criteria and visualize the network's current training progression/status
class EpochEnd_DLADS(Callback):
    def __init__(self, maxPatience, minimumEpochs, trainingProgressionVisuals, trainingVizSteps, valFlag, vizSamples, vizSampleData, dir_TrainingModelResults):
        self.maxPatience = maxPatience
        self.patience = 0
        self.bestWeights = None
        self.bestEpoch = 0
        self.bestLoss = np.inf
        self.stopped_epoch = 0
        self.minimumEpochs = minimumEpochs
        self.trainingProgressionVisuals = trainingProgressionVisuals
        self.trainingVizSteps = trainingVizSteps
        self.valFlag = valFlag
        self.train_lossList = []
        self.vizSamples = vizSamples
        self.vizSampleData = vizSampleData
        self.dir_TrainingModelResults = dir_TrainingModelResults
        if self.valFlag: self.val_lossList = []
        self.nanValue = False
        self.valLosses = []
        
        self.vizSampleBatches = [makeCompatible(prepareInput(self.vizSamples[vizSampleNum].squareChanReconImages, self.vizSamples[vizSampleNum].squareMask, self.vizSampleData.squareOpticalImage)) for vizSampleNum in range(0, len(self.vizSamples))]

    def on_epoch_end(self, epoch, logs=None):
        
        if np.isnan(logs.get('loss')): 
            self.model.stop_training = True
            self.nanValue = True
        
        #Store model convergence progress
        self.train_lossList.append(logs.get('loss'))
        if valFlag: 
            currentLoss = logs.get('val_loss')
            self.val_lossList.append(logs.get('val_loss'))
        else: 
            currentLoss = logs.get('loss')
        
        #Early stopping criteria
        if (currentLoss < self.bestLoss) and (epoch >= self.minimumEpochs):
            self.patience = 0
            self.bestLoss = currentLoss
            self.bestEpoch = epoch
            self.bestWeights = copy.deepcopy(self.model.get_weights())
        elif (epoch >= self.minimumEpochs):
            self.patience += 1
            if self.patience >= self.maxPatience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.bestWeights)
                
        #Perform visualization as needed/specified
        if trainingProgressionVisuals and ((epoch == 0) or (epoch % trainingVizSteps == 0) or (self.stopped_epoch == epoch) or (self.bestEpoch == epoch)):

            #If there are no validation tensors, then just save a plot of the training losses
            if not self.valFlag:
                f = plt.figure(figsize=(25,5))
                f.subplots_adjust(top = 0.80)
                f.subplots_adjust(wspace=0.2, hspace=0.2)
                
                #Plot losses
                ax = plt.subplot2grid((1,1), (0,0))
                ax.plot(self.train_lossList, label='Training')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                ax.set_title('Training Loss: ' + str(round(self.train_lossList[-1],8)), fontsize=15, fontweight='bold')
                
            else:
                f = plt.figure(figsize=(25,15))
                f.subplots_adjust(top = 0.88)
                f.subplots_adjust(wspace=0.2, hspace=0.2)
                
                #Plot losses
                ax = plt.subplot2grid((3,1), (0,0))
                ax.plot(self.train_lossList, label='Training')
                ax.plot(self.val_lossList, label='Validation')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                ax.set_title('Training Loss: ' + str(round(self.train_lossList[-1],8)) + '     Validation Loss: ' + str(round(self.val_lossList[-1],8)), fontsize=15, fontweight='bold')
                
                #Show a validation sample result at min and max sampling percentages (variables provided through callback initialization)
                for vizSampleNum in range(0, len(self.vizSamples)):
                    
                    squareERD = np.mean(self.model(self.vizSampleBatches[vizSampleNum], training=False)[:,:,:,0].numpy(), axis=0)
                    squareRD = self.vizSamples[vizSampleNum].squareRD
                    
                    maxRangeValue = np.max([squareRD, squareERD])
                    ERD_PSNR = compare_psnr(squareRD, squareERD, data_range=maxRangeValue)
                    ERD_SSIM = compare_ssim(squareRD, squareERD, data_range=maxRangeValue)
                    
                    if np.isnan(ERD_PSNR) or np.isnan(ERD_SSIM): 
                        self.model.stop_training = True
                        self.nanValue = True
                    
                    ax = plt.subplot2grid((3,2), (vizSampleNum+1,0))
                    im = ax.imshow(squareRD, aspect='auto', vmin=0, interpolation='none')
                    ax.set_title('RD', fontsize=15, fontweight='bold')
                    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                    
                    ax = plt.subplot2grid((3,2), (vizSampleNum+1,1))
                    im = ax.imshow(squareERD, aspect='auto', vmin=0, interpolation='none')
                    ax.set_title('ERD - PSNR: ' + str(round(ERD_PSNR,4)) + ' SSIM: ' + str(round(ERD_SSIM,4)), fontsize=15, fontweight='bold')
                    cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                    
            plt.suptitle('Epoch: '+str(epoch)+'     Patience: '+str(self.patience)+'/'+str(self.maxPatience)+'\nBest Loss: '+str(round(self.bestLoss, 5))+' at Epoch: '+str(self.bestEpoch), fontsize=20, fontweight='bold', y = 0.92)
            
            #Save resulting plot
            f.savefig(self.dir_TrainingModelResults + 'epoch_' +str(epoch) + '.tiff', bbox_inches='tight')
            plt.close()

#GLANDS; Tensorflow callback object to check early stopping criteria and visualize the network's current training progression/status
class EpochEnd_GLANDS(Callback):
    def __init__(self, maxPatience, minimumEpochs, trainingProgressionVisuals, trainingVizSteps, valFlag, vizSampleData, dir_TrainingModelResults):
        self.maxPatience = maxPatience
        self.patience = 0
        self.bestWeights_Recon = None
        self.bestWeights_Critic = None
        self.bestWeights_ERRE = None
        self.bestEpoch = 0
        self.bestLoss = np.inf
        self.stopped_epoch = 0
        self.minimumEpochs = minimumEpochs
        self.trainingProgressionVisuals = trainingProgressionVisuals
        self.trainingVizSteps = trainingVizSteps
        self.valFlag = valFlag
        self.vizSampleData = vizSampleData
        self.dir_TrainingModelResults = dir_TrainingModelResults
        self.nanValue = False
        self.train_lossReconList, self.train_lossCriticList, self.train_lossERREList = [], [], []
        if self.valFlag: self.val_lossReconList, self.val_lossCriticList, self.val_lossERREList = [], [], []
        
        #Prepare a validation sample for visualization, at complete (100%), low (1%), and high (30%) measured densities
        self.val_GT_Image = self.vizSampleData.chanImages[0]
        self.val_GT = tf.convert_to_tensor(np.expand_dims(self.val_GT_Image, axis=(0, -1)))
        #self.val_Ones = tf.ones_like(self.val_GT)
        #self.val_GT_Input = tf.concat([self.val_GT, self.val_GT, self.val_Ones], -1)
        self.val_numPos = tf.cast(tf.math.reduce_prod(tf.shape(self.val_GT)), tf.float32)
        self.arr_Random = tf.reshape(tf.random.shuffle(tf.range(self.val_numPos)), tf.shape(self.val_GT))
        self.val_Mask_Low = tf.cast(tf.greater_equal(self.arr_Random, self.val_numPos*0.99), tf.float32)
        self.val_Mask_High = tf.cast(tf.greater_equal(self.arr_Random, self.val_numPos*0.70), tf.float32)
        self.val_Mask_Low_Image = self.val_Mask_Low[0,:,:,0].numpy()
        self.val_Mask_High_Image = self.val_Mask_High[0,:,:,0].numpy()
        self.val_Low = self.val_GT*self.val_Mask_Low
        self.val_High = self.val_GT*self.val_Mask_High
    
    def on_epoch_end(self, epoch, logs=None):
        
        #Store current and progressive training losses
        curr_train_lossRecon = logs.get('Recon')
        curr_train_lossCritic = logs.get('Critic')
        curr_train_lossERRE = logs.get('ERRE')
        self.train_lossReconList.append(curr_train_lossRecon)
        self.train_lossCriticList.append(curr_train_lossCritic)
        self.train_lossERREList.append(curr_train_lossERRE)
        
        #Store current and progressive validation losses, 
        if self.valFlag: 
            curr_val_lossRecon = logs.get('val_Recon')
            curr_val_lossCritic = logs.get('val_Critic')
            curr_val_lossERRE = logs.get('val_ERRE')
            self.val_lossReconList.append(curr_val_lossRecon)
            self.val_lossCriticList.append(curr_val_lossCritic)
            self.val_lossERREList.append(curr_val_lossERRE)
            
        #Use (recon+erre)/2 loss for termination criteria, from validation if available
        #The ERRE can only improve so long as the reconstruction does...
        #if self.valFlag: currentLoss = (curr_val_lossRecon+curr_val_lossERRE)/2
        #else: currentLoss = (curr_train_lossRecon+curr_train_lossERRE)/2
        
        #Use reconstruction loss for termination criteria, from validation if available
        if self.valFlag: currentLoss = curr_val_lossRecon
        else: currentLoss = curr_train_lossRecon
        
        #Set to terminate training if there is a nan training loss
        if np.isnan(curr_train_lossRecon) or np.isnan(curr_train_lossCritic) or np.isnan(curr_train_lossERRE): 
            self.model.modelRecon.stop_training = True
            self.model.modelCritic.stop_training = True
            self.model.modelERRE.stop_training = True
            self.nanValue = True
        
        #Early stopping criteria
        if (currentLoss < self.bestLoss) and (epoch >= self.minimumEpochs):
            self.patience = 0
            self.bestLoss = currentLoss
            self.bestEpoch = epoch
            self.bestWeights_Recon = copy.deepcopy(self.model.modelRecon.get_weights())
            self.bestWeights_Critic = copy.deepcopy(self.model.modelCritic.get_weights())
            self.bestWeights_ERRE = copy.deepcopy(self.model.modelERRE.get_weights())
        elif (epoch >= self.minimumEpochs):
            self.patience += 1
            if self.patience >= self.maxPatience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.modelRecon.set_weights(self.bestWeights_Recon)
                self.model.modelCritic.set_weights(self.bestWeights_Critic)
                self.model.modelERRE.set_weights(self.bestWeights_ERRE)
                
        #Perform visualization as needed/specified
        if trainingProgressionVisuals and ((epoch == 0) or (epoch % trainingVizSteps == 0) or (self.stopped_epoch == epoch) or (self.bestEpoch == epoch)):

            #If there is no validation data, then just save training loss plots
            if not self.valFlag:
                f = plt.figure(figsize=(25,5))
                f.subplots_adjust(top = 0.80)
                f.subplots_adjust(wspace=0.2, hspace=0.2)
                
                ax = plt.subplot2grid((1,3), (0,0))
                ax.plot(self.train_lossReconList, label='Training')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                plt.title('Recon: ' + str(round(self.train_lossReconList[-1],8)), fontsize=15, fontweight='bold')

                ax = plt.subplot2grid((1,3), (0,1))
                ax.plot(self.train_lossCriticList, label='Training')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                plt.title('Critic:' + str(round(self.train_lossCriticList[-1],8)), fontsize=15, fontweight='bold')

                ax = plt.subplot2grid((1,3), (0,2))
                ax.plot(self.train_lossERREList, label='Training')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                ax.set_title('ERRE: ' + str(round(self.train_lossERREList[-1],8)), fontsize=15, fontweight='bold')
                
            else:
                
                recon_Low = self.model.modelRecon(tf.concat([self.val_Low, self.val_Mask_Low], -1))
                critic_Low = self.model.modelCritic(tf.concat([recon_Low, self.val_Low, self.val_Mask_Low], -1))
                erre_Low = self.model.modelERRE(tf.concat([recon_Low, self.val_Low, self.val_Mask_Low], -1))
                recon_Low = recon_Low[0,:,:,0].numpy()
                critic_Low = critic_Low[0,:,:,0].numpy()
                erre_Low = erre_Low[0,:,:,0].numpy()
                error_Low = np.abs(self.val_GT_Image-recon_Low)
                
                recon_High = self.model.modelRecon(tf.concat([self.val_High, self.val_Mask_High], -1))
                critic_High = self.model.modelCritic(tf.concat([recon_High, self.val_High, self.val_Mask_High], -1))
                erre_High = self.model.modelERRE(tf.concat([recon_High, self.val_High, self.val_Mask_High], -1))
                recon_High = recon_High[0,:,:,0].numpy()
                critic_High = critic_High[0,:,:,0].numpy()
                erre_High = erre_High[0,:,:,0].numpy()
                error_High = np.abs(self.val_GT_Image-recon_High)
                
                f = plt.figure(figsize=(42,15))
                f.subplots_adjust(top = 0.82)
                f.subplots_adjust(wspace=0.2, hspace=0.2)
                
                ax = plt.subplot2grid((3,3), (0,0))
                ax.plot(self.train_lossReconList, label='Training')
                ax.plot(self.val_lossReconList, label='Validation')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                plt.title('Reconstruction\nTrain: ' + str(round(self.train_lossReconList[-1],8))+'\nVal: ' + str(round(self.val_lossReconList[-1],8)), fontsize=15, fontweight='bold')

                ax = plt.subplot2grid((3,3), (0,1))
                ax.plot(self.train_lossCriticList, label='Training')
                ax.plot(self.val_lossCriticList, label='Validation')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                plt.title('Critic\nTrain: ' + str(round(self.train_lossCriticList[-1],8))+'\nVal: ' + str(round(self.val_lossCriticList[-1],8)), fontsize=15, fontweight='bold')

                ax = plt.subplot2grid((3,3), (0,2))
                ax.plot(self.train_lossERREList, label='Training')
                ax.plot(self.val_lossERREList, label='Validation')
                ax.legend(loc='upper right', fontsize=14)
                ax.set_yscale('log')
                plt.title('ERRE\nTrain: ' + str(round(self.train_lossERREList[-1],8))+'\nVal: ' + str(round(self.val_lossERREList[-1],8)), fontsize=15, fontweight='bold')
                
                ax = plt.subplot2grid((3,6), (1,0))
                im = ax.imshow(self.val_GT_Image, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('Ground-Truth', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (1,1))
                im = ax.imshow(recon_Low, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('Reconstruction - 1%', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (1,2))
                im = ax.imshow(critic_Low, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('Critic Reconstruction - 1%', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (1,3))
                im = ax.imshow(error_Low, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('Actual Error - 1%', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (1,4))
                im = ax.imshow(erre_Low, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('ERRE - 1%', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (1,5))
                im = ax.imshow(self.val_Mask_Low_Image, aspect='auto', vmin=0, vmax=1, interpolation='none', cmap='gray')
                ax.set_title('Measured Mask - 1%', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (2,0))
                im = ax.imshow(self.val_GT_Image, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('Ground-Truth', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (2,1))
                im = ax.imshow(recon_High, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('Reconstruction - 30%', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (2,2))
                im = ax.imshow(critic_High, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('Critic Reconstruction - 30%', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (2,3))
                im = ax.imshow(error_High, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('Actual Error - 30%', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (2,4))
                im = ax.imshow(erre_High, aspect='auto', vmin=0, interpolation='none')
                ax.set_title('ERRE - 30%', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
                ax = plt.subplot2grid((3,6), (2,5))
                im = ax.imshow(self.val_Mask_High_Image, aspect='auto', vmin=0, vmax=1, interpolation='none', cmap='gray')
                ax.set_title('Measured Mask - 30%', fontsize=15, fontweight='bold')
                cbar = f.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
                
            plt.suptitle('Epoch: '+str(epoch)+'     Patience: '+str(self.patience)+'/'+str(self.maxPatience)+'\nBest Loss: '+str(round(self.bestLoss, 5))+' at Epoch: '+str(self.bestEpoch), fontsize=20, fontweight='bold', y = 0.92)
            
            #Save resulting plot
            f.savefig(self.dir_TrainingModelResults + 'epoch_' +str(epoch) + '.tiff', bbox_inches='tight')
            plt.close()
            
            
            
def customTFBar_initialize_progbar(self, hook, epoch, logs=None):
        self.num_samples_seen = 0
        self.steps_to_update = 0
        self.steps_so_far = 0
        self.logs = defaultdict(float)
        self.num_epochs = self.params["epochs"]
        self.mode = "steps"
        self.total_steps = self.params["steps"]
        if hook == "train_overall":
            if self.show_overall_progress:
                self.overall_progress_tqdm = self.tqdm(
                    total=self.num_epochs,
                    bar_format=self.overall_bar_format,
                    leave=self.leave_overall_progress,
                    dynamic_ncols=True,
                    unit="epochs",
                    ascii=asciiFlag
                )
        elif hook == "test":
            if self.show_epoch_progress:
                self.epoch_progress_tqdm = self.tqdm(
                    total=self.total_steps,
                    desc="Evaluating",
                    bar_format=self.epoch_bar_format,
                    leave=self.leave_epoch_progress,
                    dynamic_ncols=True,
                    unit=self.mode,
                    ascii=asciiFlag
                )
        elif hook == "train_epoch":
            if self.show_epoch_progress:
                self.epoch_progress_tqdm = self.tqdm(
                    total=self.total_steps,
                    bar_format=self.epoch_bar_format,
                    leave=self.leave_epoch_progress,
                    dynamic_ncols=True,
                    unit=self.mode,
                    ascii=asciiFlag
                )

def customTFBar_on_epoch_end(self, epoch, logs={}):
    self._clean_up_progbar("train_epoch", logs)
    if self.show_overall_progress:
        metric_value_pairs = []
        for key, value in logs.items():
            if key in ["batch", "size"]: continue
            pair = self.metrics_format.format(name=key, value=value)
            metric_value_pairs.append(pair)
        metrics = self.metrics_separator.join(metric_value_pairs)
        self.overall_progress_tqdm.desc = metrics
        self.overall_progress_tqdm.update(1)

#Replace tqdm progress bar definitions from tensorflow-addons with customized versions
tfa.callbacks.TQDMProgressBar._initialize_progbar = customTFBar_initialize_progbar
tfa.callbacks.TQDMProgressBar.on_epoch_end = customTFBar_on_epoch_end