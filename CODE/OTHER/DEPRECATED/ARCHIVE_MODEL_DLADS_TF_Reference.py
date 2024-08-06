#==================================================================
#TEMPORARY REFERENCE
#Needed to form the model differently in order to incorporate dynamic padding sizes.
#Can and eventually did just wrap existing/prior unet definition to acheive the same.
#Nevertheless, in case an alternate coding style is more useful in understanding the architecture,
#a copy shall be included with at least one published version of the program. 
#==================================================================

class OLD_PADDED_UNET(tf.keras.Model):
    
    def __init__(self, numFilters):
        super().__init__()
        
        self.actLeak = LeakyReLU(alpha=0.2)
        self.maxPool = MaxPool2D(pool_size=(2,2))
        
        self.conv0_0 = Conv2D(numFilters, 1, padding='same', use_bias=False)
        self.conv0_1 = Conv2D(numFilters, 3, padding='same', use_bias=False)
        
        self.conv1_0 = Conv2D(numFilters*2, 1, padding='same', use_bias=False)
        self.conv1_1 = Conv2D(numFilters*2, 3, padding='same', use_bias=False)
        
        self.conv2_0 = Conv2D(numFilters*4, 1, padding='same', use_bias=False)
        self.conv2_1 = Conv2D(numFilters*4, 3, padding='same', use_bias=False)
        
        self.conv3_0 = Conv2D(numFilters*8, 1, padding='same', use_bias=False)
        self.conv3_1 = Conv2D(numFilters*8, 3, padding='same', use_bias=False)
        
        self.conv4_0 = Conv2D(numFilters*16, 1, padding='same', use_bias=False)
        self.conv4_1 = Conv2D(numFilters*16, 3, padding='same', use_bias=False)
        
        self.conv5_0 = Conv2D(numFilters*16, 2, activation='relu', padding='same', use_bias=False)
        self.conv5_1 = Conv2D(numFilters*8, 1, activation='relu', padding='same', use_bias=False)
        self.conv5_2 = Conv2D(numFilters*8, 3, activation='relu', padding='same', use_bias=False)
        
        self.conv6_0 = Conv2D(numFilters*8, 2, activation='relu', padding='same', use_bias=False)
        self.conv6_1 = Conv2D(numFilters*4, 1, activation='relu', padding='same', use_bias=False)
        self.conv6_2 = Conv2D(numFilters*4, 3, activation='relu', padding='same', use_bias=False)
    
        self.conv7_0 = Conv2D(numFilters*4, 2, activation='relu', padding='same', use_bias=False)
        self.conv7_1 = Conv2D(numFilters*2, 1, activation='relu', padding='same', use_bias=False)
        self.conv7_2 = Conv2D(numFilters*2, 3, activation='relu', padding='same', use_bias=False)
        
        self.conv8_0 = Conv2D(numFilters*2, 2, activation='relu', padding='same', use_bias=False)
        self.conv8_1 = Conv2D(numFilters, 1, activation='relu', padding='same', use_bias=False)
        self.conv8_2 = Conv2D(numFilters, 3, activation='relu', padding='same', use_bias=False)
    
        self.convOut = Conv2D(1, 1, padding='same', use_bias=False)
        
    def call(self, inputs):
        
        nshape = tuple(inputs.shape[1:3].as_list())
        padHeight, padWidth = (-(-nshape[0]//16)*16)-nshape[0], (-(-nshape[1]//16)*16)-nshape[1]
        inputs = tf.pad(inputs, tf.constant([[0, 0], [padHeight, 0], [padWidth, 0], [0, 0]]))
        
        conv0 = self.actLeak(self.conv0_0(inputs))
        conv0 = self.actLeak(self.conv0_1(conv0))
        
        conv1 = self.maxPool(conv0)
        conv1 = self.actLeak(self.conv1_0(conv1))
        conv1 = self.actLeak(self.conv1_1(conv1))
        
        conv2 = self.maxPool(conv1)
        conv2 = self.actLeak(self.conv2_0(conv2))
        conv2 = self.actLeak(self.conv2_1(conv2))
        
        conv3 = self.maxPool(conv2)
        conv3 = self.actLeak(self.conv3_0(conv3))
        conv3 = self.actLeak(self.conv3_1(conv3))
        
        conv4 = self.maxPool(conv3)
        conv4 = self.actLeak(self.conv4_0(conv4))
        conv4 = self.actLeak(self.conv4_1(conv4))
        
        conv5 = concatenate([conv3, customResize(conv4, conv3)])
        conv5 = self.conv5_0(conv5)
        conv5 = self.conv5_1(conv5)
        
        conv6 = concatenate([conv2, customResize(conv5, conv2)])
        conv6 = self.conv6_0(conv6)
        conv6 = self.conv6_1(conv6)
        
        conv7 = concatenate([conv1, customResize(conv6, conv1)])
        conv7 = self.conv7_0(conv7)
        conv7 = self.conv7_1(conv7)
        
        conv8 = concatenate([conv0, customResize(conv7, conv0)])
        conv8 = self.conv8_0(conv8)
        conv8 = self.conv8_1(conv8)
        
        out = self.convOut(conv8)
        
        return out[:, padHeight:, padWidth:, :]