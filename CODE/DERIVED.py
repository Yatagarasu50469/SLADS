#==================================================================
#DERIVED - DETERMINED GLOBAL VARIABLES
#==================================================================

#Compute precision needed to format strings in training visualizations
maxPatiencePrecision = int(np.floor(np.log10(maxPatience))+1)
maxStagnationPrecision = int(np.floor(np.log10(maxStagnation))+1)
maxEpochsPrecision = int(np.floor(np.log10(maxEpochs))+1)

#Set the axes to process network inputs, RDPP inputs, RDPP, and/or RD (per-channel or global basis)
if processPerChannel: processAxis = (1,2)
else: processAxis = None