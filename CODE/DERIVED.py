#==================================================================
#DERIVED - DETERMINED GLOBAL VARIABLES
#==================================================================

#Compute precision needed to format strings in training visualizations
maxPatiencePrecision = int(np.floor(np.log10(maxPatience))+1)
maxStagnationPrecision = int(np.floor(np.log10(maxStagnation))+1)
maxEpochsPrecision = int(np.floor(np.log10(maxEpochs))+1)