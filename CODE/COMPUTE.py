#==================================================================
#COMPUTE RESOURCE SPECIFICATION
#==================================================================

#Reset ray memory and compute; shouldn't be needed but is since pools do not seem to close properly (set log_to_driver=False to stop all PID messages)
def resetRay(numberCPUS):
    ray.shutdown()
    ray.init(num_cpus=numberCPUS, configure_logging=True, logging_level=logging.ERROR, include_dashboard=False, log_to_driver=True)

#Limit GPU(s) if indicated
if availableGPUs != 'None': os.environ["CUDA_VISIBLE_DEVICES"] = availableGPUs
gpus = tf.config.list_physical_devices('GPU')
numGPUs = len(gpus)

#For model inferencing with DLADS and GLANDS, assign 1 GPU per server/actor (so as to potentially allow for multiple), otherwise assign 0
if (erdModel == 'DLADS' or erdModel == 'GLANDS') and numGPUs>0: modelGPUs = 1
else: modelGPUS = 0

#Set how many cpu threads are to be used in parallel, disabling if there is only one thread remaining
if parallelization: 
    numberCPUS = multiprocessing.cpu_count()-reserveThreadCount
    if numberCPUS <= 1: parallelization = False
if not parallelization: numberCPUS = 1

#Reset/startup ray
resetRay(numberCPUS)

#Allow partial GPU memory allocation; allows better analysis of utilization
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

#If the number of gpus to be used is greater than 1, then increase the configuration's batch size accordingly to accomodate the distribution strategy
if len(gpus)>1: batchSize*=len(gpus)
