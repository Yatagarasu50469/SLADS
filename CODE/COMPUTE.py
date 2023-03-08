#==================================================================
#COMPUTE RESOURCE SPECIFICATION
#==================================================================

#Reset ray memory and compute; shouldn't be needed but is since pools do not seem to close properly (set log_to_driver=False to stop all PID messages)
#runtime_env={"env_vars": {"PYTHONWARNINGS": "ignore"}} doesn't work
def resetRay(numberCPUS):
    ray.shutdown()
    if ray.is_initialized(): sys.exit('Error - Ray was still initialized after calling shutdown. Please try running the code again, if this message appears a second time, you may need to restart your machine.')
    ray.init(num_cpus=numberCPUS, configure_logging=True, logging_level=logging.ERROR, include_dashboard=False, log_to_driver=True)

#Limit GPU(s) if indicated
if availableGPUs != 'None': os.environ["CUDA_VISIBLE_DEVICES"] = availableGPUs
gpus = tf.config.list_physical_devices('GPU')
numGPUs = len(gpus)

#For model inferencing with DLADS and GLANDS, assign 1 GPU per server/actor (so as to potentially allow for multiple), otherwise assign 0
if (erdModel == 'DLADS' or erdModel == 'GLANDS') and numGPUs>0: modelGPUs = 1
else: modelGPUs = 0

#Detect logical and physical core counts, determining if hyperthreading is active
logicalCountCPU = psutil.cpu_count(logical=True)
physicalCountCPU = psutil.cpu_count(logical=False)
if logicalCountCPU > physicalCountCPU: hyperthreading=True
else: hyperthreading = False

#Set parallel CPU usage limit, disabling if there is only one thread remaining
#Ray documentation indicates num_cpus should be out of the number of logical cores/threads
#In practice, specifying a number closer to, or just below, the count of physical cores maximizes performance
#Any ray.remote calls need to specify num_cpus to set environmental OMP_NUM_THREADS variable correctly
if parallelization: 
    if availableThreads==0: numberCPUS = physicalCountCPU
    else: numberCPUS = availableThreads
    if numberCPUS <= 1: parallelization = False
if not parallelization: numberCPUS = 1

#Reset/startup ray
resetRay(numberCPUS)

#Limit the number of threads to use with alphatims(0 = all, -# = how many to leave available))
#Variable only ever used for one call in alphatims code: bruker_dll.tims_set_num_threads()
#A comment in alphatims code implies it doesn't appear to actually do anything for I/O operations
#This was found to impact multiplierz which, disimilar to alphatims, uses the official Bruker SDK. 
#alphatims.utils.set_threads(0)

#Allow partial GPU memory allocation; allows better analysis of utilization
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

#If the number of gpus to be used is greater than 1, then increase the configuration's batch size accordingly to accomodate the distribution strategy
if len(gpus)>1: batchSize*=len(gpus)
