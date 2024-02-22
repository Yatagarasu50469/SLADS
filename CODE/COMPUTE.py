#==================================================================
#COMPUTE
#==================================================================

#Reset ray memory and compute; shouldn't be needed but is since pools do not seem to close and free memory properly (set log_to_driver=False to stop all PID messages)
def resetRay(numberCPUS):
    if ray.is_initialized(): rayUp = True
    else: rayUp = False
    while rayUp:
        try: 
            ray.shutdown()
            rayUp = False
        except: 
            print('\nWarning - Ray failed to shutdown correctly, if this message repeatedly appears sequentially, exit the program with CTL+c.')
    while not rayUp: 
        try: 
            _ = ray.init(num_cpus=numberCPUS, logging_level=logging.root.level, runtime_env={"env_vars": environmentalVariables}, include_dashboard=False)
            rayUp = True
        except:
            print('\nWarning - Ray failed to startup correctly, if this message repeatedly appears sequentially, exit the program with CTL+c.')

#Store string of all system GPUs (Ray hides them)
systemGPUs = ", ".join(map(str, [*range(torch.cuda.device_count())]))

#Note GPUs available/specified
if not torch.cuda.is_available(): gpus = []
if (len(gpus) > 0) and (gpus[0] == -1): gpus = [*range(torch.cuda.device_count())]
numGPUs = len(gpus)

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
