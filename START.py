#                                          █                                            █
#███████╗██╗      █████╗ ██████╗ ███████╗  █  ██████╗ ██╗      █████╗ ██████╗ ███████╗  █   ██████╗ ██╗      █████╗ ███╗   ██╗██████╗ ███████╗
#██╔════╝██║     ██╔══██╗██╔══██╗██╔════╝  █  ██╔══██╗██║     ██╔══██╗██╔══██╗██╔════╝  █  ██╔════╝ ██║     ██╔══██╗████╗  ██║██╔══██╗██╔════╝
#███████╗██║     ███████║██║  ██║███████╗  █  ██║  ██║██║     ███████║██║  ██║███████╗  █  ██║  ███╗██║     ███████║██╔██╗ ██║██║  ██║███████╗
#╚════██║██║     ██╔══██║██║  ██║╚════██║  █  ██║  ██║██║     ██╔══██║██║  ██║╚════██║  █  ██║   ██║██║     ██╔══██║██║╚██╗██║██║  ██║╚════██║
#███████║███████╗██║  ██║██████╔╝███████║  █  ██████╔╝███████╗██║  ██║██████╔╝███████║  █  ╚██████╔╝███████╗██║  ██║██║ ╚████║██████╔╝███████║
#╚══════╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝  █  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝  █   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝
#                                          █                                            █
#

#Current version information
versionNum='0.10.1'

#Import needed libraries for subprocess initialization
import glob
import natsort
import os
import platform
import shutil
import subprocess
import tempfile
from IPython.core.debugger import set_trace as Tracer

#Determine system operating system
systemOS = platform.system()

#Obtain list of configuration files
configFileNames = natsort.natsorted(glob.glob('./CONFIG_*.py'))

#Validate syntax of any currently available configuration files
_ = [exec(open(configFileName, encoding='utf-8').read()) for configFileName in configFileNames]

#Obtain list of currently available configuration files
configFileNames = natsort.natsorted(glob.glob('./CONFIG_*.py'))

#Sequentially run each configuration as a subprocess (otherwise GPU VRAM, when using TensorFlow, is not cleared), create/clear temporary directories, catch and exit on subprocess exceptions
configsRun, continueLoop = [], False
while True: 
    
    #Run any configurations not yet executed
    for configFileName in configFileNames:
        
        #If configuration doesn't exist, signal to rescan available configurations, but not quit
        if not os.path.isfile(configFileName): 
            continueLoop = True
            break
        
        #Try the configuration if it has not previously been run
        if configFileName not in configsRun:
            
            #Note that configuration was (at least) attempted to be run
            configsRun.append(configFileName)
            
            #Create a temporary directory; ray.init doesn't unlock it at shutdown, so has to be done outside the subprocess
            dir_tmp = tempfile.TemporaryDirectory(prefix='TMP_')
            
            #Run the configuration in a subprocess, continuing the loop if it succeeds
            continueLoop=False
            try: 
                if systemOS == 'Windows': process = subprocess.run(["python", "CODE/MAIN.py", configFileName, dir_tmp.name, versionNum], check=True)
                else: process = subprocess.run(["python3", "CODE/MAIN.py", configFileName, dir_tmp.name, versionNum], check=True)
                continueLoop = True
            except: 
                print('\n\nNOTICE: Program has either been manually or unexpectedly terminated. Please wait while shutdown is being performed.\n\n')
            
            #Explicitly remove the temporary directory
            try: shutil.rmtree(dir_tmp)
            except: pass
        
        #Exit the for loop if the last configuration failed
        if not continueLoop: break
    
    #Exit the while loop, if the last configuration failed
    if not continueLoop: break
    
    #Obtain list of currently available configuration files
    configFileNames = natsort.natsorted(glob.glob('./CONFIG_*.py'))
    
    #If all configuration files have been run then exit the loop
    runConfig = False
    for configFileName in configFileNames:
        if configFileName not in configsRun: runConfig = True
    if not runConfig: break
    
#Shutdown python kernel
exit()
