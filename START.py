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
versionNum='0.10.0'

#Import needed libraries for subprocess initialization
import glob
import natsort
import subprocess

#Obtain list of configuration files
configFileNames = natsort.natsorted(glob.glob('./CONFIG_*.py'))

#Validate syntax of any configuration files
_ = [exec(open(configFileName, encoding='utf-8').read()) for configFileName in configFileNames]

#Sequentially run each configuration as a subprocess (otherwise GPU VRAM is not cleared); catch and exit on subprocess exceptions
for configFileName in configFileNames: 
    try: process = subprocess.run(["python", "CODE/MAIN.py", configFileName, versionNum], check=True)
    except subprocess.CalledProcessError as err: exit()

#Shutdown python kernel
exit()
