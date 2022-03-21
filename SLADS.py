#====================================================================
#PROGRAM INFORMATION
#====================================================================
#PROGRAM NAME:      SLADS
#
#DATE CREATED:	    4 October 2019
#
#DATE MODIFIED:	    21 March 2021
#
#VERSION NUM:	    0.9.1
#
#LICENSE:           GNU General Public License v3.0
#
#DESCRIPTION:	    Multichannel implementation of SLADS (Supervised Learning 
#                   Algorithm for Dynamic Sampling) and DLADS (Deep Learning
#                   Approach for Dynamic Sampling)
#
#AUTHOR(S):         David Helminiak	EECE, Marquette University
#ADVISOR(S):        Dong Hye Ye		EECE, Marquette University
#
#COLLABORATORS:	    Julia Laskin	CHEM, Purdue University
#                   Hang Hu		    CHEM, Purdue University
#
#FUNDING:	        This project has received funding and was programmed for:
#                   NIH Grant 1UG3HL145593-01
#
#GLOBAL
#CHANGELOG:         0.1.0   Multithreading adjustments to pointwise SLADS
#                   0.1.1   Line constraints, concatenation, pruning, and results organization
#                   0.2.0   Line bounded constraints addition
#                   0.3.0   Complete code rewrite, computational improvements
#                   0.4.0   Class/function segmentation
#                   0.5.0   Overhead reduction; switch multiprocessing package
#                   0.6.0   Modifications for Nano-DESI microscope integration
#                   0.6.1   Model robustness and reduction of memory overhead
#                   0.6.2   Model loading and animation production patches
#                   0.6.3   Start/End point selection with Canny
#                   0.6.4   Custom knn metric, SSIM calc, init computations
#                   0.6.5   Clean variables and resize to physical
#                   0.6.6   SLADS-NET NN, PSNR, and multi-config
#                   0.6.7   Clean asymmetric implementation with density features
#                   0.6.8   Fixed RD generation, added metrics, and Windows compatible
#                   0.7.0   CNN/Unet/RBDN with dynamic window size
#                   0.7.1   c value selection performed before model training
#                   0.7.2   Remove custom pkg. dependency, use NN resize, recon+measured input
#                   0.7.3   Start/End line patch, SLADS(-Net) options, normalization optimization
#                   0.6.9   Do not use -- Original SLADS(-Net) variations for comparison with 0.7.3
#                   0.7.4   CPU compatibility patch, removal of NaN values
#                   0.7.5   c value selection performed before training database generation
#                   0.8.0   Raw MSI file integration (Thermo .raw, Agilent .d), only Windows compatible
#                   0.8.1   Model simplification, method cleanup, mz tolerance/standard patch
#                   0.8.2   Multichannel, fixed groupwise, square pixels, accelerated RD, altered visuals/metrics
#                   0.8.3   Mask seed fix, normalization for sim. fix, non-Ray option, pad instead of resize
#                   0.8.4   Parallel c value selection fix, remove network resizing requirement, fix experimental
#                   0.8.5   Model optimization, enable batch processing, SLADS training fix, database acceleration
#                   0.8.6   Memory reduction, reconstruction vectorization, augmentation, global mz, mz window in ppm
#                   0.8.7   Recon. script, acq. rate, seq. names, live output, offsets, input scaling, Otsu segLine
#                   0.8.8   Interpolation limits, static graph, parallel inferencing, ray deployment, test of FAISS
#                   0.8.9   Simplification
#                   0.9.0   Multichannel E/RD, distributed GPU/batch training, E/RD timing, fix seq. runs
#                   0.9.1   Parallel sample loading, unique model names, post-processing mode, replace avg. mz with TIC
#                   ~0.+.+  GAN, Custom adversarial network, Multimodal integration
#                   ~1.0.0  Initial release
#====================================================================

#==================================================================
#INITIALIZATION
#==================================================================

#Current version information
versionNum='0.9.1'

#Import needed libraries for subprocess initialization
import glob
import natsort
import numpy as np
import sys
import signal
import subprocess

#Obtain list of configuration files
configFileNames = natsort.natsorted(glob.glob('./CONFIG_*.py'))

#If there is more than one configuration file, validate their syntax
if len(configFileNames) > 1: [exec(open(configFileName, encoding='utf-8').read()) for configFileName in configFileNames]

#Run each configuration sequentially as a subprocess (GPU VRAM not cleared by Tensorflow, leading to crash otherwise); pass interrupts to active subprocess
for configFileName in configFileNames: 
    process = subprocess.Popen(["python", "runConfig.py", configFileName, versionNum], shell=False)
    try: process.wait()
    except: 
        process.send_signal(signal.SIGINT)
        exit()
print('ALL CONFIGURATIONS COMPLETE')

#Shutdown python kernel
exit()
