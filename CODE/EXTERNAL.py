#==================================================================
#EXTERNAL
#==================================================================

#ENVIRONMENTAL VARIABLES
#==================================================================

#Dictionary to store environmental variables; passes to ray workers
environmentalVariables = {}

#Setup deterministic behavior for CUDA; may change in future versions...
#"Set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit overall performance) 
#or :4096:8 (will increase library footprint in GPU memory by approximately 24MiB)."
if manualSeedValue != -1: environmentalVariables["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#Disable HDF5 lock, allowing parallel file access
environmentalVariables["HDF5_USE_FILE_LOCKING"] = "FALSE"

#Set matplotlib backend 
environmentalVariables["MPLBACKEND"] = "agg"

#Allocate memory for CUDA in an asynchronous manner when using TensorFlow; disable this line if GPU compute is < 6.1 (Maxwell and previous)
if erdModel == 'DLADS-TF': environmentalVariables["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

#Enable deterministic operations in TensorFlow as applicable
if manualSeedValue != -1 and erdModel == 'DLADS-TF': environmentalVariables["TF_DETERMINISTIC_OPS"] = "1"

#Increase memory usage threshold for Ray from the default 95%
environmentalVariables["RAY_memory_usage_threshold"] = "0.99"

#Disable Ray memory monitor as it will sometimes decide to kill processes with suprising/unexpected/unmanageable/untracable errors
#environmentalVariables["RAY_DISABLE_MEMORY_MONITOR"] = "1"

if debugMode: 

    #Ray deduplicates logs by default; sometimes verbose output is needed in debug
    environmentalVariables["RAY_DEDUP_LOGS"] = "0"
    
else:
    
    #Have TensorFlow only report errors ('3'), warnings ('2'), information ('1'), all ('0')
    if erdModel == 'DLADS-TF': environmentalVariables['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    #Stop Ray from crashing the program when errors occur (otherwise may crash despite being handled by try/catch!)
    environmentalVariables["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
    
    #Change Ray filter level
    environmentalVariables["RAY_LOG_TO_DRIVER_EVENT_LEVEL"] = "ERROR"

    #Prevent Ray from printing spill logs
    environmentalVariables["RAY_verbose_spill_logs"] = "0"

    #Restrict warning levels to only report errors; ONLY applicable to subprocesses!
    environmentalVariables["PYTHONWARNINGS"] = "ignore"

#Load enivronmental variables into current runtime before other imports
import os
for var, value in environmentalVariables.items(): os.environ[var] = value

#IMPORTS
#==================================================================

import alphatims
import alphatims.bruker
import contextlib
import copy
import cv2
import datetime
import gc
import glob
import h5py
import joblib
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import multiplierz
import multiplierz.mzAPI.raw as raw
import multivolumefile
import natsort
import numpy as np
import numpy.matlib as matlib
import pandas as pd
import pickle
import PIL
import PIL.ImageOps
import platform
import psutil
import py7zr
import pyimzml
import random
import ray
import scipy
import skimage
import shutil
import sklearn
import sys
import time
import warnings

from bisect import bisect_left, bisect_right, bisect
from collections import defaultdict
from contextlib import nullcontext
from IPython.core.debugger import set_trace as Tracer
from joblib import Parallel, delayed
from matplotlib.pyplot import figure
from multiplierz.mzAPI import mzFile
from multiplierz.spectral_process import mz_range
from numba import cuda
from numba import jit
from PIL import Image
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from ray.util.multiprocessing import Pool
from scipy import signal
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor as nnr
from sklearn.preprocessing import *
from skimage import filters
from skimage.filters import *
from skimage.metrics import mean_squared_error as compare_MSE
from skimage.metrics import structural_similarity as compare_SSIM
from skimage.metrics import peak_signal_noise_ratio as compare_PSNR
from skimage.transform import resize
from tqdm.auto import tqdm

#Imports specific to the configured machine learning package 
if erdModel == 'DLADS-TF':
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import *
    from tensorflow.keras.optimizers import *
    from tensorflow.keras.utils import *
    from tensorflow.python.ops import array_ops, image_ops
    from tensorflow.keras.layers.experimental.preprocessing import PreprocessingLayer
else: 
    import torch
    import torch.nn.functional as F
    import torch.nn.parallel
    import torchvision.transforms as transforms
    from torch import nn
    from torch import optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import datasets
    from torchvision.transforms import v2
    from torchvision.transforms.functional import InterpolationMode
    
#Additional external definitions to import through execution
exec(open("./CODE/LOGGING.py", encoding='utf-8').read())
if erdModel != 'DLADS-TF': exec(open("./CODE/PCONV2D.py", encoding='utf-8').read())

#LIBRARY AND WARNING SETUP
#==================================================================

#Benchmarks algorithms and uses the fastest; only recommended if input sizes are consistent
#if erdModel != 'DLADS-TF': torch.backends.cudnn.benchmark = True

#Allow anomaly detection in training a PyTorch model; sometimes needed for debugging
#if erdModel != 'DLADS-TF': torch.autograd.set_detect_anomaly(True)

#Setup logging configuration
setupLogging()

#Change method of instantiation for COM objects
sys.coinit_flags = 0 

#Turn off alphatims internal tqdm callback
alphatims.utils.set_progress_callback(None)

#Setup deterministic behavior for torch (this alone does not affect CUDA-specific operations)
if (manualSeedValue != -1) and (erdModel != 'DLADS-TF'): torch.use_deterministic_algorithms(True, warn_only=False)

#OS SPECIFIC
#==================================================================

#Determine system operating system
systemOS = platform.system()

#Operating system specific imports 
if systemOS == 'Windows':
    from ctypes import windll, create_string_buffer
    import struct

#==================================================================

