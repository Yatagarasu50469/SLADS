#==================================================================
#EXTERNAL
#==================================================================

#ENVIRONMENTAL VARIABLES
#==================================================================

#Dictionary to store environmental variables; passes to ray workers
environmentalVariables = {}

#Setup deterministic behavior for CUDA operations called by torch; may change in future versions...
#"Set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit overall performance) 
#or :4096:8 (will increase library footprint in GPU memory by approximately 24MiB)."
if manualSeedValue != -1: environmentalVariables["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#Disable HDF5 lock, allowing parallel file access
environmentalVariables["HDF5_USE_FILE_LOCKING"] = "FALSE"

#Set matplotlib backend 
environmentalVariables["MPLBACKEND"] = "agg"

#Allocate memory for CUDA in an asynchronous manner when using TensorFlow; disable this line if GPU compute is < 6.1 (Maxwell and previous)
if 'DLADS-TF' in erdModel : environmentalVariables["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

#Enable deterministic operations in TensorFlow as applicable; may only make `tf.nn.bias_add` operate deterministically
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
    if 'DLADS-TF' in erdModel: environmentalVariables['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
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

import contextlib
import copy
import ctypes
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
import multivolumefile
import natsort
import numpy as np
import opentimspy
import pandas as pd
import pickle
import PIL
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
#from numba import cuda, jit
from numpy import matlib
from opentimspy import OpenTIMS
from PIL import Image
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from ray.util.multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt
from scipy.signal.windows import gaussian
from scipy.stats import binned_statistic
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor as nnr
from sklearn.preprocessing import PolynomialFeatures
from skimage.filters import threshold_otsu
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from tqdm.auto import tqdm

#Imports specific to the configured machine learning package 
if 'DLADS-TF' in erdModel:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import *
    from tensorflow.keras.optimizers import *
    from tensorflow.keras.utils import *
    from tensorflow.python.ops import array_ops, image_ops
    from tensorflow.keras.layers.experimental.preprocessing import PreprocessingLayer
elif 'DLADS-PY' in erdModel or 'GLANDS' in erdModel: 
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
if 'DLADS-PY' in erdModel or 'GLANDS' in erdModel: exec(open("./CODE/PCONV2D.py", encoding='utf-8').read())

#LIBRARY AND WARNING SETUP
#==================================================================

#Benchmarks algorithms and uses the fastest; only recommended if input sizes are consistent
#if 'DLADS-PY' in erdModel or 'GLANDS' in erdModel: torch.backends.cudnn.benchmark = True

#Allow anomaly detection in training a PyTorch model; sometimes needed for debugging
#if 'DLADS-PY' in erdModel or 'GLANDS' in erdModel: torch.autograd.set_detect_anomaly(True)

#Setup logging configuration
setupLogging()

#Change method of instantiation for COM objects
sys.coinit_flags = 0 

#Setup deterministic behavior
if (manualSeedValue != -1): 
    if 'DLADS-TF' in erdModel: tf.config.experimental.enable_op_determinism()
    if 'DLADS-PY' in erdModel or 'GLANDS' in erdModel: torch.use_deterministic_algorithms(True, warn_only=False)

#OS SPECIFIC
#==================================================================

#Determine system operating system
systemOS = platform.system()

#Operating system specific imports 
if systemOS == 'Windows':
    from ctypes import windll, create_string_buffer
    import struct
    
#==================================================================

