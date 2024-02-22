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

#Disable HDF5 lock, allowing parallel access
environmentalVariables["HDF5_USE_FILE_LOCKING"] = "FALSE"

#Set matplotlib backend 
environmentalVariables["MPLBACKEND"] = "agg"

#Disable Ray memory monitor as it will sometimes decide to kill processes with suprising/unexpected/unmanageable/untracable errors
#Default appears to be killing jobs at 95% system memory capacity
#environmentalVariables["RAY_DISABLE_MEMORY_MONITOR"] = "1"

if not debugMode: 

    #Stop Ray from crashing the program when errors occur (otherwise may crash despite being handled by try/catch!)
    environmentalVariables["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

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
import colorama
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
import multiprocessing
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
import re
import requests
import scipy
import skimage
import shutil
import sklearn
import time
import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms
import warnings

from bisect import bisect_left, bisect_right, bisect
from collections import defaultdict
from contextlib import nullcontext
from contextlib import redirect_stdout
from IPython import display
from IPython.core.debugger import set_trace as Tracer
from itertools import chain
from joblib import Parallel, delayed
from matplotlib.pyplot import figure
from multiplierz.mzAPI import mzFile
from multiplierz.spectral_process import mz_range
from numba import cuda
from numba import jit
from PIL import Image
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from ray import serve
from ray.util.multiprocessing import Pool
from scipy import misc
from scipy import signal
from scipy.io import loadmat
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.special import softmax
from sklearn import linear_model
from sklearn import svm
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor as nnr
from sklearn.preprocessing import *
from sklearn.utils import shuffle
from skimage.util import view_as_windows as viewW
from skimage import filters
from skimage.filters import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.transform import resize
from sobol import *
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from typeguard import typechecked
from tqdm.auto import tqdm

#LIBRARY AND WARNING SETUP
#==================================================================

#Change method of instantiation for COM objects
sys.coinit_flags = 0 

#Restrict logging/warning levels (disable for debug
if not debugMode: 
    warnings.filterwarnings("ignore")
    logging.root.setLevel(logging.ERROR)

#Turn off alphatims internal tqdm callback
alphatims.utils.set_progress_callback(None)

#Setup deterministic behavior for torch (this alone does not affect CUDA-specific operations)
if manualSeedValue != -1: torch.use_deterministic_algorithms(True)

#OS SPECIFIC
#==================================================================

#Determine system operating system
systemOS = platform.system()

#Operating system specific imports 
if systemOS == 'Windows':
    from ctypes import windll, create_string_buffer
    import struct

#==================================================================
