#==================================================================
#EXTERNAL LIBRARY IMPORTS AND SETUP
#==================================================================

#GENERAL LIBRARY IMPORTS AND SETUP
#==================================================================

from __future__ import absolute_import, division, print_function
import alphatims
import colorama
import cv2
import contextlib
import copy
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
import natsort
import numpy as np
import numpy.matlib as matlib
import os
import pandas as pd
import pickle
import PIL
import PIL.ImageOps
import platform
import psutil
import pyimzml
import random
import re
import requests
import sys
import scipy
import skimage
import shutil
import sklearn
import time
import warnings

from bisect import bisect_left, bisect_right
from collections import defaultdict
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
from typeguard import typechecked
from sobol import *
from tqdm.auto import tqdm

matplotlib.use('Agg') #Non-interactive plotting mode
sys.coinit_flags = 0 #Change method of instantiation for COM objects


#TENSORFLOW/RAY IMPORTS AND SETUP
#==================================================================

#Make tensorflow only report errors (3), warnings (2), information (1), all (0) (disable for debug)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Allocate memory for CUDA in an asynchronous manner; disable this line if GPU compute is < 6.1 (Maxwell and previous)
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

#Disable Ray memory monitor as it will sometimes decide to kill processes with suprising/unexpected/unmanageable/untracable errors
#Default appears to be killing jobs at 95% system memory capacity
#os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'

#Stop Ray from crashing the program when errors occur (otherwise may crash despite being handled by try/catch!)
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

#Prevent Ray from printing spill logs
os.environ["RAY_verbose_spill_logs"] = "0"

#Disable HDF5 lock, allowing parallel access
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#After enivronmental settings can finish imports
import ray
from ray import serve
from ray.util.multiprocessing import Pool

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.data.experimental import DistributeOptions, AutoShardPolicy
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import *
from tensorflow.python.data.util import options as options_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

#Further restrict warning/logging levels to only report errors (disable for debug)
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
logging.root.setLevel(logging.ERROR)

#Import remaining alphatims after logging level has been set
import alphatims.bruker

#Turn off alphatims internal tqdm callback
alphatims.utils.set_progress_callback(None)

#OS SPECIFIC IMPORTS
#==================================================================
#Determine system operating system
systemOS = platform.system()

#Operating system specific imports 
if systemOS == 'Windows':
    from ctypes import windll, create_string_buffer
    import struct

#==================================================================