#==================================================================
#EXTERNAL SETUP
#==================================================================

#==================================================================
#GENERAL LIBRARY IMPORTS
#==================================================================
from __future__ import absolute_import, division, print_function
import cv2
import contextlib
import copy
import datetime
#import faiss
import gc
import glob
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
import random
import ray
import re
import requests
import sys
import scipy
import skimage
import shutil
import sklearn
import time
import warnings

from IPython import display
from IPython.core.debugger import set_trace as Tracer
from itertools import chain
from joblib import Parallel, delayed
from matplotlib.pyplot import figure
from multiplierz.mzAPI import mzFile
from multiplierz.spectral_process import mz_range
from numba import jit
from PIL import Image
from ray import serve
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
from tqdm.auto import tqdm

matplotlib.use('agg') #Non-interactive plotting mode
sys.coinit_flags = 0 #

#==================================================================
#TENSORFLOW IMPORT AND SETUP
#==================================================================

#Make tensorflow only report errors (3), warnings (2), information (1), all (0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

#Import remaining needed tensorflow libraries
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import image_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op

#==================================================================

#==================================================================
#OS SPECIFIC IMPORTS
#==================================================================
#Determine system operating system
systemOS = platform.system()

#Operating system specific imports 
if systemOS == 'Windows':
    from ctypes import windll, create_string_buffer
    import struct

#==================================================================
