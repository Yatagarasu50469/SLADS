#==================================================================
#EXTERNAL SLADS SETUP
#==================================================================

#==================================================================
#GENERAL LIBRARY IMPORTS
#==================================================================
from __future__ import absolute_import, division, print_function
import cv2
import contextlib
import copy
import gc
import glob
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
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
import ray
import re
import random
import sys
import scipy
import shutil
import sklearn
import time
import warnings

from datetime import datetime
from IPython import display
from IPython.core.debugger import Tracer
from joblib import Parallel, delayed
from matplotlib.pyplot import figure
from PIL import Image
from scipy import misc
from scipy import signal
from scipy.io import loadmat
from scipy.io import savemat
from scipy.signal import find_peaks
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
from skimage.metrics import structural_similarity
from skimage.transform import resize
from sobol import *
from tqdm.auto import tqdm

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
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

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
