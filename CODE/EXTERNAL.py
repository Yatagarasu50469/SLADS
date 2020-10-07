#==================================================================
#EXTERNAL SLADS SETUP
#==================================================================

#==================================================================
#LIBRARY IMPORTS
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
import time
import warnings
warnings.filterwarnings("ignore")

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
from sklearn import linear_model
from sklearn import svm
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor as nnr
from sklearn.preprocessing import *
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from skimage.util import view_as_windows as viewW

from skimage import filters
from skimage.filters import *
from skimage.measure import compare_psnr
from skimage.metrics import structural_similarity
from sobol import *
from tqdm.auto import tqdm

#Determine system operating system
systemOS = platform.system()

#Operating system specific imports 
if systemOS == 'Windows':
    from ctypes import windll, create_string_buffer
    import struct


#==================================================================
