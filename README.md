
     ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄ 
	▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌
	▐░█▀▀▀▀▀▀▀▀▀ ▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ 
	▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          
	▐░█▄▄▄▄▄▄▄▄▄ ▐░▌          ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄ 
	▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
	 ▀▀▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌ ▀▀▀▀▀▀▀▀▀█░▌
	          ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌          ▐░▌
	 ▄▄▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌
	▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌
	 ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀	
#

# GENERAL INFORMATION


    NAME: 		SLADS
    VERSION NUM:	0.7.1
    DESCRIPTION:	Multichannel implementation of SLADS (Supervised Learning Algorithm 
			for Dynamic Sampling with additional constraint to select groups of 
			points along a single axis. 
    
    AUTHORS:	David Helminiak	EECE, Marquette University
    		Dong Hye Ye	EECE, Marquette University
    
    COLLAB. 	Julia Laskin	CHEM, Purdue University
    		Hang Hu		CHEM, Purdue University
    
    FUNDING:	This project has received funding and was programmed for:
    		NIH Grant 1UG3HL145593-01
    
	GLOBAL
	CHANGELOG:	0.1     Multithreading adjustments to pointwise SLADS
		        0.1.1   Line constraints, concatenation, pruning, and results organization			
		        0.2     Line bounded constraints addition
	                0.3	Complete code rewrite, computational improvements
	                0.4	Class/function segmentation
	                0.5	Overhead reduction; switch multiprocessing package
	                0.6	Modifications for Nano-DESI microscope integration
                    0.6.1   Model robustness and reduction of memory overhead
                    0.6.2   Model loading and animation production patches
                    0.6.3   Start/End point selection with Canny
                    0.6.4   Custom knn metric, SSIM calc, init computations
                    0.6.5   Clean variables and resize to physical
                    0.6.6   SLADS-NET NN, PSNR, asymFinal, and multi-config
                    0.6.7   Clean asymmetric implementation with density features
                    0.6.8   Fixed RD generation, added metrics, and Windows compatible
                    0.7     CNN/Unet/RBDN with dynamic window size
                    0.7.1    c value selection performed before model training
                    ~0.8    Multichannel integration
                    ~0.9    Tissue segmentation
                    ~1.0    Initial release

# PROGRAM FILE STRUCTURE
**Note:** If testing/training is not to be performed, then contents of 'TEST', 'TRAIN', may be disregarded, but a trained SLADS model must be present in: ./RESULTS/TRAIN/.

**Warning:** If training is enabled, any model already in ./RESULTS/TRAIN/ will be overwritten. Likewise, if testing or implementation is enabled, then any data in ./RESULTS/TEST/ and/or ./RESULT/IMP/, respectively will be overwritten.

    ------->ROOT_DIR
    	|------->README
    	|------->CONFIG_0.py
    	|------->SLADS.py
    	|------->CODE
    	|	|------->DEFS.py
    	|	|------->EXPERIMENTAL.py
    	|	|------->EXTERNAL.py
    	|	|------->INTERNAL.py
    	|	|------->TESTING.py
    	|	|------->TRAINING.py
    	|------->INPUT
    	|	|------->TEST
    	|	|	|------->TEST_SAMPLE_1
    	|	|	|	|-------> mzlowMZ1_highMZ1fnsampleNamensnumRows.csv
    	|	|	|	|-------> mzlowMZ2_highMZ2fnsampleNamensnumRows.csv
    	|	|	|------->TEST_SAMPLE_2
    	|	|	|	|-------> mzlowMZ1_highMZ1fnsampleNamensnumRows.csv
    	|	|	|	|-------> mzlowMZ2_highMZ2fnsampleNamensnumRows.csv
    	|	|------->TRAIN
    	|	|	|------->TRAIN_SAMPLE_1
    	|	|	|	|-------> mzlowMZ1_highMZ1fnsampleNamensnumRows.csv
    	|	|	|	|-------> mzlowMZ2_highMZ2fnsampleNamensnumRows.csv
    	|	|	|------->TEST_SAMPLE_2
    	|	|	|	|-------> mzlowMZ1_highMZ1fnsampleNamensnumRows.csv
    	|	|	|	|-------> mzlowMZ2_highMZ2fnsampleNamensnumRows.csv
    	|	|------->IMP
    	|	|	|-------> mzlowMZ1_highMZ1fnsampleNamensnumRows.csv
    	|	|	|-------> mzlowMZ2_highMZ2fnsampleNamensnumRows.csv
    	|	|	|-------> UNLOCK
    	|	|	|-------> LOCK
    	|	|	|-------> DONE
    	|------->RESULTS
    	|	|------->TEST
    	|	|	|------->Animations
    	|	|	|------->dataPrintout.csv
    	|	|	|------->mzResults
    	|	|	|	|------->TEST_SAMPLE_1
    	|	|	|	|	|-------> lowMZ1_highMZ1.png
    	|	|	|	|	|-------> lowMZ2_highMZ2.png
    	|	|	|	|------->TEST_SAMPLE_2
    	|	|	|	|	|-------> lowMZ1_highMZ1.png
    	|	|	|	|	|-------> lowMZ2_highMZ2.png
    	|	|	|------->testingAverageSSIM_Percentage.csv
    	|	|	|------->testingAverageSSIM_Percentage.png
    	|	|------->TRAIN
    	|	|	|------->bestC.npy
    	|	|	|------->bestTheta.npy
    	|	|	|------->cValues.npy
    	|	|	|------->Images
    	|	|	|	|-------N/A
    	|	|	|------->trainedModels.npy


# INSTALLATION
This implementation of SLADS has functioned on Windows, Mac, and Linux operating systems. The package versions do not necessarily need to match with those listed, but should the program produce unexpected errors, installing a specific version of a package might be able to resolve the issue. 

	Operating System
		Win. 10:		Updated as of Jan 1 2020
		Ubuntu:			18.04
		Mac: 			10.13.6

	System
		Python			3.8.5
		pip			19.3.1

	Python Packages
		backcall: 		0.1.0
		colorama:		0.4.3
		cycler: 		0.10.0 
		decorator: 		4.4.1
		dill: 			0.3.1.1
		datetime:		4.3
		glob3:			0.0.1
		imageio: 		2.6.1
		IPython: 		7.11.1
		ipython-genutils: 	0.2.0
		jedi: 			0.15.2
		joblib:			0.14.1
		kiwisolver: 		1.1.0
		pandas:			0.25.3
		parso: 			0.5.2
		python-dateutil:	2.8.1
		numpy:			1.18.1
		matplotlib: 		3.1.2
		multiprocess: 		0.70.9 
		natsort: 		6.2.0
		networkx: 		2.4
		opencv-python:	 	4.1.2.30
		pathlib:         	1.0.1
		pickleshare: 		0.7.5
		pillow:			7.0.0
		prompt-toolkit: 	3.0.2
		psutil:			5.6.7
		pygments: 		2.5.2
		pyparsing: 		2.4.6
		pytz:			2019.3
		PyWavelets: 		1.1.1
		ray:			1.0.0
		scipy:			1.4.1
		six: 			1.13.0
		scikit-image: 		0.16.2
		scikit-learn: 		0.22.1
		sklearn:		0.0
		sobol: 			0.9
		sobol_seq:		0.1.2
		traitlets: 		4.3.3
		tqdm:			4.41.1
		wcwidth: 		0.1.8
		zope.interface		4.7.1

### **Installation on Mac OSX 10.13**
**Note:** These instructions have not been tested on a clean system running only the operating systems specified, but should be expected to function.

	$ python3 --m pip install --upgrade pip
	$ pip3 install opencv-python datetime glob3 IPython joblib pandas psutil matplotlib pillow ray scipy sobol sobol_seq natsort multiprocess ray scikit-image sklearn tqdm
	$ cd ./CODE/scikit-learn
	$ make clean
	$ pip3 install --verbose --no-build-isolation --editable .

TensorFlow is not currently officially supported on OS X machines. Although official support may become available in the  future, installation and patch instructions for Tensorflow v2.2.0 may be found at https://github.com/TomHeaven/tensorflow-osx-build. These instructions have been confirmed to work on a OS X build with a 1080 TI GPU. Instructions for v2.3.0 are not currently functional with Python 3.8. 

### **Installation on Ubuntu 18.04**
**Note:** These instructions have not been tested on a clean system running only the operating systems specified, but should be expected to function.
	
	$ sudo apt-get update
	$ sudo apt-get install python3-pip
	$ python3 -m pip install --upgrade pip
	$ pip3 install opencv-python datetime glob3 IPython joblib pandas psutil matplotlib pillow ray setuptools scipy sobol sobol_seq tensorflow natsort multiprocess scikit-image sklearn tqdm
	$ cd ./CODE/scikit-learn
	$ make clean
	$ pip3 install --verbose --no-build-isolation --editable .


### **Installation on Windows 10**
**Note:** At this time, the multiprocessing package **ray** is only experimentally functional, unexpected behaviors may occur during startup/shutdown of the program

Install Visual Studio Community 2019, with the additional options: "Desktop development with C++" and "Python development":  https://visualstudio.microsoft.com/downloads/
Also install the "Build Tools for Visual Studio 2019", under "All downloads"/"Tools for Visual Studio 2019"

Install Python 3.8.3: choosing to install with advanced options, selecting to install for all users, and add python to environment variables

Open the command prompt as an administrator (right-click on the command prompt icon and choose "Run as administrator")

For 64-bit Python, configure the build environment by entering the following lines into command prompt, where for 32-bit replace x64 by x86.
Note that the actual location of the specified file may vary depending on potential past Visual Studio installations. 

	$ SET DISTUTILS_USE_SDK=1
	$ "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

Navigate inside the command prompt to the SLADS base directory then enter the following commands:

	$ cd CODE\scikit-learn
	$ python3 -m pip install --upgrade pip
	$ pip3 install cython wheel numpy
	$ pip3 install --verbose --no-build-isolation --editable .
	$ pip3 install jupyter datetime glob3 IPython joblib pandas pathlib psutil matplotlib pillow ray scipy sobol sobol_seq natsort multiprocess scikit-image tqdm tensorflow opencv-python pydot graphviz 

# TRAINING/TESTING PROCEDURE

###  PRE-PROCESSING
Assuming the desired equipment intended for end application of SLADS, outputs Thermo-Finnigan .RAW files, the data will need to be pre-processed prior to training a SLADS model. Each data sample must be aligned between rows using a simple linear interpolation scheme according to the unique acquisition times. Each individual mz range of interest must then be exported as a .csv file. For example, given a singular sample: **sampleName**, with a number of rows: **numRows** (corresponding to the number of acquired .RAW files, or number intended for acquisition),  and a set of mz ranges: **lowMZ1**-**highMZ1**, **lowMZ2**-**highMZ2**, etc., a folder should be created which contains files named according to the convention: 

	mzlowMZ_highMZfnsampleNamensnumRows.csv

Please note that at this time in development, each mz value should be specified with exactly 8 characters. Therefore, the following sample parameters:

    sampleName: 	Slide1-Wnt-3
    numRows: 	72
    lowMZ1: 	454.8740
    highMZ1: 	454.8922
    lowMZ2: 	454.8844
    highMZ2: 	454.9026

should yield the following folder hierarchy:

    -------> Slide1-Wnt-3
    	|-------> mz454.8740_454.8922fnSlide1-Wnt-3ns72.csv
    	|-------> mz454.8844_454.9026fnSlide1-Wnt-3ns72.csv

Each of these folders may then be placed either into ./INPUT/TEST, or ./INPUT/TRAIN as desired. An example of the desired multichannel sample input format is included in the ./EXAMPLE/ folder. 

###  **CONFIGURATION**
**Warning:** This section is no longer entirely up to date, refer to ./CONFIG_0.py for updated variable descriptions. Several of the parameters listed may either not implemented at this time, replaced, removed, or have been disabled. 

**Note:** 

All critical parameters for SLADS may be altered in a configuration file (Ex. ./CONFIG_0.py). Variable descriptions are provided inside of an example configuration provided and are grouped according to the following method:

    L0:     Tasks to be performed
    L1:     Task methods
        L1-0:   Pointwise
        L1-1:   Linewise
        L1-2:   Training Data Generation
    L2:     Model parameters
    L3:     Runtime/Output settings
    L4:     Non-operational - Do not change
    L5:     Debug/Deprecated - Will most likely be removed in future


Multiple configuration files in the form of CONFIG_*descriptor*.py, can be generated for which SLADS will be run sequentially. RESULTS_*descriptor* folders will correspond with the numbering of the CONFIG files, with the RESULTS folder without a description, containing results from the last run performed. 
    
###  **RUN**
After configuration, to run the program perform the following command in the root directory:

	$ python3 ./SLADS

###  **RESULTS**
All results will be placed in ./RESULTS/ (in the case of testing, at the conclusion of a sample's scan) as follows:

	TRAIN: Training results
		bestC.npy: Determined optimal c value determined in training
		Training Data Images: Training images with/without borders, summary sample images, c value curves 
        Model Training Images: Visualized training convergence images
        trainingSamples.p: Database of training sample inputs, suffix indicates intended acquisition method (point/line)
        trainingDatabase.p: Final training database for each c value, suffix indicates intended acquisition method (point/line)
		trainedModels.npy: Trained models corresponding to each possible c value

	TEST: Testing results
		Animations: Resultant visualizations and multimedia for test samples
				TEST_SAMPLE_1: Folder of frames for a sample's scan
				TEST_SAMPLE_2: Folder of frames for a sample's scan
				Videos: Videos of final scan progression
					TEST_SAMPLE_1.avi: Video of scan for sample
					TEST_SAMPLE_2.avi: Video of scan for sample
		dataPrintout.csv: Averaged final test results
		*.csv: Metric progressions averaged across testing samples
		mzResults: Final measurement results at specified mz ranges
			TEST_SAMPLE_1: Folder of final measurements at mz ranges
			TEST_SAMPLE_2: Folder of final measurements at mz Ranges
		testingAverageSSIM_Percentage.csv: Average SSIM progression results
		testingAverageSSIM_Percentage.png: Visualized average SSIM progression

In the case that multiple configuration files are provided in the form of: CONFIG_*descriptor*.py, the RESULTS folder will be duplicated with the same suffix for ease of testing. 

# OPERATIONAL PROCEDURE

**Warning:** This section’s procedure has not been confirmed as functional at this time. Below is a brief proposal of how SLADS may be easily integrated with physical scanning equipment

**Note:** In order to use a SLADS model in a physical implementation, the files resultant from the training procedure must be located within './RESULTS/TRAIN_RESULTS/', particularly:  bestC.npy and bestTheta.npy.

Prior to engaging the physical equipment run SLADS with the **impModel** variable enabled in the configuration file. All other testing and training flags within **Parameters: L0,** should be disabled. The program will then wait for a file: **LOCK** to be placed within the ./INPUT/IMP/ folder; which when it appears will trigger the program to read in any data saved into the same folder and produce a set of points to scan, saved in a file: **UNLOCK**. SLADS will delete the **LOCK** folder then, signalling the equipment that point selections have been made and in preparation for the next acquisition iteration. As with the training and testing datasets, it is expected that the data will be given to SLADS in .csv files for each of the specified mz ranges in accordance with the format mentioned in the **TRAINING/TESTING PROCEDURE** section. When SLADS has reached its termination criteria it will produce a different file: **DONE**, instead of: **UNLOCK**, to signal the equipment that scanning has concluded. 

# FAQ
###  **SLADS procdues an error: Could not connect to socket /tmp/ray/session_.../sockets/raylet**

Although the error would suggest there is something wrong with the network connectivity (can double check firewall settings that port 6375 is allowed to receive/send traffic), it is actually more likely to be an issue with Ray's ability to connect to its dependent services. At this time there isn't a fix available, though some success can be had simply continuing to re-run the script until it does manage to connect. If using Mac OS X, you might be able to mitigate the issue (albeit with additional text written onscreen) by installing ray at version 0.8.6.

    pip3 uninstall ray
    pip3 install ray==0.8.6
