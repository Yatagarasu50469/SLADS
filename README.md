
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


    NAME: 		lineSLADS
    VERSION NUM:	0.6.6
    DESCRIPTION:	Multichannel implementation of SLADS (Supervised Learning Algorithm 
			for Dynamic Sampling with additional constraint to select groups of 
			points along a single axis. 
    
    AUTHORS:	David Helminiak	EECE, Marquette University
    		Dong Hye Ye	EECE, Marquette University
    
    COLLAB. 	Julia Laskin	CHEM, Purdue University
    		Ruichuan Yin	CHEM, Purdue University
    		Hang Hu		CHEM, Purdue University
    
    FUNDING:	This project has received funding and was programmed for:
    		NIH Grant 1UG3HL145593-01
    
	GLOBAL
	CHANGELOG:	0.1     Multithreading adjustments to pointwise SLADS
		        0.1.1   Line constraints, concatenation, pruning, and results organization			
		        0.2     Comple program rewrite
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
	               ~0.7	Python 3.8 parallization
	               ~0.8	RAW file feature extraction
		       ~0.9	GPU acceleratiaon
	               ~1.0	Initial release

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

This implementation of SLADS has functioned on Windows, Mac, and Linux operating systems, with a clean Windows 10 installation described below. The package versions do not necessarily need to match with those listed, but should the program produce unexpected errors, installing a specific version of a package might be able to resolve the issue. 

	Operating System
		Win. 10:		Updated as of Jan 1 2020
		Ubuntu:			18.04
		Mac: 			10.13.6

	System
		Python			3.8.1
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
        pathlib:         1.0.1
		pickleshare: 		0.7.5
		pillow:			7.0.0
		prompt-toolkit: 	3.0.2
		psutil:			5.6.7
		pygments: 		2.5.2
		pyparsing: 		2.4.6
		pytz:			2019.3
		PyWavelets: 		1.1.1
		ray:			0.8.5
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

### **Installation on Mac/Linux**
**Note:** These instructions have not been tested on a clean system running only the operating systems specified, but should be expected to function.

	$ python --m pip install --upgrade pip
	$ pip3 install opencv-python datetime glob3 IPython joblib pandas psutil matplotlib pillow 
	   ray scipy sobol sobol_seq natsort multiprocess ray scikit-image sklearn tqdm
      $ cd ./CODE/scikit-learn
      $ make clean
      $ pip3 install --verbose --no-build-isolation --editable .

### **Installation on Windows 10**
**Note:** At this time, for Windows 10, the Linux subsystem must be used, as the multiprocessing package **ray** has not been (nor is likely to be in the near future) released for Windows. 

Open **PowerShell** as an Administrator and run the following:
	
	$ Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-
	   Linux

Restart when prompted and then open the **Microsoft Store**. Search for Ubuntu and install the appropriate version. After launching the Ubuntu program, wait for the installation to complete, then setup a username and password as desired. Run the following commands (interacting as directed): 

	$ sudo apt-get update
	$ sudo apt-get install python3-pip
	$ sudo apt-get install python3-opencv
	$ pip3 install datetime glob3 IPython joblib pandas psutil matplotlib pillow 
	   ray scipy sobol sobol_seq natsort multiprocess ray scikit-image sklearn tqdm

Place the SLADS-# release folder onto the Windows desktop. Note this must be done prior to building the custom sci-kit pacakge. Back inside of **Ubuntu**, enter the following to move into the SLADS folder replacing **username** and **versionNum** as appropriate:

	$ cd /mnt/c/Users/username/Desktop/SLADS-versionNum
    $ cd ./CODE/scikit-learn
    $ make clean
    $ pip3 install --verbose --no-build-isolation --editable .
    
The configuration: CONFIG_0.py may be edited through Windows with the editor of your choice, after which SLADS may be run using: 

	$ python3 SLADS.py

Multiple configuration files in the form of CONFIG_*descriptor*.py, can be generated for which SLADS will be run sequentially. RESULTS_*descriptor* folders will correspond with the numbering of the CONFIG files, with the RESULTS folder without a description, containing results from the last run performed. 


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
**Warning:** This section is no longer up to date, refer to ./CONFIG_0.py for updated variable descriptions. Several of the parameters listed may either not implemented at this time, replacd, removed, or have been disabled. 

All critical parameters for SLADS may be altered in ./CONFIG_0.py where:

	L0: Specifies the overall program function(s) to be performed
		trainingModel: Should a new SLADS model be generated
			- Uses data from ./INPUT/TRAIN/
		testingModel: Should a SLADS model be evaluated
			- Uses data from ./INPUT/TEST/
		LOOCV: Should Leave-One-Out Cross Validation be performed 
			- Disables testingModel by default, uses data from ./INPUT/TRAIN/
		impModel: Is a SLADS model being physically implemented
			- Uses data from ./INPUT/IMP/
	
	L1: Specifies general model training parameters
        preventResultsOverwrite: Should existing results folders not be allowed to be overwritten
        densityMeasures: Should the density measures be used
        regModel: Which regression model should be used: LS, or SLADS-Net NN
        algorithmNN: Which algorithm should be used for nearest neighbor
        scanMethod: Which scanning method shoud be used: pointwise or linewise
        partialLineFlag: If linewise, should partial segments of a line be scanned
        lineMethod: What method should be used for linewise point selection
        windowSize: Window size for approximate RD summation; 15 for 64x64, (width,height)
        stopPerc: What percentage of pixels should be measured before termination
        impSampleName: What name should be used for sample data obtained with impModel

	L2: Specifies variables that should not typically be changed
        measurementPercs: What sampling percentages should be used during training
        cValues: What possible c values should be tested for distortion calculations
        animationGen: Should animations be generated for testing/implementation
        lineRevistFlag: Should lines be allowed to be revisited
        percRAM: Percent free RAM to allocate pool; leave enough free for results overhead
        numFreeThreads: Number of processor threads to leave free
        multiSubFeatures: Should the original/normlized extracted features be used, or the recon values
        polyFit: Should a polynomial deg 2 fit be used, or a RBF sampler
        numMasks: How many masks should be used for each percentage during training
        consoleRunning: Running in a console/True, jupyter-notebook/False
    
    L3: Specifies variables that should not be changed
        mzWeighting: How should the mz visualizations be weighted: 'equal'
        LOOCV: Is LOOCV to be performed
    
    L4: Specifies variables that will most likely be removed, or radically altered in the future
        imageType: Type of Images: D - for discrete (binary) image; C - for continuous
        findStopThresh: Should a stopping threshold be found
    

		
###  **RUN**
After configuration, to run the program perform the following command in the root directory:

	$ python3 ./SLADS

###  **RESULTS**
All results will be placed in ./RESULTS/ (in the case of testing, at the conclusion of a sample's scan) as follows:

	TRAIN: Training results
		bestC.npy: Determined optimal c value determined in training
		bestTheta.npy: Best model corresponding with the determined best c value
		cValues.npy: List of possible c values that the best c value was chosen from
		Images: Empty directory for debug use; observation of training convergence
		trainedModels.npy: Trained models corresponding to each possible c value

	TEST: Testing results
		Animations: Resultant visualizations and multimedia for test samples
				TEST_SAMPLE_1: Folder of frames for a sample's scan
				TEST_SAMPLE_2: Folder of frames for a sample's scan
				Videos: Videos of final scan progression
					TEST_SAMPLE_1.avi: Video of scan for sample
					TEST_SAMPLE_2.avi: Video of scan for sample
		dataPrintout.csv: Averaged final test results
		mzResults: Final measurement results at specified mz ranges
			TEST_SAMPLE_1: Folder of final measurements at mz ranges
			TEST_SAMPLE_2: Folder of final measurements at mz Ranges
		testingAverageSSIM_Percentage.csv: Average SSIM progression results
		testingAverageSSIM_Percentage.png: Visualized average SSIM progression

In the case that multiple configuration files are provided in the form of: CONFIG_*descriptor*.py, the RESULTS folder will be duplicated with the same suffix for ease of testing. 

#====================================================================
# OPERATIONAL PROCEDURE

**Warning:** This section’s procedure has not been implemented at this time. Below is a brief proposal of how SLADS may be easily integrated with physical scanning equipment

**Note:** In order to use a SLADS model in a physical implementation, the files resultant from the training procedure must be located within './RESULTS/TRAIN_RESULTS/', particularly:  bestC.npy and bestTheta.npy.

Prior to engaging the physical equipment run SLADS with the **impModel** variable enabled in the configuration file. All other testing and training flags within **Parameters: L0,** should be disabled. The program will then wait for a file: **LOCK** to be placed within the ./INPUT/IMP/ folder; which when it appears will trigger the program to read in any data saved into the same folder and produce a set of points to scan, saved in a file: **UNLOCK**. SLADS will delete the **LOCK** folder then, signalling the equipment that point selections have been made and in preparation for the next acquisition iteration. As with the training and testing datasets, it is expected that the data will be given to SLADS in .csv files for each of the specified mz ranges in accordance with the format mentioned in the **TRAINING/TESTING PROCEDURE** section. When SLADS has reached its termination criteria it will produce a different file: **DONE**, instead of: **UNLOCK**, to signal the equipment that scanning has concluded. 

# FAQ
###  **SLADS procdues an error: Could not connect to socket /tmp/ray/session_2020-04-07_23-15-16_042185_13532/sockets/raylet**

Although the error would suggest there is something wrong with the network connectivity (can double check firewall settings), it is actually more likely to be an issue with available disk space. This can be handeled by manually specifying a temporary directory for the plasma memory storage on an alternate drive with more free space. Modify the **ray.init** command located in ./CODE/INTERNAL.py as shown below, replacing **/mntPoint/tmp** with a blank directory **tmp** located on another hard drive with more free space. 

        ray.init(num_cpus=num_threads, memory=amount_RAM, object_store_memory=int(amount_RAM*0.5), log_to_driver=False, logging_level=logging.ERROR, plasma_directory="/mntPoint/tmp")

