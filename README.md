
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
    VERSION NUM:	0.8.0
    LICENSE:    	GNU General Public License v3.0
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
                    0.7.1   c value selection performed before model training
                    0.7.2   Remove custom pkg. dependency, use NN resize, recon+measured input
                    0.7.3   Start/End line patch, SLADS(-Net) options, normalization optimization
                    0.7.4   CPU compatibility patch, removal of NaN values
                    0.7.5   c value selection performed before training database generation
                    0.6.9   Do not use -- Original SLADS(-Net) variations for comparison with 0.7.3
                    0.8.0   Raw MSI file integration (Thermo .raw, Agilent .d), only Windows compatible
                    ~0.8.1  Multichannel integration
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
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->mz.csv
    	|	|	|	|------->sampleName-line1.RAW
    	|	|	|	|------->sampleName-line2.RAW
    	|	|	|	|------->...
    	|	|	|------->TEST_SAMPLE_2
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->mz.csv
    	|	|	|	|------->sampleName-line1.RAW
    	|	|	|	|------->sampleName-line2.RAW
    	|	|	|	|------->...
    	|	|------->TRAIN
    	|	|	|------->TRAIN_SAMPLE_1
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->mz.csv
    	|	|	|	|------->sampleName-line1.RAW
    	|	|	|	|------->sampleName-line2.RAW
    	|	|	|	|------->...
    	|	|	|------->TEST_SAMPLE_2
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->mz.csv
    	|	|	|	|------->sampleName-line1.RAW
    	|	|	|	|------->sampleName-line2.RAW
    	|	|	|	|------->...
    	|	|------->IMP
    	|	|	|------->sampleInfo.txt
    	|	|	|------->mz.csv
    	|	|	|------->sampleName-line1.RAW
    	|	|	|------->sampleName-line2.RAW
    	|	|	|------->...
    	|	|	|-------> UNLOCK
    	|	|	|-------> LOCK
    	|	|	|-------> DONE
    	|------->RESULTS
    	|	|------->TEST
    	|	|	|------->Animations
    	|	|	|	|------->TEST_SAMPLE_1
    	|	|	|	|------->TEST_SAMPLE_2
    	|	|	|	|------->...
    	|	|	|------->mzResults
    	|	|	|	|------->TEST_SAMPLE_1
    	|	|	|	|------->TEST_SAMPLE_2
    	|	|	|	|------->...
    	|	|	|------->dataPrintout.csv
    	|	|	|------->testingAverageSSIM_Percentage.csv
    	|	|	|------->testingAverageSSIM_Percentage.png
    	|	|------->TRAIN
    	|	|	|------->Model Training Images
    	|	|	|	|------->...
    	|	|	|------->Training Data Images
    	|	|	|	|------->...
    	|	|	|------->optimalC.npy
    	|	|	|------->trainingDatabase.npy
    	|	|	|------->trainingSamples.npy
    	|	|	|------->model_cValue_


# INSTALLATION
This implementation of SLADS is only functional within Windows 10, given a reliance on vendor provided .dll's, as utilized by the multiplierz package. The package versions do not necessarily need to match with those listed. However, should the program produce unexpected errors, installing a specific version of a package might be able to resolve the issue. Note that the multiplierz pacakage, must be installed from the provided link under the installation commands.

	Operating System
		Win. 10:		Updated as of Jan 1 2021

	System
		Python			3.8.5
		pip			20.2.4

	Python Packages
        opencv-python   4.4.0.46
        datetime        4.3
        glob3           0.0.1
        IPython         7.16.1
        joblib          0.17.0
        pandas          1.1.4
        psutil          5.7.3
        matplotlib      3.3.2
        pillow          8.0.1
        ray             1.0.0
        setuptools      50.3.0
        scipy           1.5.3
        sobol           0.9
        sobol-seq       0.2.0
        tensorflow      2.4.1
        natsort         7.0.1
        multiprocess    0.70.11.1
        scikit-image    0.17.2
        scikit-learn    0.23.2
        sklearn         0.0
        tqdm            4.51.0

### **Installation on Windows 10**

If GPU acceleration is to be used, a compatible CUDA Toolkit and cuDNN must be installed on the system. Installation instructions may be found through NVIDIA: https://docs.nvidia.com/deeplearning/cudnn/install-guide/

Install Visual Studio Community 2019, with the additional options: "Desktop development with C++" and "Python development":  https://visualstudio.microsoft.com/downloads/
Also install the "Build Tools for Visual Studio 2019", under "All downloads"/"Tools for Visual Studio 2019"

Install Python 3.8.3+: choosing to install with advanced options, selecting to install for all users, and add python to environment variables

Open the command prompt as an administrator (right-click on the command prompt icon and choose "Run as administrator")

For 64-bit Python, configure the build environment by entering the following lines into command prompt, where for 32-bit replace x64 by x86.
Note that the actual location of the specified file may vary depending on potential past Visual Studio installations. 

	$ SET DISTUTILS_USE_SDK=1
	$ "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

Navigate inside the command prompt to the SLADS base directory then enter the following commands:

	$ python3 -m pip install --upgrade pip
	$ pip3 install jupyter datetime glob3 IPython joblib pandas pathlib psutil matplotlib pillow ray scipy sobol sobol-seq natsort multiprocess scikit-image sklearn tqdm tensorflow numpy opencv-python pydot graphviz
	$ pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@master

# TRAINING/TESTING PROCEDURE
###  **CONFIGURATION**

**Note:** 
All critical parameters for SLADS may be altered in a configuration file (Ex. ./CONFIG_0.py). Variable descriptions are provided inside of an example configuration provided and are grouped according to the following method:

    L0:     Tasks to be performed
    L1:     Task methods
        L1-0:   Pointwise
        L1-1:   Linewise
        L1-2:   Training Data Generation
    L2:     DLADS model parameters
    L3:     Runtime/Output settings
    L4:     Non-operational - Do not change
    L5:     Debug/Deprecated - Will most likely be removed in future


Multiple configuration files in the form of CONFIG_*descriptor*.py, can be generated for which SLADS will be run sequentially. RESULTS_*descriptor* folders will correspond with the numbering of the CONFIG files, with the RESULTS folder without a description, containing results from the last run performed. 

Two files must be included in each of the sample folders (specifically within the subdirectories in TRAIN, TEST, and IMP, as the tasks to be performed may dictate). sampleInfo.txt should include the following ordered information:

	Number of lines in the sample (not to be confused with the number of files in the directory!)
	File extention for the MSI data (Thermo .raw, Agilent .d, etc.)
	Time resolution (Column spacing for row alignment; < 0.001 is not recommended given computational expense)
	Maximum line time (Maximum location (time) on any given line in the scan)
	m/z visualization method (Allowable values include: 'sum', or 'xic', though 'xic' is the recommended setting)
	m/z specification method (Allowable values include: 'range', or 'value', where 'value' will use the m/z tolerance to determine the final ranges)
	Normalization method (Allowable values include: 'tic', 'standard', 'none')
	m/z tolerance (ppm) (only needs to be specified if m/z specification is set to 'value', or if 'standard' is being used for the normalization method)

Each piece of information should be on its own line without additional description, as shown in the EXAMPLE sample directory. 

mz.csv can either contain line separated values of m/z locations or comma/line separated values of m/z ranges to be extracted and used for a sample (as specified in sampleInfo.txt). At this time, each m/z should be handpicked to highlight underlying biological structures, as they will be averaged together during SLADS operation to form the ERD. 

If using 'value', then the file should contain:

	central_mz_1
	central_mz_2
	...
Else if using 'range, then the file should contain:

	low_mz_1, high_mz_1
	low_mz_1, high_mz_1
	...


If the normalization method is specified as standard, then an additional file: mzStandards.csv should also be included. The file should contain line separated values of m/z locations (m/z tolerance will be used to determine the final ranges) to be used for normalization.

	mz_1
	mz_2
	...

###  **RUN**
After configuration, to run the program perform the following command in the root directory:

	$ python3 ./SLADS

###  **RESULTS**
All results will be placed in ./RESULTS/ (in the case of testing, at the conclusion of a sample's scan) as follows:

	TRAIN: Training results
		optimalC.npy: Determined optimal c value determined in training
		Training Data Images: Training images with/without borders, summary sample images, c value curves 
		Model Training Images: Visualized training convergence images
		trainingSamples.p: Database of training sample inputs, suffix indicates intended acquisition method (point/line)
		trainingDatabase.p: Final training database for each c value, suffix indicates intended acquisition method (point/line)
		model_cValue_: Trained model corresponding to the indicated c value (.npy for SLADS(-Net))

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

In the case that multiple configuration files are provided in the form of: CONFIG_*descriptor*.py, the RESULTS folder will be duplicated with the same suffix for ease of testing.  Configuration file will be copied into the results directory at the termination of the program. 

# OPERATIONAL PROCEDURE

**Warning:** This section’s procedure has not been confirmed as functional at this time. Below is a brief proposal of how SLADS may be easily integrated with physical scanning equipment

**Note:** In order to use a SLADS model in a physical implementation, the files resultant from the training procedure must be located within './RESULTS/TRAIN_RESULTS/', particularly:  bestC.npy and bestTheta.npy.

Prior to engaging the physical equipment run SLADS with the **impModel** variable enabled in the configuration file. All other testing and training flags within **Parameters: L0,** should be disabled. The program will then wait for a file: **LOCK** to be placed within the ./INPUT/IMP/ folder; which when it appears will trigger the program to read in any data saved into the same folder and produce a set of points to scan, (Can multiply by Time Resolution, as specified in the sample's sampleInfo.txt to find equivalent times to scan) saved in a file: **UNLOCK**. SLADS will delete the **LOCK** folder then, signalling the equipment that point selections have been made and in preparation for the next acquisition iteration. As with the training and testing datasets, it is expected that the data will be given to SLADS in .csv files for each of the specified mz ranges in accordance with the format mentioned in the **TRAINING/TESTING PROCEDURE** section. When SLADS has reached its termination criteria it will produce a different file: **DONE**, instead of: **UNLOCK**, to signal the equipment that scanning has concluded. A sampleInfo.txt and mz.csv must be included in the implementation directory as outlined in the CONFIGURATION section. 

# FAQ
###  **SLADS procdues an error: Could not connect to socket /tmp/ray/session_.../sockets/raylet**

Although the error would suggest there is something wrong with the network connectivity (can double check firewall settings that port 6375 is allowed to receive/send traffic), it is actually more likely to be an issue with Ray's ability to connect to its dependent services. At this time there isn't a fix available, though some success can be had simply continuing to re-run the script until it does manage to connect. If using Mac OS X, you might be able to mitigate the issue (albeit with additional text written onscreen) by installing ray at version 0.8.6.

    pip3 uninstall ray
    pip3 install ray==0.8.6

###  **Why is SLADS not compatible with Linux distributions, or Mac operating systems**

As of v0.8.0, SLADS obtains information directly from MSI raw files, rather than pre-processed .csv m/z visualizations. These operations are reliant on vendor specific .dll files as provided in the multiplierz package. Supperficially it appears as though the multiplierz API might function within Linux. For example, the packages pythonnet and comtypes can be installed, albeit with some difficulty, but cannot actually function in a linux environment. An alternative approach, that may work, might be to attempt an installation through wineDocker. 

While it does not currently function, multiplierz may be installed directly on Ubuntu 18.04 with the following commands:

	wget -q https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
	sudo dpkg -i packages-microsoft-prod.deb
	sudo apt-get install apt-transport-https clang libglib2.0-dev mono-dev nuget
	pip3 install git+https://github.com/pythonnet/pythonnet.git@master
	pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@master

