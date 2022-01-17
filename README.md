
     ▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄ 
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

# GENERAL INFORMATION


    NAME: 		SLADS
    VERSION NUM:	0.8.8
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
	CHANGELOG:	0.1.0   Multithreading adjustments to pointwise SLADS
                    0.1.1   Line constraints, concatenation, pruning, and results organization			
                    0.2.0   Line bounded constraints addition
                    0.3.0   Complete code rewrite, computational improvements
                    0.4.0   Class/function segmentation
                    0.5.0   Overhead reduction; switch multiprocessing package
                    0.6.0   Modifications for Nano-DESI microscope integration
                    0.6.1   Model robustness and reduction of memory overhead
                    0.6.2   Model loading and animation production patches
                    0.6.3   Start/End point selection with Canny
                    0.6.4   Custom knn metric, SSIM calc, init computations
                    0.6.5   Clean variables and resize to physical
                    0.6.6   SLADS-NET NN, PSNR, asymFinal, and multi-config
                    0.6.7   Clean asymmetric implementation with density features
                    0.6.8   Fixed RD generation, added metrics, and Windows compatible
                    0.7.0   CNN/Unet/RBDN with dynamic window size
                    0.7.1   c value selection performed before model training
                    0.7.2   Remove custom pkg. dependency, use NN resize, recon+measured input
                    0.7.3   Start/End line patch, SLADS(-Net) options, normalization optimization
                    0.7.4   CPU compatibility patch, removal of NaN values
                    0.7.5   c value selection performed before training database generation
                    0.6.9   Do not use -- Original SLADS(-Net) variations for comparison with 0.7.3
                    0.8.0   RAW MSI file integration (.raw, .d)
                    0.8.1   Model simplification, method cleanup, mz tolerance/standard patch
                    0.8.2   Multichannel, fixed groupwise, square pixels, accelerated RD, altered visuals/metrics
                    0.8.3   Mask seed fix, normalization for sim. fix, non-Ray option, pad instead of resize
                    0.8.4   Parallel c value selection fix, remove network resizing requirement, fix experimental
                    0.8.5   Model optimization, enable batch processing, SLADS training fix, database acceleration
                    0.8.6   Memory reduction, mz reconstruction vectorization, augmentation, global mz, mz window in ppm
                    0.8.7   Recon. script, Options for acq. rate, sequential names, live output, row offsets, and input scaling
                    0.8.8   Interpolation limits, static graph, parallel inferencing, ray deployment, test of FAISS
                    ~0.+.+  Custom adversarial network, Multimodal integration
                    ~1.0.0  Initial release

# PROGRAM FILE STRUCTURE
**Note:** If testing/training is not to be performed, then contents of 'TEST', 'TRAIN', may be disregarded, but a trained SLADS model must be present in: ./RESULTS/TRAIN/.

**Warning:** If training is enabled, any model already in ./RESULTS/TRAIN/ will be overwritten. Likewise, if testing or implementation is enabled, then any data in ./RESULTS/TEST/ and/or ./RESULT/IMP/, respectively will be overwritten.


    ------->ROOT_DIR
    	|------->README
    	|------->CONFIG_#-description.py
    	|------->SLADS.py
    	|------->mz.csv
    	|------->RECON
		|	|------->INPUT
		|	|	|------->TEST_SAMPLE_1
		|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->measuredMask.csv
    	|	|	|	|------->mz.csv
    	|	|	|	|------->physicalLineNums.csv
    	|	|------->RESULTS
    	|	|------->mz.csv
    	|	|------->mzReconstruction.ipynb
    	|------->CODE
    	|	|------->DEFS.py
    	|	|------->EXPERIMENTAL.py
    	|	|------->EXTERNAL.py
    	|	|------->INTERNAL.py
    	|	|------->SIMULATION.py
    	|	|------->TRAINING.py
    	|------->INPUT
    	|	|------->TEST
    	|	|	|------->TEST_SAMPLE_1
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->...
    	|	|	|------->TEST_SAMPLE_2
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->...
    	|	|------->TRAIN
    	|	|	|------->TRAIN_SAMPLE_1
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->...
    	|	|	|------->TEST_SAMPLE_2
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->...
    	|	|------->IMP
    	|	|	|------->sampleInfo.txt
    	|	|	|------->physicalLineNums.csv
    	|	|	|------->sampleName-line-0001.RAW
    	|	|	|------->sampleName-line-0002.RAW
    	|	|	|------->...
    	|	|	|-------> UNLOCK
    	|	|	|-------> LOCK
    	|	|	|-------> DONE
    	|------->RESULTS
    	|	|------->IMP
    	|	|	|-------IMP_SAMPLE_1
    	|	|	|	|------->Average
    	|	|	|	|------->mz
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|	|	|------->physicalLineNums.csv
    	|	|------->TEST
    	|	|	|-------dataPrintout.csv
    	|	|	|-------PSNR and SSIM Results (.csv and .png)
    	|	|	|-------TEST_SAMPLE_1
    	|	|	|	|------->Average
    	|	|	|	|------->mz
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|	|-------TEST_SAMPLE_2
    	|	|	|	|------->Average
    	|	|	|	|------->mz
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|-------VALIDATION
    	|	|	|-------dataPrintout.csv
    	|	|	|-------PSNR and SSIM Results (.csv and .png)
    	|	|	|-------VALIDATION_SAMPLE_1
    	|	|	|	|------->Average
    	|	|	|	|------->mz
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|	|-------VALIDATION_SAMPLE_2
    	|	|	|	|------->Average
    	|	|	|	|------->mz
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
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
This implementation of SLADS is generally only functional within Windows 10, given a reliance on vendor provided .dll's, as utilized by the multiplierz package. The package versions do not necessarily need to match with those listed. However, should the program produce unexpected errors, installing a specific version of a package might be able to resolve the issue. Note that the multiplierz pacakage, must be installed from the provided link under the installation commands.

	Operating System
		Win. 10:	Updated as of Jan 1 2021

	System
		Python		3.6.9
		pip		21.2.4

	Python Packages
		DateTime	4.3
		glob3		0.0.1
		ipython		8.0.0
		joblib		1.1.0
		matplotlib	3.5.1
		multiplierz	2.2.1
		natsort		8.0.2
		numba		0.55.0
		numpy		1.21.5
		opencv-python	4.5.5.62
		pandas		1.3.5
		Pillow		9.0.0
		psutil		5.9.0
		ray		1.9.2
		scikit-image	0.19.1
		scikit-learn	0.23.2                 
		scipy		1.7.3
		setuptools	47.3.1
		sklearn		0.0
		sobol		0.9
		sobol-seq	0.2.0
		tensorflow	2.7.0
		tqdm		4.62.3

### **Installation on Windows 10**

If GPU acceleration is to be used, a compatible CUDA Toolkit and cuDNN must be installed on the system. Installation instructions may be found through NVIDIA: https://docs.nvidia.com/deeplearning/cudnn/install-guide/

Install Visual Studio Community 2019, with the additional options: "Desktop development with C++" and "Python development":  https://visualstudio.microsoft.com/downloads/
Also install the "Build Tools for Visual Studio 2019", under "All downloads"/"Tools for Visual Studio 2019"

Install Python 3.8.0+: choosing to install with advanced options, selecting to install for all users, and add python to environment variables

Open the command prompt as an administrator (right-click on the command prompt icon and choose "Run as administrator")

For 64-bit Python, configure the build environment by entering the following lines into command prompt, where for 32-bit replace x64 by x86.
Note that the actual location of the specified file may vary depending on potential past Visual Studio installations. 

	$ SET DISTUTILS_USE_SDK=1
	$ "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

Navigate inside the command prompt to the SLADS base directory then enter the following commands:

	$ python3 -m pip install --upgrade pip
	$ pip3 install jupyter datetime glob3 IPython joblib pandas pathlib psutil matplotlib numba pillow ray ray[serve] scipy sobol sobol-seq natsort multiprocess scikit-image sklearn tensorflow tensorflow-addons tqdm numpy opencv-python pydot graphviz
	$ pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@master

	$ python
	$ from multiplierz.mzAPI.management import registerInterfaces
	$ registerInterfaces()

If the final printout indicates actions relating to the MSI file format intended for use, then follow through as neccessary. 

# TRAINING/TESTING PROCEDURE
###  **CONFIGURATION**

**Note:** 
All critical parameters for SLADS may be altered in a configuration file (Ex. ./CONFIG_0.py). Variable descriptions are provided inside of an example configuration provided and are grouped according to the following method:

    L0:     Tasks to be performed
    L1:     Task methods
        L1-0:   Implementation
        L1-1:   Pointwise
        L1-2:   Linewise
        L1-3:   Training Data Generation
    L2:     DLADS model parameters
    L3:     Runtime/Output settings
    L4:     Non-operational - Do not change
    L5:     Debug/Deprecated - Will most likely be removed in future

Multiple configuration files in the form of CONFIG_*descriptor*.py, can be generated for which SLADS will be run sequentially. RESULTS_*descriptor* folders will correspond with the numbering of the CONFIG files, with the RESULTS folder without a description, containing results from the last run performed. 

One file must be included in each of the sample folders (specifically within the subdirectories in TRAIN, TEST, and IMP, as the tasks to be performed may dictate). sampleInfo.txt should include the following ordered information:

	- Number of lines in the sample 
		Not to be confused with the number of line files present in the directory!
	- Width (mm) 
		Instrument field of view, not of just the sample dimensions
	- Height (mm)
		Instrument field of view, not of just the sample dimensions
		Must also account for any missing line files (if applicable) for training/testing!
	- Scan Rate (um/s)
		Output pixel positions and spatial resolution are based on this parameter in combination with the acquisition rate
	- Acquisition Rate (spectra/s)
		Output pixel positions and spatial resolution are based on this parameter in combination with the scan rate
	- Monoisotopic m/z (-1 indicates None, which will use TIC instead)
		Used for normalization of the m/z images
	- m/z tolerance (ppm)
		Only specify/include if m/z specification is set to 'value', or if using 'standard' normalization
	- m/z precision
		Number of decimal places used when considering spectrum locations/values
	- Sequential filenames (1 indicates sequential, 0 for physical row location)
		Indicates whether the filename line numbers going to be labeled sequentially rather than by physical row number

Each piece of information should be on its own line without additional description, as shown in the EXAMPLE sample directory. 

Another file mz.csv should be placed in the base directory, which contains line separated values of m/z locations, where the ranges used for visualization are determined through the specified m/z tolerance, to be extracted and used for a sample (as specified in sampleInfo.txt). If there are different mz that are specific to a sample, then a mz.csv should be placed in the sample directory, alongside the sampleInfo.txt file. Local/sample mz.csv files are used even when a global mz.csv is defined, though both need to have the same number of values/channels for model compatability. At this time, each m/z should be handpicked to highlight underlying structures of interest. If using the SLADS variants, or original RD generation method the m/z should be geometrically complementary. 

Each MSI data file (ex. .d, or .raw), must be labeled with the standard convention: 

	sampleName-line-000#.extension

If using Agilent equipment with linewise acquisition modes for an implementation run, then the line numbers are in sequence, rather than according to physical row number. In this case, enable the unorderedNames flag in the configuration file. 

###  **RUN**
After configuration, to run the program perform the following command in the root directory:

	$ python3 ./SLADS

###  **RESULTS**
All results will be placed in ./RESULTS/ (in the case of testing, at the conclusion of a sample's scan) as follows:

	TRAIN: Training results
		optimalC.npy: Determined optimal c value determined in training
		Training Data Images: Training images with/without borders, summary sample images, c value curves 
		Model Training Images: Visualized training convergence images
		trainingDatabase.p: Database of training samples; random 1% point masks from 1-40%
		validationDatabase.p: Database of validation samples; random 1% point masks from 1-40%
		testingDatabase.p: Database of testing sample data to allow for rapid back-to-back simulations
		trainingValidationSampleData.p: Database of training and validation sample data
		model_cValue_: Trained model corresponding to the indicated c value (.npy for SLADS-LS and SLADS-Net)

	TEST: Testing results
		TEST_SAMPLE_1: Folder of frames for a sample's scan
			Average: Individual and overall images of the progressive scanning for averaged m/z
			mz: Individual and overall images of the progressive scanning for each specified m/z
			Videos: Videos of final scan progression
			physicalLineNums.csv: Mapping from sequential filename numbering to physical row number
			measuredMask.csv: Final measurement mask; 1 for measured, 0 for unmeasured
		avgPSNR_Percentage.csv(.png): Progressive PSNR of the reconstruction for the averaged m/z
		dataPrintout.csv(.png): Summary of final results
		ERDPSNR_Percentage.csv(.png): Progressive PSNR of the ERD
		ERDSSIM_Percentage.csv(.png): Progressive SSIM of the ERD
		mzAvgPSNR_Percentage.csv(.png) Progressive averaged PSNR for reconstructions of all m/z
		mzAvgSSIM_Percentage.csv(.png) Progressive averaged SSIM for reconstructions of all m/z

	VALIDATION: Validation results
		Note: Identical structure to TEST

In the case that multiple configuration files are provided in the form of: CONFIG_*descriptor*.py, the RESULTS folder will be duplicated with the same suffix for ease of testing. Configuration file will be copied into the results directory at the termination of the program. 

# OPERATIONAL PROCEDURE

**Warning:** This section’s procedure has not been confirmed as functional at this time. Below is a brief proposal of how SLADS may be easily integrated with physical scanning equipment

**Note:** In order to use a SLADS model in a physical implementation, the files resultant from the training procedure must be located within './RESULTS/TRAIN_RESULTS/'.

Prior to engaging the physical equipment run SLADS with the **impModel** variable enabled in the configuration file. All other testing and training flags within **Parameters: L0,** should be disabled. The program will then wait for a file: **LOCK** to be placed within the ./INPUT/IMP/ folder; which when it appears will trigger the program to read in any data saved into the same folder and produce a set of points to scan, (row number, and position in um to start scanning for 1 second, based on specified scan rate for the sample) saved in a file: **UNLOCK**. SLADS will delete the **LOCK** folder then, signalling the equipment that point selections have been made and in preparation for the next acquisition iteration. As with the training and testing datasets, it is expected that the data will be given to SLADS in MSI files in accordance with the format mentioned in the **TRAINING/TESTING PROCEDURE** section. When SLADS has reached its termination criteria it will produce a different file: **DONE**, instead of: **UNLOCK**, to signal the equipment that scanning has concluded. A sampleInfo.txt must be included in the implementation directory as outlined in the CONFIGURATION section. 

# FAQ
###  **I read through the README thoroughly, but I'm still getting an error, or am confused about how a feature should work...**

Feel free to open an issue on the Github repository; though direct support cannot be guaranteed at this time. 

###  **Why am I receiving a 'list index out of range' error from the 'readScanData' method**

Most likely this is due to MSI line filenames not matching the convention outlined above.

###  **SLADS produces an error: Could not connect to socket /tmp/ray/session_.../sockets/raylet**

Although the error would suggest there is something wrong with the network connectivity (can double check firewall settings that port 6375 is allowed to receive/send traffic), it is actually more likely to be an issue with Ray's ability to connect to its dependent services. At this time there isn't a fix available, though some success can be had simply continuing to re-run the script until it does manage to connect. If using Mac OS X, you might be able to mitigate the issue (albeit with additional text written onscreen) by installing ray at version 0.8.6.

    pip3 uninstall ray
    pip3 install ray==0.8.6

###  **Why is SLADS not compatible with Linux distributions, or Mac operating systems**

As of v0.8.0, SLADS obtains information directly from MSI files, rather than pre-processed .csv m/z visualizations. These operations are reliant on vendor specific .dll files as provided in the multiplierz package. Supperficially it appears as though the multiplierz API might function within Linux. For example, the packages pythonnet and comtypes can be installed, but cannot actually function in a linux environment. An alternative approach, that may work, might be to attempt an installation through wineDocker. 

While it does not currently function for some MSI formats, (verified operational for XCalibur .RAW but not Agilent .d) multiplierz may be installed directly on Ubuntu 18.04 with the following commands:

	wget -q https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
	sudo dpkg -i packages-microsoft-prod.deb
	rm packages-microsoft-prod.deb
	sudo apt-get install apt-transport-https clang libglib2.0-dev nuget
	sudo apt-get update
  	sudo apt-get install -y aspnetcore-runtime-5.0
	sudo apt-get install -y dotnet-sdk-5.0
	sudo apt-get install -y dotnet-runtime-5.0
	sudo apt install gnupg ca-certificates
	sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
	echo "deb https://download.mono-project.com/repo/ubuntu stable-bionic main" | sudo tee /etc/apt/sources.list.d/mono-official-stable.list
	sudo apt update
	sudo apt install mono-devel
	pip3 install git+https://github.com/pythonnet/pythonnet.git@master
	pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@master

###  **The legacy single mz mode training images look incorrect**

Integration with nano-DESI MSI was never intended to perform evaluation of some mz reconstructions over the whole spectrum; it's simply too computationally expensive to consider. The single mz mode operates by setting the ground-truth average image as the ground-truth singuler mz visualization, since in SLADS operation, only the average image is used to determine E/RD. Evaluation is performed with reconstructed multiple m/z images against their ground-truth conterparts. The multiple ground-truth mz are averaged together into the correct ground-truth average mz image when simulated scanning is completed. Since this routine is not setup to be performed during training, or generation of the training/validation database, the saved images for the averaged reconstruction and averaged ground-truth are incorrectly labeled. The single mz mode is intended to be removed and should not be used except for very specific circumstances!
