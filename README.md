
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
    VERSION NUM:	0.9.1
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
                    0.8.9   Simplification
                    0.9.0   Multichannel E/RD, distributed GPU/batch training, E/RD timing, fix seq. runs
                    0.9.1   Parallel sample loading, unique model names, post-processing mode, replace avg. mz with TIC
                    ~0.+.+  Custom adversarial network, Multimodal integration
                    ~1.0.0  Initial release

# PROGRAM FILE STRUCTURE
**Note:** If testing/training is not to be performed, then contents of 'TEST', 'TRAIN', may be disregarded, but a trained SLADS model must be present in: ./RESULTS/TRAIN/.

**Warning:** If training is enabled, any model already in ./RESULTS/TRAIN/ will be overwritten. Likewise, if testing or implementation is enabled, then any data in ./RESULTS/TEST/ and/or ./RESULT/IMP/, respectively will be overwritten.


    ------->ROOT_DIR
    	|------->README
    	|------->CONFIG_#-description.py
    	|------->SLADS.py
    	|------->runConfig.py
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
    	|	|	|	|------->mz.csv
    	|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->...
    	|	|	|------->TEST_SAMPLE_2
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->mz.csv
    	|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->...
    	|	|------->TRAIN
    	|	|	|------->TRAIN_SAMPLE_1
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->mz.csv
    	|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->...
    	|	|	|------->TRAIN_SAMPLE_2
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->mz.csv
    	|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->...
    	|	|------->POST
    	|	|	|------->POST_SIMULATION_SAMPLE_1
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->mz.csv
    	|	|	|	|------->measuredMask.csv
    	|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->...
    	|	|	|------->POST_EXPERIMENTAL_SAMPLE_2
    	|	|	|	|------->sampleInfo.txt
    	|	|	|	|------->mz.csv
    	|	|	|	|------->measuredMask.csv
    	|	|	|	|------->physicalLineNums.csv
    	|	|	|	|------->sampleName-line-0001.RAW
    	|	|	|	|------->sampleName-line-0002.RAW
    	|	|	|	|------->...
    	|	|------->IMP
    	|	|	|------->sampleInfo.txt
    	|	|	|------->mz.csv
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
    	|	|	|	|------->Progression
    	|	|	|	|------->mz
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|	|	|------->physicalLineNums.csv
    	|	|------->POST
    	|	|	|-------POST_SIMULATION_SAMPLE_1
    	|	|	|	|------->Progression
    	|	|	|	|------->mz
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|	|-------POST_EXPERIMENTAL_SAMPLE_2
    	|	|	|	|------->TIC
    	|	|	|	|------->mz
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|------->TEST
    	|	|	|-------dataPrintout.csv
    	|	|	|-------PSNR and SSIM Results (.csv and .png)
    	|	|	|-------TEST_SAMPLE_1
    	|	|	|	|------->Progression
    	|	|	|	|------->mz
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|	|-------TEST_SAMPLE_2
    	|	|	|	|------->Progression
    	|	|	|	|------->mz
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|-------VALIDATION
    	|	|	|-------dataPrintout.csv
    	|	|	|-------PSNR and SSIM Results (.csv and .png)
    	|	|	|-------VALIDATION_SAMPLE_1
    	|	|	|	|------->TIC
    	|	|	|	|------->Progression
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|	|-------VALIDATION_SAMPLE_2
    	|	|	|	|------->TIC
    	|	|	|	|------->Progression
    	|	|	|	|------->Videos
    	|	|	|	|------->measuredMask.csv
    	|	|------->TRAIN
    	|	|	|------->Model Training Images
    	|	|	|	|------->...
    	|	|	|------->Training Data Images
    	|	|	|	|------->...
		|	|	|	|------->cValueOptimization.csv
		|	|	|	|------->trainingValuation_RDTimes.csv
    	|	|	|------->optimalC.npy
    	|	|	|------->trainingDatabase.npy
    	|	|	|------->trainingSamples.npy
    	|	|	|------->model_cValue_

# INSTALLATION
This implementation of SLADS is generally only functional within Windows 10, given a reliance on vendor provided .dll's, as utilized by the multiplierz package. The package versions do not necessarily need to match with those listed. However, should the program produce unexpected errors, installing a specific version of a package might be able to resolve the issue. Note that the multiplierz pacakage, must be installed from the provided link under the installation commands.

	Operating System
		Win. 10:	Updated as of Jan 1 2021

	System
		Python		3.8.3
		pip		21.2.4

	Python Packages
		aiorwlock	1.3.0
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
		ray		1.11.0
		scikit-image	0.19.1
		scikit-learn	0.23.2                 
		scipy		1.7.3
		setuptools	47.3.1
		sklearn		0.0
		sobol		0.9
		sobol-seq	0.2.0
		tensorflow	2.8.0
		tqdm		4.62.3

### **Installation on Windows 10 and 11**

If GPU acceleration is to be used, a compatible CUDA Toolkit and cuDNN must be installed on the system. Installation instructions may be found through NVIDIA: https://docs.nvidia.com/deeplearning/cudnn/install-guide/

First enable ".Net Framework 3.5 (includes .NET 2.0 and 3.0)" under "Windows Features" (search for in the Windows Start Menu)

Then install 2019 Visual Studio Build Tools, checking boxes for "Desktop development with C++" and "Universal Windows Platform development": https://visualstudio.microsoft.com/vs/older-downloads/

Next install Visual Studio Community 2019, with the additional options: "Desktop development with C++" , "Universal Windows Platform development", and "Python development"

Any version more recent than 2019 will not be able to compile dependencies for reading MSI files with "multiplierz" package. 

Also install the "Build Tools for Visual Studio 2019", under "All downloads"/"Tools for Visual Studio 2019"

Install Python 3.8.0+: choosing to install with advanced options, selecting to install for all users, and add python to environment variables

Open the command prompt as an administrator (right-click on the command prompt icon and choose "Run as administrator")

For 64-bit Python, configure the build environment by entering the following lines into command prompt, where for 32-bit replace x64 by x86.
Note that the actual location of the specified file may vary depending on potential past Visual Studio installations. 

	$ SET DISTUTILS_USE_SDK=1
	$ "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

Open a new command prompt (critically, not as an administrator!) and enter the following commands:

	$ python -m pip install --upgrade pip
	$ pip3 install jupyter datetime glob3 IPython joblib pandas pathlib psutil matplotlib numpy numba pillow ray ray[serve] scipy sobol sobol-seq natsort multiprocess scikit-image sklearn tensorflow tensorflow-addons tqdm opencv-python pydot graphviz aiorwlock
	$ pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@master
	
Either switch back to, or open a new command prompt as an administrator and enter the following command:
	
	$ python -c "from multiplierz.mzAPI.management import registerInterfaces; registerInterfaces()"

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
	- m/z tolerance (ppm)
		Only specify/include if m/z specification is set to 'value', or if using 'standard' normalization
	- m/z precision
		Number of decimal places used when considering spectrum locations/values
	- Sequential filenames (1 indicates sequential, 0 for physical row location)
		Indicates whether the filename line numbers going to be labeled sequentially rather than by physical row number
	- Data Completeness (1 for fully acquired, 0 for partially acquired)
		Indicates whether the MSI files contain all of the sample data, or only some of it, as produced during implementation

Each piece of information should be on its own line without additional description, as shown in the EXAMPLE sample directory. 

Another file mz.csv should be placed in the base directory, which contains line separated values of m/z locations, where the ranges used for visualization are determined through the specified m/z tolerance, to be extracted and used for a sample (as specified in sampleInfo.txt). If there are different mz that are specific to a sample, then a mz.csv should be placed in the sample directory, alongside the sampleInfo.txt file. Local/sample mz.csv files are used even when a global mz.csv is defined, though both need to have the same number of values/channels for model compatability. At this time, each m/z should be handpicked to highlight underlying structures of interest. If using the SLADS variants, or original RD generation method the m/z should be geometrically complementary. 

Each MSI data file (ex. .d, or .raw), must be labeled with the standard convention: 

	sampleName-line-000#.extension

If using Agilent equipment with linewise acquisition modes for an implementation run, then the line numbers are in sequence, rather than according to physical row number. In this case, enable the unorderedNames flag in the configuration file. 

###  **RUN**
After configuration, to run the program perform the following command in the root directory:

	$ python ./SLADS

###  **RESULTS**
All results will be placed in ./RESULTS/ as follows:

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
		TEST_SAMPLE_1
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
		Identical structure to TEST
		
	POST: Post-Processing results
		POST_EXPERIMENTAL_SAMPLE_2
			Average: Individual and overall images of the progressive scanning for averaged m/z
			mz: Individual and overall images of the progressive scanning for each specified m/z
			physicalLineNums.csv: Mapping from sequential filename numbering to physical row number
			measuredMask.csv: Final measurement mask; 1 for measured, 0 for unmeasured

In the case that multiple configuration files are provided in the form of: CONFIG_*descriptor*.py, the RESULTS folder will be duplicated with the same suffix for ease of testing. Configuration file will be copied into the results directory at the termination of the program. 

# OPERATIONAL PROCEDURE

**Note:** In order to use a SLADS model in a physical implementation, the files resultant from the training procedure must be located within './RESULTS/TRAIN_RESULTS/'.

Prior to engaging the physical equipment run SLADS with the **impModel** variable enabled in the configuration file. All other testing and training flags within **Parameters: L0,** should be disabled. The program will then wait for a file: **LOCK** to be placed within the ./INPUT/IMP/ folder; which when it appears will trigger the program to read in any data saved into the same folder and produce a set of points to scan, (row number, and position in um to start scanning for 1 second, based on specified scan rate for the sample) saved in a file: **UNLOCK**. SLADS will delete the **LOCK** folder then, signalling the equipment that point selections have been made and in preparation for the next acquisition iteration. As with the training and testing datasets, it is expected that the data will be given to SLADS in MSI files in accordance with the format mentioned in the **TRAINING/TESTING PROCEDURE** section. When SLADS has reached its termination criteria it will produce a different file: **DONE**, instead of: **UNLOCK**, to signal the equipment that scanning has concluded. A sampleInfo.txt must be included in the implementation directory as outlined in the CONFIGURATION section. 

# FAQ
###  **I read through the README thoroughly, but I'm still getting an error, am confused about how a feature should work, or would like a feature/option added**

Feel free to check if it has already been addressed in, or open an issue on, the Github repository's issue tab. 

###  **Why am I receiving a 'list index out of range' error from the 'readScanData' method**

Most likely this is due to MSI line filenames not matching the convention outlined above.

###  **Why is SLADS not compatible with Linux distributions, or Mac operating systems**

As of v0.8.0, SLADS obtains information directly from MSI files, rather than pre-processed .csv m/z visualizations. These operations are reliant on vendor specific .dll files as provided in the multiplierz package. Supperficially it appears as though the multiplierz API might function within Linux. For example, the packages pythonnet and comtypes can be installed, but cannot actually function in a linux environment. An alternative approach, that may work, might be to attempt an installation through wineDocker, though this has not been attempted. 

While it does not currently function for some MSI formats, (verified operational for XCalibur .RAW but not Agilent .d) multiplierz may be installed directly on Ubuntu 18.04 or in a Docker container with the following commands:
	
	$ python -m pip install --upgrade pip
	$ sudo apt-get update
	$ sudo apt-get install -y wget git python3-opencv
	$ pip3 install jupyter datetime glob2 IPython joblib pandas pathlib2 psutil matplotlib numba pillow ray ray[serve] scipy sobol sobol-seq natsort multiprocess scikit-image sklearn tensorflow tensorflow-addons tqdm numpy opencv-python pydot graphviz aiorwlock
	$ wget -q https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
	$ sudo dpkg -i packages-microsoft-prod.deb
	$ rm packages-microsoft-prod.deb
	$ sudo apt-get install -y apt-transport-https clang libglib2.0-dev nuget
	$ sudo apt-get update
  	$ sudo apt-get install -y aspnetcore-runtime-5.0
	$ sudo apt-get install -y dotnet-sdk-5.0
	$ sudo apt-get install -y dotnet-runtime-5.0
	$ sudo apt-get install -y gnupg ca-certificates
	$ sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
	$ echo "deb https://download.mono-project.com/repo/ubuntu stable-bionic main" | sudo tee /etc/apt/sources.list.d/mono-official-stable.list
	$ sudo apt-get update
	$ sudo apt-get install -y mono-devel
	$ pip3 install git+https://github.com/pythonnet/pythonnet.git@master
	$ pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@master
	$ python -c "from multiplierz.mzAPI.management import registerInterfaces; registerInterfaces()"

The last line may produce a warning that module 'ctypes' has no attribute 'windll'; this should be safe to ignore for use with XCalibur .RAW files. 

General Docker container setup (assuming system already installed with Docker and the appropriate CUDA Toolkit)
	$ docker run -it --shm-size=2gb --runtime=nvidia --name DLADS tensorflow/tensorflow:latest-gpu
	
Additional useful flags for docker container setup
	'-v /mnt/Volume/:/workspace': Map mounted volume on host system to /workspace directory inside of the container
	'-p 8889:8888': Map port 8888 inside of the container to 8889 on the host network (in case of development with jupyter-notebook and multiple users)
	
###  **Training across multiple GPUs fails with NCCL Errors**

Presuming this error occurs on a Linux OS, increase the available shared memory /dev/shm/ to at least 512 MB. If using a Docker container this can be done by first shutting down Docker completely (sudo systemctl stop docker) editing the container's hostconfig.json file (edited with root privileges at /var/lib/docker/containers/containerID/hostconfig.json), changing the ShmSize to 536870912 and then starting docker back up (sudo systemctl start docker). The changed size may be verified with: df -h /dev/shm

###  **Program produces confusing outputs that look like warnings or errors**

Some common outputs that can safely be ignored are produced from Ray during model deployment or from multiprocessing pools. These cannot currently be suppressed/disabled. Some examples, as shown below, can be safely ignored; if in doubt, please feel free to check if this has already been addressed within, or open an issue on, the Github repository's issue tab. 

	$ INFO checkpoint_path.py:16 -- Using RayInternalKVStore for controller checkpoint and recovery.
	$ INFO http_state.py:98 -- Starting HTTP proxy with name 'SERVE_CONTROLLER_ACTOR:ExQkiR:SERVE_PROXY_ACTOR ...
	$ INFO api.py:475 -- Started Serve instance in namespace ...
	$ INFO:     Started server process ...
	$ INFO api.py:249 -- Updating deployment 'ModelServer'. component=serve deployment=ModelServer
	$ INFO deployment_state.py:920 -- Adding 1 replicas to deployment 'ModelServer'. component=serve deployment=ModelServer
	$ INFO api.py:261 -- Deployment 'ModelServer' is ready at ... component=serve deployment=ModelServer
	$ core_worker_process.cc:348: The global worker has already been shutdown. This happens when the language frontend accesses the Ray's worker after it is shutdown. The process will exit
	$ INFO deployment_state.py:940 -- Removing 1 replicas from deployment 'ModelServer'. component=serve deployment=ModelServer
	
###  **Cannot load Agilent .d files, when running on Windows**

Previous install guides had the multiplierz package installed with pip3 as an administrator, which causes issues when running the program as a non-administrator. As an administrator, uninstall multiplierz (pip3 uninstall multiplierz), then install as per the updated instructions. Note that following the installation, the registerInterfaces command still needs to be performed as an administrator.






