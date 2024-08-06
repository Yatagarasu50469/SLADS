<p align="center">
  <img src='/CODE/OTHER/HEADER.PNG' height='500'>
</p>

***
# PROGRAM

    NAME:           SLADS/DLADS/GLANDS
    MODIFIED:       6 August 2024
    VERSION:        0.10.1
    LICENSE:        GNU General Public License v3.0
    DESCRIPTION:    Dynamic sampling algorithms with updated/developing implementations of:
                       -SLADS       Supervised Learning Approach for Dynamic Sampling, using Least-Squares (LS) regression or a Multi-Layer Perceptron (MLP) neural Network (Net)
                       -DLADS       Deep Learning Approach for Dynamic Sampling, using a Convolutional Neural Network (CNN)
                       -GLANDS      Generative Learning Adversarial Network for Dynamic Sampling, using a Generative Adversarial Network (GAN)
    
    AUTHOR(S):      David Helminiak    EECE, Marquette University
    ADVISOR(S):     Dong Hye Ye        COSC, Georgia State University
    
    FUNDING:        Development of the SLADS and DLADS variations herein was originally funded by and developed for NIH Grant 1UG3HL145593-01
                    GLANDS development has no funding to presently delcare

***
# PROGRAM FILE STRUCTURE

**Note:** If testing/training is not to be performed, then contents of 'TEST', 'TRAIN', may be disregarded, but a trained model must be present in: ./RESULTS/TRAIN/.

**Warning:** If training is enabled, any model already in ./RESULTS/TRAIN/ will be overwritten. Likewise, if testing or implementation is enabled, then any data in ./RESULTS/TEST/ and/or ./RESULT/IMP/, respectively will be overwritten.

    ------->ROOT_DIR
	    |------->README.md
	    |------->CHANGELOG.md
        |------->START.py
        |------->CONFIG_#-description.py
        |------->channels.csv
        |------->CODE
        |    |------->AESTHETICS.py
        |    |------->COMPUTE.py
        |    |------->DEFINITIONS.py
        |    |------->EXPERIMENTAL.py
        |    |------->EXTERNAL.py
        |    |------->INTERNAL.py
        |    |------->MODEL_DLADS.py
        |    |------->MODEL_GLANDS.py
        |    |------->POSTPROCESS.py
        |    |------->REMOTE.py
        |    |------->RUN_CONFIG.py
        |    |------->SIMULATION.py
        |    |------->TRAINING.py
        |------->INPUT
        |    |------->IMAGES
        |    |    |------->TRAIN
        |    |    |    |------->sampleName.(png, tiff, jpg)
        |    |    |------->TEST
        |    |    |    |------->sampleName.(png, tiff, jpg)
        |    |------->TRAIN
        |    |    |------->SAMPLE_1
        |    |    |    |------->sampleInfo.txt
        |    |    |    |------->channels.csv
        |    |    |    |------->sampleName-line-0001.(d, RAW)
        |    |    |    |------->sampleName-line-0002.(d, RAW)
        |    |    |    |------->...
        |    |    |------->SAMPLE_2
        |    |    |    |------->sampleInfo.txt
        |    |    |    |------->channels.csv
        |    |    |    |------->mask.csv
        |    |    |    |------->sampleName.imzML
        |    |    |    |------->sampleName.ibd
        |    |    |------->SAMPLE_3
        |    |    |    |------->sampleInfo.txt
        |    |    |    |------->sampleName-chan-label.(png, tiff, jpg)
        |    |------->TEST
        |    |    |------->SAMPLE_4
        |    |    |    |------->sampleInfo.txt
        |    |    |    |------->channels.csv
        |    |    |    |------->sampleName-line-0001.(d, RAW)
        |    |    |    |------->sampleName-line-0002.(d, RAW)
        |    |    |    |------->...
        |    |    |------->SAMPLE_5
        |    |    |    |------->sampleInfo.txt
        |    |    |    |------->channels.csv
        |    |    |    |------->mask.csv
        |    |    |    |------->sampleName.imzML
        |    |    |    |------->sampleName.ibd
        |    |    |------->SAMPLE_6
        |    |    |    |------->sampleInfo.txt
        |    |    |    |------->sampleName-chan-label.(png, tiff, jpg)
        |    |------->POST
        |    |    |------->SIMULATION_SAMPLE_1
        |    |    |    |------->sampleInfo.txt
        |    |    |    |------->channels.csv
        |    |    |    |------->measuredMask.csv
        |    |    |    |------->sampleName-line-0001.RAW
        |    |    |    |------->sampleName-line-0002.RAW
        |    |    |    |------->...
        |    |    |------->EXPERIMENTAL_SAMPLE_1
        |    |    |    |------->sampleInfo.txt
        |    |    |    |------->channels.csv
        |    |    |    |------->measuredMask.csv
        |    |    |    |------->physicalLineNums.csv
        |    |    |    |------->sampleName-line-0001.RAW
        |    |    |    |------->sampleName-line-0002.RAW
        |    |    |    |------->...
        |    |------->IMP
        |    |    |------->sampleInfo.txt
        |    |    |------->channels.csv
        |    |    |------->physicalLineNums.csv
        |    |    |------->sampleName-line-0001.RAW
        |    |    |------->sampleName-line-0002.RAW
        |    |    |------->...
        |    |    |-------> UNLOCK
        |    |    |-------> LOCK
        |    |    |-------> DONE
        |------->OTHER
        |    |------->LOGO.PNG
        |    |------->SOCIAL.PNG
        |------->RESULTS
        |    |------->IMP
        |    |    |------->SAMPLE_1
        |    |    |    |------->Channels
        |    |    |    |    |------->...
        |    |    |    |------->Progression
        |    |    |    |    |------->...
        |    |    |    |------->Videos
        |    |    |    |    |------->...
        |    |    |    |------->measuredMask.csv
        |    |    |    |------->physicalLineNums.csv
        |    |------->POST
        |    |    |------->SIMULATION_SAMPLE_1
        |    |    |    |------->Channels
        |    |    |    |    |------->...
        |    |    |    |------->Progression
        |    |    |    |    |------->...
        |    |    |    |------->Videos
        |    |    |    |    |------->...
        |    |    |    |------->measuredMask.csv
        |    |    |------->EXPERIMENTAL_SAMPLE_2
        |    |    |    |------->Channels
        |    |    |    |    |------->...
        |    |    |    |------->Progression
        |    |    |    |    |------->...
        |    |    |    |------->Videos
        |    |    |    |    |------->...
        |    |    |    |------->measuredMask.csv
        |    |------->TEST
        |    |    |------->TEST_SAMPLE_1
        |    |    |    |------->Channels
        |    |    |    |    |------->...
        |    |    |    |------->Progression
        |    |    |    |    |------->...
        |    |    |    |------->Videos
        |    |    |    |    |------->...
        |    |    |    |------->measuredMask.csv
        |    |    |    |------->PSNR and SSIM Results (.csv and .tiff)
        |    |    |------->dataPrintout.csv
        |    |    |------->PSNR and SSIM Results (.csv and .tiff)
        |    |------->VALIDATION
        |    |    |------->VALIDATION_SAMPLE_1
        |    |    |    |------->Channels
        |    |    |    |    |------->...
        |    |    |    |------->Progression
        |    |    |    |    |------->...
        |    |    |    |------->Videos
        |    |    |    |    |------->...
        |    |    |    |------->measuredMask.csv
        |    |    |    |------->PSNR and SSIM Results (.csv and .tiff)
        |    |    |------->dataPrintout.csv
        |    |    |------->PSNR and SSIM Results (.csv and .tiff)
        |    |------->TRAIN
        |    |    |------->Model Training Images
        |    |    |    |------->...
        |    |    |------->model_modelType_channelType_windowType_windowDim_c_cValue
        |    |    |    |------->...
        |    |    |------->Training Data Images
        |    |    |    |------->...
        |    |    |------->Validation Data Images
        |    |    |    |------->...
        |    |    |------->cValueOptimization.csv
        |    |    |------->trainingHistory.csv
        |    |    |------->optimalC.npy
        |    |    |------->trainingDatabase.p
        |    |    |------->trainingValidationSampleData.p
        |    |    |------->trainingValidation_RDTimes.csv
        |    |    |------->validationDatabase.p
***
# INSTALLATION

**Note:** Throughout this document the '$ ' prefix is used to denote new lines and are not intended to be copied/executed!

Follow the instructions provided in the pre-installation guide specific to your system's operating system followed by those in the **Main Installation** section. The package versions do not necessarily need to match with those listed. However, should the program produce unexpected errors, installing a specific version of a package might be able to resolve the issue. Note that the multiplierz pacakage, must be installed from the provided repository fork, as specified in the installation guide for some methods to work properly; see **FAQ** for further information. 

**Software/Package Combinations**  
	
    Python             3.11.8
    pip                24.2
	
    aiorwlock          1.4.0
    antialiased-cnns   0.3
    colorama           0.4.6
    datetime           5.5
    fastapi            0.108.0
    glob2              0.7
    graphviz           0.20.3
    ipython            8.26.0
    joblib             1.4.2
    matplotlib         3.9.1
    multiplierz        2.2.2.dev1
    multiprocess       0.70.16
    multivolumefile    0.2.3
    natsort            8.4.0
    numba              0.60.0
    numpy              1.26.4
    opencv-python      4.10.0.84
    pandas             2.2.2
    pathlib            1.0.1
    pillow             10.4.0
    psutil             6.0.0
    py7zr              0.21.1
    pydot              3.0.1
    pyimzml            1.5.4
    pypiwin32          306
    ray                2.33.0
    scikit-image       0.22.0
    scikit-learn       1.5.1
    scipy              1.14.0
    sobol              0.9
    sobol-seq          0.2.0
    torch              2.2.2+cu121
    torchaudio         2.2.2+cu121
    torchvision        0.17.2+cu121
    tqdm               4.66.4
    typeguard          4.3.0
	
**Legacy DLADS-TF Model Variant Requirements**  

	Python             3.10.12
	tensorflow         2.15.1
    torch              2.1.1+cu121
    torchaudio         2.1.1+cu121
    torchvision        0.16.1+cu121
	
**Minimum Hardware Requirements:** As more functionality is continually being added, minimum hardware specifications cannot be exactly ascertained, however validation of functionality is performed on systems containing 64+ GB DDR3/4/5 RAM, 32+ CPU threads at 3.0+ GHz, 1080Ti/2080Ti+/4090 GPUs, and 1TB+ SSD storage. While v0.8.9 and below have managed to utilize pre-trained models with only a dual core CPU, 8 GB DDR2, and no discrete GPU, this is not an advisable set of hardware for utilizing this program. 

**GPU/CUDA Acceleration:** Highly recommended. Note that there shouldn't be a need to manually install the CUDA toolkit, or cudnn as pytorch (and TensorFlow) installation using pip should come include the neccessary files. 

**MSI Compatability:** Using alphatims and the custom fork of multiplierz, the following MSI file formats are functional: Agilent .D (Native Windows only), Thermo .RAW, Bruker .tdf/.tsf, and .imzML. The following MSI file formats supported by muliplierz have not been tested in this program: .wiff and .t2d 

## Windows (Native) Pre-Installation

If not already setup, install the latest Python v3.11 version (https://www.python.org/downloads/) selecting the options to install for all users, and addding python.exe to PATH. Operation using Python v3.12+ has not yet been validated.

Enable ".Net Framework 3.5 (includes .NET 2.0 and 3.0)" under "Windows Features" (search for in the Windows Start Menu)

Then install 2019 Visual Studio Build Tools, checking boxes for "Desktop development with C++" and "Universal Windows Platform development". 
Last known working links for downloading:   (https://download.visualstudio.microsoft.com/download/pr/93f24e82-778c-46ae-92f9-8d3010ecd011/ce6d976f23a41678262845b1ca6c441be204abf196ed6f03768734c2426242f5/vs_BuildTools.exe)  
(https://aka.ms/vs/16/release/vs_buildtools.exe)  

Next install Visual Studio Community 2019, with the additional options: "Desktop development with C++" , "Universal Windows Platform development", and "Python development"
Note: Any version more recent than 2019 will not be able to compile dependencies for reading MSI files with "multiplierz" package. 
Last known working links for downloading:  
(https://download.visualstudio.microsoft.com/download/pr/93f24e82-778c-46ae-92f9-8d3010ecd011/a5da04d78b1f94ab145a365733476df7a1ec6219fa17f09c7e2f3c7cd74d9c9e/vs_Community.exe)  
(https://aka.ms/vs/16/release/vs_community.exe)  

Install Git: (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

Install CMake, checking the option to add CMake to the system PATH for all users: (https://cmake.org/install/)

Open the command prompt as an administrator (right-click on the command prompt icon and choose "Run as administrator")

For 64-bit Python, configure the build environment by entering the following lines into command prompt, where for 32-bit replace x64 by x86.
The actual location of the specified file may vary depending on potential past Visual Studio installations. 

    $ SET DISTUTILS_USE_SDK=1
    $ "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
	
Open a command prompt (Not as an administrator!): 

	$ python -m pip install --upgrade pip
	$ pip3 install pywin32

## Docker Pre-Installation

This program can also be installed in a Docker container (confirmed functional on CentOS 7/8). 
OS-specific Docker installation instructions: (https://docs.docker.com/engine/install/) 
For GPU acceleration you will need to first install the NVIDIA container toolkit and configure Docker to use the correct runtime: (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit) 
After installing/entering the Docker container, follow the Ubuntu Pre-Installation instructions.  

The quick commands for initial container setup (shm-size should be set to 50% of available system RAM):
    $ docker run --gpus all -it --shm-size=64gb --runtime=nvidia --name GLANDS nvcr.io/nvidia/pytorch:24.05-py3
	$ rm -rf /usr/local/lib/python3.10/dist-packages/cv2
    $ python -m pip install --upgrade pip
    $ pip3 uninstall -y -r <(pip freeze)
	
Additional useful flags for a Docker container setup:

    '-v /mnt/Volume/:/workspace': Map mounted volume on host system to /workspace directory inside of the container; allows transmission of data between container and host
    '-p 8889:8888': Map port 8888 inside of the container to 8889 (change on a per-user basis) on the host network (in case of performing development with jupyter-notebook on systems with multiple users)

## Ubuntu 20.04+ and WSL2 (Windows Subsystem for Linux) Pre-Installation

Open a terminal window and perform the following operations: 
	
    $ sudo apt-get update -y 
	$ sudo apt-get upgrade -y 
    $ sudo apt-get install -y python3-pip build-essential software-properties-common ca-certificates gnupg libgl1-mesa-dev
	$ python3 -m pip install --upgrade pip
    $ sudo gpg --homedir /tmp --no-default-keyring --keyring /usr/share/keyrings/mono-official-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
    $ echo "deb [signed-by=/usr/share/keyrings/mono-official-archive-keyring.gpg] https://download.mono-project.com/repo/ubuntu stable-focal main" | sudo tee /etc/apt/sources.list.d/mono-official-stable.list
	$ sudo apt-get update -y
	$ sudo apt-get install -y wget git apt-transport-https clang libglib2.0-dev nuget aspnetcore-runtime-8.0 dotnet-sdk-8.0 mono-complete 
    $ pip3 install pythonnet

## Main Installation

Open a terminal or command prompt (**not as an administrator**) and run the commands shown below. If intending to use the legacy TensorFlow model variant ('DLADS-TF'), **do not run** the first line (that installs torch, torchvision, and torchaudio), but follow the relevant directions in the **FAQ** after completing the other instructions.
    
	$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    $ pip3 install opencv-python datetime glob2 IPython joblib pandas pathlib psutil matplotlib numpy numba pillow ray[serve]==2.33.0 scipy scikit-learn sobol sobol-seq natsort multiprocess scikit-image tqdm pydot graphviz aiorwlock pyimzml colorama typeguard py7zr multivolumefile alphatims
	$ pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@master

If on Windows, open command prompt **as an administrator** and replace python3 with python below; the final printout may indicate actions relating to the MSI file format intended for use; follow through as neccessary. If on Ubuntu, this command will produce a warning that module 'ctypes' has no attribute 'windll'; this should be safe to ignore for use with XCalibur .RAW files.
	
	$ python3 -c "from multiplierz.mzAPI.management import registerInterfaces; registerInterfaces()"

***
# TRAINING/TESTING PROCEDURE

## CONFIGURATION
**Note:** There are very few sanity checks inside of the code to ensure only correct/valid configurations are used. If an unexpected error occurs, please double check the sample and program configuration files to ensure they are correct before opening an issue or contacting the author. Thank you.

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

Multiple configuration files in the form of CONFIG_*descriptor*.py, can be generated and then automatically run sequentially. RESULTS_*descriptor* folders will correspond with the numbering of the CONFIG files, with the RESULTS folder without a description, containing results from the last run performed. 

For all samples, a 'sampleInfo.txt' file must be included in each of the sample directories (specifically within the subdirectories in TRAIN, TEST, and IMP, as the tasks to be performed may dictate). The actual content is dependent on the file type as listed below, where each piece of information should be on its own line without additional description, as shown in the EXAMPLE sample directory. 

For all samples, a 'mask.csv' file may optionally be included to limit measurements to with a defined area of the sample field of view; the dimensionality of the included data must match with the sample. Zeros indicate a location that should never be scanned, where ones indicate scannable locations. 

For MSI data, another file: 'channels.csv', should also be placed in the base directory, which contains line separated values of m/z locations to be targeted/visualized. If there are different m/z that are specific to a sample, then a 'channels.csv' should be placed in the sample directory. Local/sample 'channels.csv' files are used even when a global 'channels.csv' is defined. At this time, each m/z should be handpicked to highlight underlying structures of interest. 

For MALDI samples, aligned optical images may be incorporated into the model, however all samples must have an image of the same resolution included and training must have been conducted with the same optical flags enabled as intended to be used during simulation/implementation. The optical image should be placed in each sample's directory as 'optical.tiff'.

For DESI Bruker .tdf samples (.d files with timsTOF enabled) alphatims may be used instead of multiplierz, by specifying the sample type DESI-ALPHA. Alphatims is only compatible with DESI Bruker .tdf files, use of alphatims with Bruker .tsf, or other vendor MSI files is not supported!

### DESI MSI
    - Sample Type
        Specify the kind of file for correct parsing of the remaining fields (DESI, DESI-ALPHA, MALDI, IMAGE)
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
        Informs the m/z range used for each specified central m/z value (m/z range computed as: specified_m/z times (1-m/z tolerance) to specified_m/z times (1+m/z tolerance)
    - m/z lower bound
        Lower bound of the instrument acquired m/z spectra; not just the range of what m/z are to be targeted
    - m/z upper bound
        Upper bound of the instrument acquired m/z spectra; not just the range of what m/z are to be targeted
    - Sequential filenames (1 indicates sequential, 0 for physical row location)
        Indicates whether the filename line numbers going to be labeled sequentially rather than by physical row number

Each DESI MSI data file (ex. extensions: .d, or .raw), must be named with the standard convention below, increasing line-0001 as appropriate. If using Agilent equipment with linewise acquisition modes for an implementation run, then the line numbers are in sequence, rather than according to physical row number.

    sampleName-line-0001.extension

### MALDI MSI
    - Sample Type
        Specify the kind of file for correct parsing of the remaining fields (DESI, MALDI, IMAGE)
    - Columns (px)
        Number of columns in the sample
    - Rows (px)
        Number of rows in the sample
    - m/z tolerance (ppm)
        Informs the m/z range used for each specified central m/z value (m/z range computed as: specified_m/z times (1-m/z tolerance) to specified_m/z times (1+m/z tolerance)
    - m/z lower bound
        Lower bound of the instrument acquired m/z spectra; not just the range of what m/z are to be targeted
    - m/z upper bound
        Upper bound of the instrument acquired m/z spectra; not just the range of what m/z are to be targeted

Each MALDI MSI data file (ex. .ibd and .imzML), must be named with the standard convention: 

    sampleName.extension

### Images
    - Sample Type
        Specify the kind of file for correct parsing of the remaining fields (DESI, MALDI, IMAGE)
    - Columns (px)
        Number of columns in the sample
    - Rows (px)
        Number of rows in the sample
    - Channels
        Number of channels in the sample
    
Regular image files with extensions: .png, .jpg, and .tiff (extensions are case-sensitive) are supported as inputs. Each file only represents a single channel of information and is named with the standard convention below. sampleName should not itself contain '-chan-' which is used to parse out the channel label. Note that the label should ideally be descriptive of the channel (i.e. Red, Green, Blue, Alpha, 0, 1, 2, etc.). Targeting only specific channels out of those included as inputs is not currently supported.

    sampleName-chan-label.extension
	
If a large number of new .png, .jpg, and/or .tiff images (extensions are case-sensitive), with single or combined multiple channels, are to be added as training and/or testing inputs, then a helper method will automatically prepare them for use with the main program. Place such images into ./INPUT/IMAGES/TRAIN/ or ./INPUT/IMAGES/TEST/ as appropriate. The program will split multichannel images, generate sampleInfo.txt files, and move the results into the corresponding ./INPUT/TRAIN and ./INPUT/TEST directories. If a sampleName already exists in the destination location, it will be overwritten. The original files in ./INPUT/IMAGES will be deleted to prevent repeated processing. Each file should follow the convention below, where the desired destination is set by sampleName, and the number following '-numChan-' (starting with '1') indicates how many channels are in the actual image (required as on its own opencv will read/write grayscale images as 3 identical channels. sampleName should not itself contain '-chan-' or '-numChan-'. 4 dimensional images are not presently supported. 

    sampleName-numChan-#.extension

### RUN
After configuration, to run the program perform the following command in the root directory (if on Windows, replace python3 with python):

    $ python3 ./START.py

### RESULTS
All results will be placed in ./RESULTS/ as follows (presuming MSI data):

    TRAIN: Training results
        Model Training Images: Progressive model training convergence plots
        model_modelType_channelType_windowType_windowDim_c_cValue: Trained model corresponding to the indicated parameters (.npy for SLADS-LS and SLADS-Net)
        Training Data Images: Training images with/without borders, summary sample images, and c value curves 
        Validation Data Images: Validation images with/without borders and summary sample images
        trainingHistory.csv: Progresive model training losses
        optimalC.npy: Determined optimal c value determined in training
        trainingDatabase.p: Database of training samples
        trainingValidationSampleData.p: Database of training and validation sample data
        trainingValidation_RDTimes.csv: Summary of RD computation times for training and validation sample data
        validationDatabase.p: Database of validation samples
        
    TEST: Testing results
        TEST_SAMPLE_1
            Progression: Individual and overall images of the progressive scanning for averaged m/z
            Channels: Individual and overall images of the progressive scanning for each specified m/z
            Videos: Videos of final scan progression
            physicalLineNums.csv: Mapping from sequential filename numbering to physical row number
            measuredMask.csv: Final measurement mask; 1 for measured, 0 for unmeasured
            dataPrintout.csv: Summary of final results
            (PSNR/SSIM)_allAvg.(csv/png): Progressive averaged PSNR/SSIM for all reconstructions
            (PSNR/SSIM)_chanAvg.(csv/png): Progressive averaged PSNR/SSIM for targeted channel reconstructions
            (PSNR/SSIM)_ERD.(csv/png): Progressive averaged PSNR/SSIM for the ERD 
            (PSNR/SSIM)_sumImage.(csv/png): Progressive averaged PSNR/SSIM for the ERD 
        dataPrintout.csv: Summary of final results, across all testing samples
        (PSNR/SSIM)_allAvg.(csv/png): Progressive averaged PSNR/SSIM for all reconstructions, across all testing samples
        (PSNR/SSIM)_chanAvg.(csv/png): Progressive averaged PSNR/SSIM for targeted channel reconstructions, across all testing samples
        (PSNR/SSIM)_ERD.(csv/png): Progressive averaged PSNR/SSIM for the ERD, across all testing samples
        (PSNR/SSIM)_sumImage.(csv/png): Progressive averaged PSNR/SSIM for the sum of all channel reconstructions, across all testing samples

    VALIDATION: Validation results
        Identical structure to TEST
        
    POST: Post-Processing results
        POST_EXPERIMENTAL_SAMPLE_2
            Progression: Individual and overall images of the progressive scanning for averaged channel
            Channels: Individual and overall images of the progressive scanning for each specified channel
            physicalLineNums.csv: Mapping from sequential filename numbering to physical row number
            measuredMask.csv: Final measurement mask; 1 for measured, 0 for unmeasured

In the case that multiple configuration files are provided in the form of: CONFIG_*descriptor*.py, the RESULTS folder will be duplicated with the same suffix for ease of testing. Configuration file will be copied into the results directory at the termination of the program. 

***
# EXPERIMENTAL IMPLEMENTATION

**Note:** Currently only supported/validated for DESI MSI equipment; MALDI and general image formats are under development.

**Note:** In order to use a trained model in a physical implementation, the files resultant from the training procedure must be located within './RESULTS/TRAIN/'.

1. Prior to engaging the physical equipment, run the program with the **impModel** variable enabled in the configuration file. A custom implementation input directory, to use instead of the default: './INPUT/IMP/', may be specified with the **impInputDir** variable, under section L1-1, during configuration. All other testing and training flags within in the configuration file, under section L0, should be disabled. 

2. Initialize the program with 'python3 START.py' (if on Windows, replace python3 with python) and place 'sampleInfo.txt' and 'channels.csv' (if not using a global 'channels.csv' file in the SLADS/DLADS root directory) into the implementation input directory. The 'sampleInfo.txt' and 'channels.csv' files should follow the format outlined in the CONFIGURATION section of the README documentation. 

3. Place a blank file named **LOCK** into the implementation input directory to signal the program that the 'sampleInfo.txt' and 'channels.csv' files have been placed. 

2. Creation of the **LOCK** file triggers the program to read in any data saved in the implementation input directory and produce a set of points (row number, and column positions in um) to physically scan. This data is saved in a file: **UNLOCK**. 

3. **LOCK** will be automatically deleted to signal the equipment that **UNLOCK** now contains new measurement positions.

4. The program waits for **LOCK** to re-appear, which should be created after the requested data has been scanned into the implementation input directory. The data must follow the formatting specified in the TRAINING/TESTING PROCEDURE section of the README documentation. 

5. When the termination criteria have been met, the program produces a file: **DONE**, instead of: **UNLOCK**, to signal the equipment that scanning has concluded. 

***
# FAQ

### I read through the README thoroughly, but I'm still getting an error, am confused about how a feature should work, or would like a feature/option added
Please check if it has already been addressed, or open an issue on the Github repository at https://github.com/Yatagarasu50469/SLADS/issues 

At this time there is no legacy version support; it would be advisable to verify that the latest release is being used and installed packages match with those listed in the corresponding README. 

### Most of the code appears to focus on integration with MSI modalities, can it be used for others?
Absolutely! Currently support is offered for regular images with single and multiple channels. If compatability with another file format is needed for your use case, please either open an issue in the repository, or feel free to fork this project and open a pull request with the added functionality. 

### Can this program be run using MacOS?
This is probably still possible to some extent (was confirmed to be functional at one point at/below v0.9.6 on a Hackintosh with an NVIDIA 1080Ti), but verifiable, up-to-date installation instructions are not currently available. 

### Why are Agilent .D files only supported on Windows?
Multiplierz relies on vendor-provided, proprietary .dll files to read MSI data and are therefore subject to their limitations. Substantial efforts have been made to port those files into a Linux-compatible format, though without any success. 

### Why is manual installation of the mutliplierz package needed?
Compatibility support for Bruker .tsf MSI files is not currently available in the main package (Reference: https://github.com/BlaisProteomics/multiplierz/issues/10). Additionally, a required overhead bypass flag has not yet been validated (Reference: https://github.com/BlaisProteomics/multiplierz/issues/9)

### Why has the framework been switched to PyTorch?
The short answer is that TensorFlow, as of v2.10.1, has dropped support for GPU acceleration on Windows and Agilent .D files can only be read/used on native Windows. An older model variant (referenced as 'DLADS-TF' during configuration) has been included (principally for benchmarking and historical reference) and should be functional if a TensorFlow installation (lower than v2.16) is available/desirable. Theoretically, installation of TensorFlow v2.10.1 would enable the model to run with CUDA acceleration on native Windows, though issues with keras augmentation layers in that release would be expected to significantly hamper model training performance. This legacy model variant has been successfully installed and run in an Ubuntu 22.04 guest, on a Windows 11 host, through WSL2 (Windows Subsystem for Linux), using the directions below. 

**Warning:** If the packages are installed out of order, or if previous packages, particularly dependencies/requirements/installationss of TensorFlow and PyTorch, were not fully/completely purged, then this process will likely fail to provide a working instance. 

After following the instructions within the appropriate **Pre-Installation** and **Main Installation** sections (being sure **not** to install torch, torchvision, and torchaudio there) enter the instructions below into a terminal to enable NVIDIA GPU acceleration (if applicable/intended). TensorFlow v2.15.1 has been used, as 1) it preceeds the release/alterations made with/for Keras v3.0 (TensorFlow v2.16+), which breaks direct compatibility with the prior DLADS architecture code; and 2) its available binaries use similar third-party package versioning to PyTorch v2.1.1. This is likely to be the last compatible versioning to host both TensorFlow and PyTorch in the same Python environment, while using the same major release of CUDA (v12.2 and v12.2 respectively). More recent TensorFlow releases would be expected break ability to run PyTorch models and vice-versa. A Keras v3.0+ implementation, with swappable TensorFlow/PyTorch/Jax backends may be considered in future to potentially resolve this, but it is more likely that TensorFlow support/improvements will simply be dropped completely. 

    #Confirm any expected GPUs are available
    $ nvidia-smi
    
	#Install TensorFlow
	$ pip3 install tensorflow[and-cuda]==2.15.1
	
	#Add paths to ~/.bashrc (Reference: https://github.com/tensorflow/tensorflow/issues/65035)
	echo 'export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> ~/.bashrc
	echo 'CUDNN_DIR=$(dirname $(dirname $(python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)")))' >> ~/.bashrc
	echo 'export LD_LIBRARY_PATH=$(find ${CUDNN_DIR}/*/lib/ -type d -printf "%p:")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
	echo 'PTXAS_DIR=$(dirname $(dirname $(python3 -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)")))' >> ~/.bashrc
	echo 'export PATH=$(find ${PTXAS_DIR}/*/bin/ -type d -printf "%p:")${PATH:+:${PATH}}' >> ~/.bashrc
	
	#Reload bashrc
    $ source ~/.bashrc
	
	#Verify GPUs are available, disregardinng complaints regarding NUMA support. 
    $ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
	
	#Follow the instructions from the Main Installation section, though ignoring the whole line installing torch, then return here
	$ pip3 install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

###Running in WSL2 (Windows Subsystem for Linux), Ray crashes due to insufficient memory
WSL2 is still a virutal machine and does not see/allocate all of available system memory (the current default is half). This may be increased manually and might be sufficient to allow the program to run. If memory limit is still exceeded, try decreasing the number of availableThreads in the configuration being run. Open a command prompt and enter the instruction below, choosing to create a new file if prompted. 

	$ cd %UserProfile%
	$ notepad.exe .wslconfig
	
	#Paste the following into the new notepad window, make changes as appropriate for your system, then save without an extension and exit
	[wsl2]
	memory=64GB
	processors=8

	#Back in command prompt, use the next instruction to remove the .txt extension from .wslconfig
	$ ren .wslconfig.txt .wslconfig

	#Open PowerShell as an administrator and run the following to reboot WSL2
	$ wsl --shutdown
	$ restart-service LxssManager

## Program Operation

### Why aren't all of the system GPUs used during training?
Distributed GPU training is not currently supported for DLADS and has been removed from DLADS-TF given a proclivity to yield inconsistent results. Multiple GPUs can be and are leveraged in simulated testing when more than a single sample is being evaluated. 

### Cannot load a known-supported MSI format or receiving a message ImportError: cannot import name 'Iterator' from 'collections'
Most likely this is due to use of an older multiplierz installation. You will need to uninstall the existing multiplierz installation and re-download it from the forked repository, where a patches have since been applied. 

    $ pip3 uninstall multiplierz
    $ pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@master

### Training across multiple GPUs fails with NCCL Errors
Presuming this error occurs on a Linux OS, increase the available shared memory /dev/shm/ to at least 512 MB. If using a Docker container this can be done by first shutting down Docker completely (sudo systemctl stop docker) editing the container's hostconfig.json file (edited with root privileges at /var/lib/docker/containers/containerID/hostconfig.json), changing the ShmSize to 536870912 and then starting docker back up (sudo systemctl start docker). The changed size may be verified with: df -h /dev/shm

### Why am I receiving a 'list index out of range' error from the 'readScanData' method
Most likely this is due to filenames not matching the outlined naming convention.

### Program produces confusing outputs that look like warnings or errors
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
    
### Program seems very slow
The most likely cause is from large sample dimensionality (i.e. both rows and columns are in excess of 100). It the computational platform appears to lock up, it is recommended to check either Task Manager (Windows) or htop (Ubuntu: sudo apt-get install htop) and verify there is both sufficient RAM and computational capabilities to utilize the selected model. DLADS is much more computationally intensive than SLADS-LS, but will try to offload work onto a discrete CUDA GPU if one is available. Warnings/Errors will be output if RAM limits are exceeded on either system or GPU. 

Verify that liveOutputFlag in the configuration file is set to False, visualization and metric extraction proceudres are very computationally expensive as reconstructions are generated across the whole spectrum and visualizations are written onto a physical drive, rather than held in RAM.

Verify that parallelism is enabled, as the program can perform some operations concurrently. 

### The program appears to regularly freeze and I can't interact with the computer
Certain sections of the program are extremely compute/memory/storage instensive and are expected to freeze the graphical output of even upper-end hardware; to verify check Task Manager on Windows, htop on Linux-based platforms, and/or nvidia-smi output for utlization levels. 

### Error indicates that the GPU should be disabled, settting parallelism to False doesn't resolve this
The parallelism flag in the configuration file does not control GPU usage, only CPU usage. In order to disable GPUs entirely, set availableGPUs = '-1'.

### Generating training database with the erdModel set to GLANDS, neither SLADS nor DLADS models, can be trained
Database generations with GLANDS is not backwards compatible with SLADS/DLADS operations, since SLADS/DLADS rely on a static pre-generated database of ground-truth RD maps and sparse-sampled reconstructions formed with IDW interpolation. GLANDS does not utilize/train with a ground-truth RD target, instead the architecture dynamically generates random sparse data during the training process. 

## Results

### Looking at the model training images for DLADS shows a lack of loss convergence
Try disabling data augmentation, setting augTrainData to False, if this fixes the issue than likely the default parameters (specified in MODEL_DLADS.py under the DataPreprocessing_DLADS class), particularly if a random crop is being used, this may be too large/small to extract meaningful features. If this doesn't fix the issue, than try lowering the learning rate and/or increasing the number of start filters. Beyond those parameters, provided there are sufficient training samples, a change in the network architecture may be required.

### Looking at the model training images for DLADS shows significant overfit
Double check that data augmentation is enabled, otherwise decrease the number of start filters used for the model. If this still does not fix the issue, it may be desirable to use a less complex model than DLADS, such as the included SLADS-LS or SLADS-Net options.

### The resulting reconstructions for SLADS/DLADS are quite blurry
If the provided data is fairly homogeneous, with predominant value fluctuations around structural edges, try decreasing the number of neighbors used in IDW reconstruction (numNeighbors) to as low as 1.

### Encountering OOM errors during new code development
Ray/Python pin objects in memory if any reference to them still exists; references (particularly to large objects) must be prevented or deleted. Admittedly, there's probably a better way of handling this, but the current coding practices for reducing memory overhead and OOM errors are as follows:
1. Delete Ray references when they are no longer needed, then calling gc.collect()
2. Reset Ray (resetRay(numberCPUS), which also has been set to call gc.collect() after major remote computations and results have been retrieved
3. On returns from remote calls, copy the data to prevent reference storage
   -if _ = ray.get() (i.e. returning a None object) -> No problem, a reference was not created
   -if ray.get(), returns list/array -> use ray.get().copy()
   -if ray.get(), returns something else -> use copy.deepcopy(ray.get())
4. Delete large objects when they are no longer needed, then calling gc.collect()
5. Call gc.collect() after major methods return to MAIN.py

***
# PUBLICATIONS

### RESEARCH PRODUCED WITH THIS CODE

**Simulatated Acquisition of MALDI MSI with DLADS**  
**Version(s):** v0.9.5  
**Subject:** MALDI MSI  
**Citation(s):** D. Helminiak, T. Boskamp, and D. H. Ye, “Multimodal Deep Learning Approach for Dynamic Sampling with Automatic Feature Selection in Matrix-Assisted Laser Desorption/Ionization Mass Spectrometry Imaging,” Electronic Imaging, vol. 36, no. 15. Society for Imaging Science & Technology, pp. 143-1-143–6, Jan. 21, 2024. doi: 10.2352/ei.2024.36.15.coimg-143.   
**Available:** (https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/ei/36/15/COIMG-143)

**Updated Simulatated Acquisition of DESI MSI with DLADS**  
**Version(s):** v0.9.1  
**Subject:** DESI MSI  
**Citation(s):** D. Helminiak, H. Hu, J. Laskin, D. Ye. “Deep Learning Approach for Dynamic Sparse Sampling for Multi-Channel Mass Spectrometry Imaging“, IEEE Transactions on Computational Imaging, 9, 250-259 (2023). DOI:10.1109/TCI.2023.3248947  
**Available:** (https://ieeexplore.ieee.org/document/10052699), (https://arxiv.org/abs/2210.13415)   

**Experimental DESI MSI Integration with DLADS**  
**Version(s):** Experimental results were determined with v0.8.9, and simulation results with v0.9.1  
**Subject:** DESI MSI  
**Citation(s):** H. Hu, D. Helminiak, M. Yang, D. Unsihuay, R.T. Hilger, D. H. Ye, and J. Laskin, "High-throughput mass spectrometry imaging with dynamic sparse sampling," ACS Measurement Science Au, Aug. 2022.   
**Available:** (https://pubs.acs.org/doi/10.1021/acsmeasuresciau.2c00031)

**Development for Simulatated Acquisition of DESI MSI with DLADS - Master's Thesis**  
**Version(s):** v0.8.6  
**Subject:** DESI MSI  
**Citation(s):** D. S. Helminiak, “Deep learning approach for dynamic sampling for high-throughput nano-desi msi,” Master's Thesis, Marquette University, 2021, copyright - Database copyright ProQuest LLC; ProQuest does not claim copyright in the individual underlying works; Last updated - 2022-02-21.  
**Available:** (https://epublications.marquette.edu/theses_open/710/)  
**Presentation:** (https://www.youtube.com/watch?v=WY5Ae19pmiE)

**Initial Simulatated Acquisition of with DLADS**  
**Version(s):** v0.6.9 and v0.7.3  
**Subject:**  DESI MSI  
**Citation(s):** D. Helminiak, H. Hu, J. Laskin, and D. H. Ye, “Deep learning approach for dynamic sparse sampling for high-throughput mass spectrometry imaging,” Electronic Imaging, vol. 2021, no. 15, pp. 290–1–290–7, Jan. 2021.  
**Available:** (https://doi.org/10.2352/issn.2470-1173.2021.15.coimg-290)

### SIMILAR & PRECEEDING RESEARCH

**Original SLADS-Net**  
**Version(s):** (https://github.com/saugatkandel/fast_smart_scanning)  
**Subject:** Scanning Electron Microscopy and Fast Autonomous Scanning Toolkit (FAST)
**Citation(s):** Kandel, S., Zhou, T., Babu, A.V. et al. Demonstration of an AI-driven workflow for autonomous high-resolution scanning microscopy. Nat Commun 14, 5501 (2023). https://doi.org/10.1038/s41467-023-40339-1
**Available:** (https://www.nature.com/articles/s41467-023-40339-1), (https://arxiv.org/pdf/2301.05286)

**Original SLADS-Net**  
**Version(s):** (https://github.com/cphatak/SLADS-Net), (https://github.com/anl-msd/SLADS-Net)  
**Subject:**  Scanning Electron Microscopy  
**Citation(s):** Y. Zhang, G. Godaliyadda, N. Ferrier, E. Gulsoy, C. Bouman, and C. Phatak, “Slads-net: Supervised learning approach for dynamic sampling using deep neural networks,” 2018.  
**Available:** (https://doi.org/10.2352/ISSN.2470-1173.2018.15.COIMG-131)

**Unsupervised-SLADS (U-SLADS)**  
**Subject:**  Metal Dendrite Sampling  
**Citation(s):** Y. Zhang, X. Huang, N. Ferrier, E. Gulsoy, and C. Phatak, “U-slads: Unsupervised learning approach for dynamic dendrite sampling,” 2018  
**Available:** (https://arxiv.org/abs/1807.02233)

**Original SLADS-LS**  
**Subject:**  Energy Dispersive X-Ray Spectroscopy  
**Citation(s):** Y. Zhang, G. Godaliyadda, N. Ferrier, E. Gulsoy, C. Bouman, and C. Phatak, “Reduced electron exposure for energy-dispersive spectroscopy using dynamic sampling,” Ultramicroscopy, vol. 184, pp. 90 – 97, 2018.  
**Available:** (https://sciencedirect.com/science/article/pii/S0304399117303157)

**Original SLADS-LS**  
**Subject:** Electron Back Scatter Diffraction  
**Citation(s):** G. Godaliyadda, D. Ye, M. Uchic, M. Groeber, G. Buzzard, and C. Bouman, “A framework for dynamic image sampling based on supervised learning,” IEEE Transactions on Computational Imaging, vol. 4, no. 1, pp. 1–16, mar 2018.  
**Available:** (https://doi.org/10.1109/tci.2017.2777482)

**Original SLADS-Net**
**Subject:** Confocal Raman Microscopy  
**Citation(s):** S. Zhang, Z. Song, G. Godaliyadda, D. Ye, A. Chowdhury, A. Sengupta, G. Buzzard, C. Bouman, and G. Simpson, “Dynamic sparse sampling for confocal raman microscopy,” Analytical Chemistry, vol. 90, no. 7, pp. 4461–4469, mar 2018.  
**Available:** (https://doi.org/10.1021/acs.analchem.7b04749)

**Original SLADS-LS**  
**Subject:** X-Ray Diffraction  
**Citation(s):** N. Scarborough, G. Godaliyadda, D. Ye, D. Kissick, S. Zhang, J. Newman, M. Sheedlo, A. Chowdhury, R. Fischetti, C. Das, G. Buzzard, C. Bouman, and G. Simpson, “Dynamic x-ray diffraction sampling for protein crystal positioning,” Journal of Synchrotron Radiation, vol. 24, no. 1, pp. 188–195, jan 2017.  
**Available:** (https://doi.org/10.1107/s160057751601612x)

**Original SLADS-LS - Dissertation**  
**Subject:** Scanning Electron Microscopy, X-Ray Diffraction, Energy Dispersive Spectroscopy, Confocal Raman Microscopy  
**Citation(s):** G. M. D. P. Godaliyadda, “A Supervised Learning Approach for Dynamic Sampling (SLADS),” Ph.D. dissertation, Purdue University, 2017, copyright - Database copyright ProQuest LLC; ProQuest does not claim copyright in the individual underlying works; Last updated - 2022-01-06.  
**Available:** (https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2765&context=open_access_dissertations)

**Original SLADS-LS**  
**Subject:** Electron Backscatter Diffraction Microscopy  
**Citation(s):** G. M. D. Godaliyadda, D. Ye, M. D. Uchic, M. A. Groeber, G. T. Buzzard, C. A. Bouman, "A Supervised Learning Approach for Dynamic Sampling" in Proc. IS&T Int’l. Symp. on Electronic Imaging: Computational Imaging XIV, 2016  
**Available:** (https://doi.org/10.2352/ISSN.2470-1173.2016.19.COIMG-153)
