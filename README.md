 	██████╗ ██╗      █████╗ ██████╗ ███████╗
 	██╔══██╗██║     ██╔══██╗██╔══██╗██╔════╝
 	██║  ██║██║     ███████║██║  ██║███████╗
 	██║  ██║██║     ██╔══██║██║  ██║╚════██║
 	██████╔╝███████╗██║  ██║██████╔╝███████║
 	╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝

# GENERAL INFORMATION

    NAME:           SLADS/DLADS/GLANDS
    VERSION:        0.9.6
    LICENSE:        GNU General Public License v3.0
    DESCRIPTION:    Dynamic sampling algorithms with updated/developing implementations of:
                       -SLADS-LS    Supervised Learning Approach for Dynamic Sampling, using Least-Squares (LS) regression
                       -SLADS-Net   Supervised Learning Approach for Dynamic Sampling, using a Multi-Layer Perceptron (MLP) network
                       -DLADS       Deep Learning Approach for Dynamic Sampling, using a Convolutional Neural Network (CNN)
                       -GLANDS      Generative Learning Adversarial Network for Dynamic Sampling, using a Generative Adversarial Network (GAN)
    
    AUTHOR(S):      David Helminiak    EECE, Marquette University
    ADVISOR(S):     Dong Hye Ye        COSC, Georgia State University
    
    FUNDING:        This project originally received funding and was programmed for NIH Grant 1UG3HL145593-01
    
    CHANGELOG:      0.1.0   Multithreading adjustments to pointwise SLADS
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
                    0.6.6   SLADS-Net NN, PSNR, asymFinal, and multi-config
                    0.6.7   Clean asymmetric implementation with density features
                    0.6.8   Fixed RD generation, added metrics, and Windows compatible
                    0.7.0   CNN/U-Net/RBDN with dynamic window size
                    0.7.1   c value selection performed before model training
                    0.7.2   Remove custom pkg. dependency, use NN resize, recon+measured input
                    0.7.3   Start/End line patch, SLADS(-Net) options, normalization optimization
                    0.7.4   CPU compatibility patch, removal of NaN values
                    0.7.5   c value selection performed before training database generation
                    0.6.9   Do not use -- Original SLADS(-Net) variations for comparison with 0.7.3
                    0.8.0   Raw MSI file integration (.raw, .d), .d files only Windows compatible
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
                    0.9.2   .imzML, Bruker .d, image support, RD speedup, fix RD times, single sample training, FOV mask support
                    0.9.3   Whole spectra metrics, improved data aug. and file loading, fix RAM OOM, .imzML out, I/O norm. options
                    0.9.4   Fix group-based and c value selection, distributed multi-GPU simulations, updated Otsu segmented linewise
                    0.9.5   MALDI optical, input config, .tdf/.tsf compatibility, disable whole spectra metrics option, .imzML output
                    0.9.6   Adjust ERD processing and visualizations, add progression map, compute times, and alphatims option
                    x.x.x+  Implementation line revisiting
                    x.x.x+  Add norm. options
                    x.x.x+  GLANDS
                    x.x.x+  Iterative feature selection mechanism for selection of target channels
                    x.x.x+  Experimental MALDI integration
                    x.x.x+  Release installation method for python package manager

# PROGRAM FILE STRUCTURE
**Note:** If testing/training is not to be performed, then contents of 'TEST', 'TRAIN', may be disregarded, but a trained model must be present in: ./RESULTS/TRAIN/.

**Warning:** If training is enabled, any model already in ./RESULTS/TRAIN/ will be overwritten. Likewise, if testing or implementation is enabled, then any data in ./RESULTS/TEST/ and/or ./RESULT/IMP/, respectively will be overwritten.

    ------->ROOT_DIR
        |------->START.py
        |------->README.md
        |------->CONFIG_#-description.py
        |------->channels.csv
        |------->CODE
        |    |------->AESTHETICS.py
        |    |------->COMPUTE.py
        |    |------->DEFINITIONS.py
        |    |------->EXPERIMENTAL.py
        |    |------->EXTERNAL.py
        |    |------->INTERNAL.py
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
        |    |    |    |------->PSNR and SSIM Results (.csv and .png)
        |    |    |------->dataPrintout.csv
        |    |    |------->PSNR and SSIM Results (.csv and .png)
        |    |------->VALIDATION
        |    |    |------->VALIDATION_SAMPLE_1
        |    |    |    |------->Channels
        |    |    |    |    |------->...
        |    |    |    |------->Progression
        |    |    |    |    |------->...
        |    |    |    |------->Videos
        |    |    |    |    |------->...
        |    |    |    |------->measuredMask.csv
        |    |    |    |------->PSNR and SSIM Results (.csv and .png)
        |    |    |------->dataPrintout.csv
        |    |    |------->PSNR and SSIM Results (.csv and .png)
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
        |    |    |------->history.csv
        |    |    |------->optimalC.npy
        |    |    |------->trainingDatabase.p
        |    |    |------->trainingValidationSampleData.p
        |    |    |------->trainingValidation_RDTimes.csv
        |    |    |------->validationDatabase.p
        
# INSTALLATION
**Note:** Use of Agilent .d files is only possible on Windows, given a reliance on vendor provided .dll's, as utilized by the multiplierz package. The package versions do not necessarily need to match with those listed. However, should the program produce unexpected errors, installing a specific version of a package might be able to resolve the issue. Note that the multiplierz pacakage, must be installed from the provided link under the installation commands.

This code has been verified to function on Windows 10/11, CentOS 7/8 (through Docker containers) and Ubuntu 18.04/20.04 operating systems. As more functionality is being added, minimum hardware specifications cannot be exactly ascertained, however validation of functionality is performed on systems containing 64+ GB DDR3/4/5 RAM, 32+ CPU threads at 3.0+ GHz, 1080Ti/2080Ti+ GPUs, and 1TB+ SSD storage. While v0.8.9 and below have managed to utilize pre-trained models with only a dual core CPU, 8 GB DDR2, and no discrete GPU, this is not an advisable set of hardware for utilizing this program. 

    **Software**
    Python             3.8.10
    pip                22.2.2

    **Python Packages**
    aiorwlock          1.3.0
    colorama           0.4.6
    datetime           5.0
    glob2              0.7
    graphviz           0.20.1
    IPython            8.11.0
    joblib             1.2.0
    matplotlib         3.7.0
    multiprocess       0.70.14
    natsort            8.3.0
    numba              0.56.4
    numpy              1.23.5
    opencv-python      4.7.0.72
    pandas             1.5.3
    pathlib            1.0.1
    pillow             9.4.0
    psutil             5.9.4
    pydot              1.4.2
    pyimzml            1.5.3
    ray                2.1.0 
    scikit-image       0.20.0
    scikit-learn       1.2.1
    scipy              1.9.1
    sobol              0.9
    sobol-seq          0.2.0
    tensorflow-addons  0.18.0
    tensorflow-gpu     2.8.4 
    tqdm               4.64.1


### **Installation on Windows 10 and 11**

If not already setup, install Python 3.8 selecting the options to install for all users, and addding python.exe to PATH: https://www.python.org/downloads/release/python-3913/

Enable ".Net Framework 3.5 (includes .NET 2.0 and 3.0)" under "Windows Features" (search for in the Windows Start Menu)

Then install 2019 Visual Studio Build Tools, checking boxes for "Desktop development with C++" and "Universal Windows Platform development": https://visualstudio.microsoft.com/vs/older-downloads/

Next install Visual Studio Community 2019, with the additional options: "Desktop development with C++" , "Universal Windows Platform development", and "Python development"
Note: Any version more recent than 2019 will not be able to compile dependencies for reading MSI files with "multiplierz" package. 

Install Git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

Install CMake, checking the option to add CMake to the system PATH for all users): https://cmake.org/install/

If GPU acceleration is to be used, a compatible CUDA Toolkit and cuDNN (and zlib) must be installed on the system, follow the instructions at: https://docs.nvidia.com/deeplearning/cudnn/install-guide/
For the cudnn dependency zlib, download it from the link provided in the install guide and copy zlibwapi.dll from the included dll_x64 subflolder to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x.x\bin (where x.x.x depend on the CUDA version installed)
Open "Edit System Variables" (search for in the Windows Start Menu), click "Environment Variables" and add a new system variable: XLA_Flags with the value: --xla_gpu_cuda_data_dir='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7'

Open the command prompt as an administrator (right-click on the command prompt icon and choose "Run as administrator")

For 64-bit Python, configure the build environment by entering the following lines into command prompt, where for 32-bit replace x64 by x86.
The actual location of the specified file may vary depending on potential past Visual Studio installations. 

    $ SET DISTUTILS_USE_SDK=1
    $ "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

Open a new command prompt (critically, not as an administrator!) and enter the following commands (If GPU acceleration is not to be used change tensorflow-gpu==2.8.4 below to tensorflow==2.8.4):

    $ python -m pip install --upgrade pip
    $ pip3 install datetime glob2 IPython joblib pandas pathlib psutil matplotlib numpy numba pillow ray[serve] scipy sobol sobol-seq natsort multiprocess scikit-image scikit-learn tensorflow-gpu==2.8.4 tensorflow-addons==0.18.0 tqdm opencv-python pydot graphviz aiorwlock pyimzml colorama pywin32
    $ pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@master
	$ pip3 install git+https://github.com/Yatagarasu50469/alphatims.git@master

Either switch back to, or open a new command prompt as an administrator and enter the following command:
    
    $ python -c "from multiplierz.mzAPI.management import registerInterfaces; registerInterfaces()"

If the final printout indicates actions relating to the MSI file format intended for use, then follow through as neccessary. 

# TRAINING/TESTING PROCEDURE
###  **CONFIGURATION**

**Note:** 
There are very few sanity checks inside of the code to ensure only correct/valid configurations are used. If an unexpected error occurs, please double check the sample and program configuration files to ensure they are correct before opening an issue or contacting the program author. Thank you.

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

For MALDI samples, aligned optical images may be incorporated into the model, however all samples must have an image of the same resolution included and training must have been conducted with the same optical flags enabled as intended to be used during simulation/implementation. The optical image should be placed in each sample's directory as 'optical.png'.

For DESI Bruker .tdf samples (.d files with timsTOF enabled) alphatims may be used instead of multiplierz, by specifying the sample type DESI-ALPHA. Alphatims is only compatible with DESI Bruker .tdf files, use of alphatims with Bruker .tsf, or other vendor MSI files is not supported!

###  **DESI MSI**
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

###  **MALDI MSI**
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

###  **Images**
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

###  **RUN**
After configuration, to run the program perform the following command in the root directory:

    $ python ./START.py

###  **RESULTS**
All results will be placed in ./RESULTS/ as follows (presuming MSI data):

    TRAIN: Training results
        Model Training Images: Progressive model training convergence plots
        model_modelType_channelType_windowType_windowDim_c_cValue: Trained model corresponding to the indicated parameters (.npy for SLADS-LS and SLADS-Net)
        Training Data Images: Training images with/without borders, summary sample images, and c value curves 
        Validation Data Images: Validation images with/without borders and summary sample images
        history.csv: Progresive model training losses
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

# EXPERIMENTAL IMPLEMENTATION

**Note:** Currently only supported/validated for DESI MSI equipment; MALDI and general image formats are under development.

**Note:** In order to use a trained model in a physical implementation, the files resultant from the training procedure must be located within './RESULTS/TRAIN/'.

1\. Prior to engaging the physical equipment, run the program with the **impModel** variable enabled in the configuration file. A custom implementation input directory, to use instead of the default: './INPUT/IMP/', may be specified with the **impInputDir** variable, under section L1-1, during configuration. All other testing and training flags within in the configuration file, under section L0, should be disabled. 

2\. Initialize the program with 'python START.py' and place 'sampleInfo.txt' and 'channels.csv' (if not using a global 'channels.csv' file in the SLADS/DLADS root directory) into the implementation input directory. The 'sampleInfo.txt' and 'channels.csv' files should follow the format outlined in the CONFIGURATION section of the README documentation. 

3\. Place a blank file named **LOCK** into the implementation input directory to signal the program that the 'sampleInfo.txt' and 'channels.csv' files have been placed. 

2\. Creation of the **LOCK** file triggers the program to read in any data saved in the implementation input directory and produce a set of points (row number, and column positions in um) to physically scan. This data is saved in a file: **UNLOCK**. 

3\. **LOCK** will be automatically deleted to signal the equipment that **UNLOCK** now contains new measurement positions.

4\. The program waits for **LOCK** to re-appear, which should be created after the requested data has been scanned into the implementation input directory. The data must follow the formatting specified in the TRAINING/TESTING PROCEDURE section of the README documentation. 

5\. When the termination criteria have been met, the program produces a file: **DONE**, instead of: **UNLOCK**, to signal the equipment that scanning has concluded. 

# FAQ
###  **I read through the README thoroughly, but I'm still getting an error, am confused about how a feature should work, or would like a feature/option added**

Please check if it has already been addressed, or open an issue on the Github repository: https://github.com/Yatagarasu50469/SLADS/issues
Also, be aware that at this time there is no legacy version support; it would be advisable to verify that the latest release is being used and installed packages match with those listed in the corresponding README. 

###  **ImportError: cannot import name 'Iterator' from 'collections'**

The version of mulitiplierz originally installed up to v0.9.2 only functions was tested up to python 3.9.31. For python 3.10+ (which ray does not currently support without errors), you will need to uninstall the existing multiplierz installation and re-download it from the forked repository, where a patch has since been applied. 

    $ pip3 uninstall multiplierz
    $ pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@master

###  **Why am I receiving a 'list index out of range' error from the 'readScanData' method**

Most likely this is due to filenames not matching the outlined naming convention.

###  **Why is SLADS not compatible with Linux distributions, or Mac operating systems when using MSI data**

As of v0.8.0, SLADS obtains information directly from MSI files, rather than pre-processed .csv m/z visualizations. For Agilent .d files, these operations are reliant on vendor specific .dll files as provided in the multiplierz package. Supperficially it appears as though the multiplierz API might function within Linux. For example, the packages pythonnet and comtypes can be installed, but cannot actually function in a linux environment. An alternative approach, that may work, might be to attempt an installation through wineDocker, though this has not been attempted. 

While it does not currently function for some MSI formats, (verified operational for .RAW and .imzML/.ibd files) multiplierz may be installed directly on Ubuntu 18.04+ or in a Docker container with the following commands:
    
    $ python -m pip install --upgrade pip
    $ sudo apt-get update
    $ sudo apt-get install -y wget git python3-opencv
    $ pip3 install datetime glob2 IPython joblib pandas pathlib2 psutil matplotlib numba pillow ray[serve] scipy sobol sobol-seq natsort multiprocess scikit-image sklearn tensorflow-gpu=2.8.4 tensorflow-addons==0.18.0 tqdm numpy opencv-python pydot graphviz aiorwlock pyimzml colorama
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
    $ pip3 install git+https://github.com/pythonnet/pythonnet.git@2ad4e7006a3646be8457f74940541812e79a926e
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

Previous version install procedures have the multiplierz package installed with pip3 as an administrator, which causes issues when running the program as a non-administrator. As an administrator, uninstall multiplierz (pip3 uninstall multiplierz), then install as per the updated instructions. Note that following the installation, the registerInterfaces command still needs to be performed as an administrator.

###  **Program seems very slow**

The most likely cause is from large sample dimensionality (i.e. both rows and columns are in excess of 100). It the computational platform appears to lock up, it is recommended to check either Task Manager (Windows) or htop (Ubuntu: sudo apt-get install htop) and verify there is both sufficient RAM and computational capabilities to utilize the selected model. DLADS is much more computationally intensive than SLADS-LS, but will try to offload work onto a discrete CUDA GPU if one is available. Warnings/Errors will be output if RAM limits are exceeded on either system or GPU. 

Verify that liveOutputFlag in the configuration file is set to False, visualization and metric extraction proceudreas are very computationally expensive as reconstructions are generated across the whole spectrum and visualizations are written onto a physical drive, rather than held in RAM.

Verify that parallelism is enabled as the program can perform some operations concurrently. 

###  **Error indicates that the GPU should be disabled, settting parallelism to False doesn't resolve this**

The parallelism flag in the configuration file does not control GPU usage, only CPU usage. In order to disable GPUs entirely, set availableGPUs = '-1'.

###  **Received a Windows Access Error**

Windows (at least in versions before 11) may prevent file/folder operations if they are currently open. Since folders are removed and generated during startup, make sure all explorer windows and files related/resulting to/from the program are closed before starting a run. If the error persists, then verify that the specified files/folders have appropriate permissions.

###  **Looking at the model training images for DLADS shows a lack of loss convergence**

Try disabling data augmentation, setting augTrainData to False, if this fixes the issue than likely the default parameters(specified in TRAINING.py under the DataGen class), particularly the random crop size, being used may be too large to extract meaningful features. If this doesn't fix the issue, than try lowering the learning rate and/or increasing the number of start filters. Beyond those parameters, provided there are sufficient training samples, a change in the network architecture may be required.

###  **Looking at the model training images for DLADS shows significant overfit**

Double check that data augmentation is enabled, otherwise decrease the number of start filters used for the model. If this still does not fix the issue, it may be desirable to use a less complex model than DLADS, such as the included SLADS-LS or SLADS-Net options.

###  **The resulting reconstructions are quite blurry**

If the provided data is fairly homogeneous, with predominant value fluctuations around structural edges, try decreasing the number of neighbors used in IDW reconstruction (numNeighbors) as low as 1.

###  **The program appears to regularly freeze and I can't interact with the computer**

Certain sections of the program are extremely compute/memory/storage instensive and are expected to freeze the graphical output of even upper-end hardware; to verify check Task Manager on Windows, htop on Linux-based platforms, and/or nvidia-smi output for utlization levels. 

###  **Most of the code appears to focus on integration with MSI modalities, can it be used for others?**

Absolutely! Currently support is offered for regular images with single and multiple channels. If compatability with another file format is needed for your use case, please either open an issue in the repository, or feel free to fork this project and open a pull request with the added functionality.

###  **Why the limitation to TensorFlow version 2.8**

There are a couple of limiting factors. Above v2.8, TensorFlow incorporates a newer version of Keras that attempts and fails to vectorize the data augmentation layers, falling back on inefficient while loops, which harms training performance. Further, v2.11 no longer offers native CUDA/GPU acceleration, instead requiring the use of Microsoft's directML plugin (replace tensorflow-gpu in pip package installations with tensorflow-directml-plugin and install Windows Subsystem Linux (WSL). This should allow for training non-NVIDIA GPUs (untested), but cannot presently handle multi-GPU training with a distributed strategy. Manually disabling the strategy (tested, but no option included in configuration) does allow the plugin to work, though there is still a noticable performance penalty. If a more modern version of TensorFLow is required for your workflow, please open an issue in the GitHub repository for creation of a directML option, or if there is not a need for Agilent .d file compatability, follow the Linux install instructions (under another FAQ entry) after installing Ubuntu on WSL. 

###  **v0.9.4 and below cannot read MSI data**

The install branch for mulitiplierz was modified as of v0.9.5 and is not backwards compatible. You will need to uninstall the existing (default) multiplierz installation and re-download it from an older commit of the forked repository. 

    $ pip3 uninstall multiplierz
    $ pip3 install git+https://github.com/Yatagarasu50469/multiplierz.git@4a458283441f609e2f4118ca462073a97f401bdc

### **Receiving "cuda_malloc_async isn't currently supported on GPU..." indicates an issue with my CUDA version, but I have verified the installation is correct.**

If your NVIDIA GPU architecture is older than Pascal (i.e. Maxwell and earlier with compute capability below 6.1), then you will need to comment out the following line in EXTERNAL.py:

    os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

# PUBLICATIONS

###  **RESEARCH PRODUCED WITH THIS CODE**

**Simulatated Acquisition of MALDI MSI with DLADS**\
**Version(s):** v0.9.2\
**Subject:** MALDI MSI\
**Citation(s):** Upcoming\
**Available:** Upcoming

**Updated Simulatated Acquisition of DESI MSI with DLADS**\
**Version(s):** v0.9.1\
**Subject:** DESI MSI\
**Citation(s):** D. Helminiak, H. Hu, J. Laskin, and D. H. Ye, “Deep Learning Approach for Dynamic Sampling for Multichannel Mass Spectrometry Imaging.” arXiv, 2022.\
**Available:** https://arxiv.org/abs/2210.13415

**Experimental DESI MSI Integration with DLADS**\
**Version(s):** Experimental results were determined with v0.8.9, and simulation results with v0.9.1\
**Subject:** DESI MSI\
**Citation(s):** H. Hu, D. Helminiak, M. Yang, D. Unsihuay, R.T. Hilger, D. H. Ye, and J. Laskin, "High-throughput mass spectrometry imaging with dynamic sparse sampling," ACS Measurement Science Au, Aug. 2022. \
**Available:** https://pubs.acs.org/doi/10.1021/acsmeasuresciau.2c00031

**Development for Simulatated Acquisition of DESI MSI with DLADS - Master's Thesis**\
**Version(s):** v0.8.6\
**Subject:** DESI MSI\
**Citation(s):** D. S. Helminiak, “Deep learning approach for dynamic sampling for high-throughput nano-desi msi,” Master's Thesis, Marquette University, 2021, copyright - Database copyright ProQuest LLC; ProQuest does not claim copyright in the individual underlying works; Last updated - 2022-02-21.\
**Available:** https://epublications.marquette.edu/theses_open/710/ 

**Initial Simulatated Acquisition of with DLADS**\
**Version(s):** v0.6.9 and v0.7.3\
**Subject:**  DESI MSI\
**Citation(s):** D. Helminiak, H. Hu, J. Laskin, and D. H. Ye, “Deep learning approach for dynamic sparse sampling for high-throughput mass spectrometry imaging,” Electronic Imaging, vol. 2021, no. 15, pp. 290–1–290–7, Jan. 2021.\
**Available:** https://doi.org/10.2352/issn.2470-1173.2021.15.coimg-290

###  **SIMILAR & PRECEEDING RESEARCH**

**Original SLADS-Net**\
**Version(s):** https://github.com/cphatak/SLADS-Net, https://github.com/anl-msd/SLADS-Net\
**Citation(s):** Y. Zhang, G. Godaliyadda, N. Ferrier, E. Gulsoy, C. Bouman, and C. Phatak, “Slads-net: Supervised learning approach for dynamic sampling using deep neural networks,” 2018.\
**Subject:**  Scanning Electron Microscopy\
**Available:** https://doi.org/10.2352/ISSN.2470-1173.2018.15.COIMG-131

**Unsupervised-SLADS (U-SLADS)**\
**Citation(s):** Y. Zhang, X. Huang, N. Ferrier, E. Gulsoy, and C. Phatak, “U-slads: Unsupervised learning approach for dynamic dendrite sampling,” 2018\
**Subject:**  Metal Dendrite Sampling\
**Available:** https://arxiv.org/abs/1807.02233

**Original SLADS-LS**\
**Subject:**  Energy Dispersive X-Ray Spectroscopy\
**Citation(s):** Y. Zhang, G. Godaliyadda, N. Ferrier, E. Gulsoy, C. Bouman, and C. Phatak, “Reduced electron exposure for energy-dispersive spectroscopy using dynamic sampling,” Ultramicroscopy, vol. 184, pp. 90 – 97, 2018.\
**Available:** https://sciencedirect.com/science/article/pii/S0304399117303157

**Original SLADS-LS**\
**Citation(s):** G. Godaliyadda, D. Ye, M. Uchic, M. Groeber, G. Buzzard, and C. Bouman, “A framework for dynamic image sampling based on supervised learning,” IEEE Transactions on Computational Imaging, vol. 4, no. 1, pp. 1–16, mar 2018.\
**Subject:** Electron Back Scatter Diffraction\
**Available:** https://doi.org/10.1109/tci.2017.2777482

**Original SLADS-Net**
**Citation(s):** S. Zhang, Z. Song, G. Godaliyadda, D. Ye, A. Chowdhury, A. Sengupta, G. Buzzard, C. Bouman, and G. Simpson, “Dynamic sparse sampling for confocal raman microscopy,” Analytical Chemistry, vol. 90, no. 7, pp. 4461–4469, mar 2018.\
**Subject:** Confocal Raman Microscopy\
**Available:** https://doi.org/10.1021/acs.analchem.7b04749

**Original SLADS-LS**\
**Citation(s):** N. Scarborough, G. Godaliyadda, D. Ye, D. Kissick, S. Zhang, J. Newman, M. Sheedlo, A. Chowdhury, R. Fischetti, C. Das, G. Buzzard, C. Bouman, and G. Simpson, “Dynamic x-ray diffraction sampling for protein crystal positioning,” Journal of Synchrotron Radiation, vol. 24, no. 1, pp. 188–195, jan 2017.\
**Subject:** X-Ray Diffraction\
**Available:** https://doi.org/10.1107/s160057751601612x

**Original SLADS-LS - Dissertation**\
**Citation(s):** G. M. D. P. Godaliyadda, “A Supervised Learning Approach for Dynamic Sampling (SLADS),” Ph.D. dissertation, Purdue University, 2017, copyright - Database copyright ProQuest LLC; ProQuest does not claim copyright in the individual underlying works; Last updated - 2022-01-06.\
**Subject:** Scanning Electron Microscopy, X-Ray Diffraction, Energy Dispersive Spectroscopy, Confocal Raman Microscopy\
**Available:** https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2765&context=open_access_dissertations

**Original SLADS-LS**\
**Citation(s):** G. M. D. Godaliyadda, D. Ye, M. D. Uchic, M. A. Groeber, G. T. Buzzard, C. A. Bouman, "A Supervised Learning Approach for Dynamic Sampling" in Proc. IS&T Int’l. Symp. on Electronic Imaging: Computational Imaging XIV, 2016\
**Subject:** Electron Backscatter Diffraction Microscopy\
**Available:** https://doi.org/10.2352/ISSN.2470-1173.2016.19.COIMG-153
