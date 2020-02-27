
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 SLADS - A supervised Learning Approach to Dynamic Sampling is an algorithm 
 that allows a user to dynamically sample an image.
											
 This code and the SLADS method were developed by G.M. Dilshan P. Godaliyadda^,
 Dong Hye Ye^, Michael Uchic*, Michael Groeber*, Gregery Buzzard# and Charles 
 Bouman^  
 ^ ECE department Purdue University					
 * Mathematics department Purdue University						 
 # Material Science Directorate, Air-Force Research Laboratory Dayton, OH

 This project was funded by:

 - Air Force Office of Scientific Research (MURI - Managing the Mosaic of 
   Microstructure, grant # FA9550-12-1-0458)
 - Air Force Research Laboratory Materials and Manufacturing directorate 
   (Contract # FA8650-10-D-5201-0038)		 

 For questions regarding the code contact <dilshangod@gmail.com>	   		
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++






===============================================================================
SECTION 0: How to use this code
===============================================================================

- Install Python (version 3.5+) and the Libraries numpy, pylab, scipy, sklearn,
  matplotlib. 
  You can also install them using Anaconda package. 
  <https://www.continuum.io/downloads>
  **** If version 2.7 is used instead make sure to enter ALL INPUTS for the 
       SCRIPTS as a FLOAT instead of an int e.g. 4 should be 4.0 (applies to
       numpy arrays as well)

- Read manuscript explaining the SLADS algorithm: 
  <https://engineering.purdue.edu/~bouman/publications/pdf/SLADS-2017.pdf>

- User must first train using images that look similar to the image (in 
  simulation) or object (in actual experiment) they wish to sample by following
  the instructions in Section 1

- Then to ensure the algorithm is performing properly the user is encouraged to
  run the script runSimulation.py on another image (similar to the object they
  wish to sample) by following the instructions in Section 2

- The user can now implement SLADS on an imaging device by following the 
  instructions in Section 3
===============================================================================






===============================================================================
SECTION 1: Instructions for Training SLADS using runTraining.py
===============================================================================

**** NOTE: cannot run SLADS simulation without training

This script allows the user to perform training for SLADS. 
- First the code will find training coefficients for different values of 'c' 
  and will automatically select the optimal 'c' for testing and save it for use
  in simulations or experiments. 
- The code also allows the user to find the stopping threshold for a desired
  distortion level. 

0. Select images that are:
   - similar to the intended testing data
   - the same format
   - the same size
   - not color images (i.e. only single grayscale value for each pixel)


1. Saving Images for training
	
	1.1. In folder './ResultsAndData/TrainingData/' create folder titled 
             'TrainingDB_X' (e.g 'TrainingDB_1')

	1.2. In folder './ResultsAndData/TrainingData/TrainingDB_X' create folder
             titled ‘Images’ and save images for training

	1.3. In folder './ResultsAndData/TrainingData/TrainingDB_X' create folder 
             titled 'ImagesToFindC' and  save images that will be used to choose the
             best 'c' (in the approximate RD) and the threshold on the stopping 
             function (that corresponds to a desired distortion level)


2. Initializing the script to run training

	2.1 In runTraining.py go to section 'USER INPUTS: L-0' and enter 
	    information that corresponds to training data saved in 
            './ResultsAndData/TrainingData/TrainingDB_X'

            ** Double check 'ImageType' is set to 'C' for continuous and 'D' 
               for discrete images
   	    ** All entries that need to be set here are very important and need to
	       match the training images

	2.2. In runTraining.py  modify section 'USER INPUTS: L-1'
             - if mask sizes in training need to be changes
             - if the user wants to change the approximate RD summation window size 
             - if the user wants to modify ERD update window size for SLADS
        
             - if the user wants to set Total distortion value for stopping 
               condition 
             - if the initial mask type for SLADS needs to be changed        
             ** If using mask type 'H' as initial mask make sure there is a folder  
                inside'./ResultsAndData/InitialSamplingMasks' that corresponds to 
                the settings of variable 'InitialMask' with a file 
                'SampleMatrix.npy'
                e.g. if initial percentage 1% and size of image 64x64 folder name:
                H_1_64_64_Percentage_1.0
   	     ** ONLY MODIFY IF user has EXPERT knowledge of SLADS training procedure
                ELSE DO NOT CHANGE
	     ** points 1 and 2 relate to finding coefficient vector in training
	        points 3 and 4 relate to the SLADS experiments when finding c
	        point 5 relates to finding stopping threshold


3. Run runTraining.py 
(can run on the terminal by typing  >> python3 runTraining.py)

** Resulting coefficient vector will be saved in:
   './ResultsAndData/TrainingSavedFeatures/TrainingDB_X/c_n'
   where 'n' in 'c_n' is the chosen value of 'c'
===============================================================================





===============================================================================
SECTION 2: Instructions for Running SLADS Simulation using runSLADSSimulation.py
===============================================================================

This script allows the user to run a simulation of SLADS on a desired image. 

1. Make sure training data is available for simulation
	
	1.1. If Training database 'X' is to be used for experiment make sure folder
             'TrainingDB_X' exists in path ‘./ResultsAndData/TrainingSavedFeatures/'
	1.2. If the value of 'c' chosen for experiment is 'n' make sure folder 'c_n'
	     exists in path './ResultsAndData/TrainingSavedFeatures/TrainingDB_X/'

2. Save testing data

	2.1. Save one image you wish to sample in a folder named 'TestingImageSet_Y'
             (e.g. TestingImageSet_1) in path './ResultsAndData/TestingImages/'

3. Initializing the script to run simulation

	3.1 In runSLADSSimulation.py go to section 'USER INPUTS: L-0' and enter 
	    information that corresponds to:
	    - training data saved in './ResultsAndData/TrainingData/
	      TrainingDB_X'
            - testing image saved in  './ResultsAndData/TestingImages/
	      TestingImageSet_Y'
	    - simulation details (e.g. image size, stopping percentage etc.)

            ** Double check 'ImageType' is set to 'C' for continuous and 'D' 
                for discrete images
   	    ** All entries that need to be set here are very important and need to 
	       match the training and testing data
	    ** If using mask type 'H' as initial mask make sure there is a folder  
               inside'./ResultsAndData/InitialSamplingMasks' that corresponds to the
               settings of variable 'InitialMask' with a file 'SampleMatrix.npy'
               e.g. if initial percentage 1% and size of image 64x64 folder name:
               H_1_64_64_Percentage_1.0

	3.2. In runSLADSSimulation.py modify section 'USER INPUTS: L-1'
	    - if for group-wise sampling needs to be changed
	    - if the user wants to modify ERD update window size for SLADS
 
4. Run runSLADSSimulation.py 
(can run on the terminal by typing  >> python3 runSLADSSimulation.py)

** The results will be saved in:
  './ResultsAndData/SLADSSimulationResults/Folder_Name'
   where, Folder_Name is the name of the folder you entered in runSLADSSimulation.py
===============================================================================







===============================================================================
SECTION 3: Instructions for Running SLADS on Imaging Device using runSLADS.py
           NOTE: Will NOT run without integrating code to imaging device
===============================================================================

This script allows the user to run an actual SLADS experiment by plugging in a
measurement routine that can acquire measurements using an imaging device. 
Note this will not run until this routine is included.

1. Make sure training data available for simulation
	
	1.1. If Training database 'X' is to be used for experiment make sure folder 
             'TrainingDB_X' exists in path './ResultsAndData/TrainingSavedFeatures/'
	1.2. If the value of 'c' chosen for experiment is 'n' make sure folder 'c_n'
	     exists in path './ResultsAndData/TrainingSavedFeatures/TrainingDB_X/'

2. Initializing the script to run simulation

	2.1 In runSLADS.py go to section 'USER INPUTS: L-0' and enter information
	    that corresponds to:
	    - training data saved in './ResultsAndData/TrainingData/
	    TrainingDB_X'
	    - experiment details (e.g. image size, stopping percentage etc.)

            ** Double check 'ImageType' is set to 'C' for continuous and 'D' for 
               discrete images
   	    ** All entries that need to be set here are very important and need to 
	       match the training and testing data
	    ** If using mask type 'H' as initial mask make sure there is a folder  
               inside'./ResultsAndData/InitialSamplingMasks' that corresponds to the
               settings of variable 'InitialMask' with a file 'SampleMatrix.npy'
               e.g. if initial percentage 1% and size of image 64x64 folder name:
               H_1_64_64_Percentage_1.0
	2.2. In runSLADS.py modify section 'USER INPUTS: L-1'
	     - if for group-wise sampling needs to be changed
	     - if the user wants to modify ERD update window size for SLADS

3. Open ./code/runSLADSOnce.py and search for CODE HERE (will show two 
   locations). In each location enter the code that will perform measurements
   according the specifications.
 
4. Run runSLADS.py (can run on the terminal by typing  >> python3 runSLADS.py)

** The results will be saved in './ResultsAndData/SLADSResults/Folder_Name'.
   Here, Folder_Name is the name of the folder you entered in runSLADS.py
===============================================================================







===============================================================================
SECTION 4: Content
===============================================================================

./code                                : folder with code

./ResultsAndData                      : folder with data needed for experiments
                                        and where results will saved

./ResultsAndData/InitialSamplingMasks : contains in folders the initial    
                                        measurement masks of type ‘H’ i.e. 
                                        low-discrepancy sampling 
                                        e.g. if initial percentage 1% and size 
                                        of image 64x64 folder name: 
                                        H_1_64_64_Percentage_1
                                        SampleMatrix.npy: Matrix with 0's and 
                                        1's, 1 - measurement location
				
./ResultsAndData/SLADSSimulationResults: folder where SLADS simulation results 
                                         are saved

./ResultsAndData/SLADSResults          : folder where SLADS results are saved


./ResultsAndData/TestingImages         : contains images that will be used for 
                                         SLADS simulations (i.e to run 
                                         runSLADSSimulation.py)

./ResultsAndData/TrainingData          : Contains the results of training and
                                         the images used for training

./ResultsAndData/TrainingSavedFeatures : folder where coefficient vectors 
                                         computed in training will be saved	
===============================================================================
