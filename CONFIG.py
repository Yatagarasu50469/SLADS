#====================================================================
#SLADS CONFIGURATION
#====================================================================

#PARAMETERS: L0
#==================================================================
#Is training of a model to be performed
trainingModel = True

#Is testing of a model to be performed
testingModel = True

#Should leave-one-out Cross Validation be performed instead of testing
#Not implemented at this time
LOOCV = False

#Is this an implementation run
#Not implemented at this time
impModel = False

#PARAMETERS: L1
#==================================================================

#Window size for approximate RD summation; 15 for 512x512
windowSize = [3, 21]

#Stopping percentage for number of acquired pixels
stopPerc = 50

#PARAMETERS: L2
#==================================================================

#Sampling percentages for training
measurementPercs = [1, 5, 10, 20, 30, 40]

#Possible c values for RD approximation
cValues = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

#Should animations be generated during testing/implementation
animationGen = True

#Type of Images: D - for discrete (binary) image; C - for continuous
#Not implemented at this time
imageType = 'C'

#Should a stopping threshold be found that corresponds to the best determined c value
#Not implemented at this time
findStopThresh = False

#Percent of reduction in distrotion limit for numRandomChoice determination
percOfRD = 50

#Set the number of available CPU threads, leave 2 free if possilbe
num_threads = multiprocessing.cpu_count()
if num_threads > 2: num_threads -= 2

#Running in a console/True, jupyter-notebook/False
consoleRunning = True
