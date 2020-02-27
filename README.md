▄▄▄▄▄▄▄▄▄▄▄ ▄ ▄▄▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄▄▄▄▄

▐░░░░░░░░░░░▐░▌ ▐░░░░░░░░░░░▐░░░░░░░░░░▌▐░░░░░░░░░░░▌

▐░█▀▀▀▀▀▀▀▀▀▐░▌ ▐░█▀▀▀▀▀▀▀█░▐░█▀▀▀▀▀▀▀█░▐░█▀▀▀▀▀▀▀▀▀

▐░▌ ▐░▌ ▐░▌ ▐░▐░▌ ▐░▐░▌

▐░█▄▄▄▄▄▄▄▄▄▐░▌ ▐░█▄▄▄▄▄▄▄█░▐░▌ ▐░▐░█▄▄▄▄▄▄▄▄▄

▐░░░░░░░░░░░▐░▌ ▐░░░░░░░░░░░▐░▌ ▐░▐░░░░░░░░░░░▌

▀▀▀▀▀▀▀▀▀█░▐░▌ ▐░█▀▀▀▀▀▀▀█░▐░▌ ▐░▌▀▀▀▀▀▀▀▀▀█░▌

▐░▐░▌ ▐░▌ ▐░▐░▌ ▐░▌ ▐░▌

▄▄▄▄▄▄▄▄▄█░▐░█▄▄▄▄▄▄▄▄▄▐░▌ ▐░▐░█▄▄▄▄▄▄▄█░▌▄▄▄▄▄▄▄▄▄█░▌

▐░░░░░░░░░░░▐░░░░░░░░░░░▐░▌ ▐░▐░░░░░░░░░░▌▐░░░░░░░░░░░▌

▀▀▀▀▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀▀▀▀▀ ▀ ▀ ▀▀▀▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀▀▀▀▀

\#====================================================================

**GENERAL INFORMATION**

\#====================================================================

NAME: lineSLADS

VERSION NUM: 0.5

DESCRIPTION: Multichannel implementation of SLADS (Supervised Learning
Algorithm

for Dynamic Sampling with additional constraint to select groups of
points

along a single axis.

AUTHORS: David Helminiak EECE, Marquette University

Dong Hye Ye EECE, Marquette University

COLLAB. Julia Laskin CHEM, Purdue University

Ruichuan Yin CHEM, Purdue University

Hang Hu CHEM, Purdue University

FUNDING: This project has received funding and was programmed for:

NIH Grant 1UG3HL145593-01

GLOBAL

CHANGELOG: 0.1 Multithreading adjustments to pointwise SLADS

0.1.1 Line constraints, concatenation, pruning, and results organization

0.2 Comple program rewrite

0.3 Complete code rewrite, computational improvements

0.4 Class/function segmentation

0.5 Overhead reduction; switch multiprocessing package

\~0.6 Modifications for Nano-DESI microscope integration

\~0.7 Tissue model library generation

\~0.8 Deep feature extraction

\~0.9 GPU acceleratiaon

\~1.0 Initial release

\#====================================================================

\#====================================================================

**PROGRAM FILE STRUCTURE**

\#====================================================================

**Note**: If testing/training is not to be performed, then contents of
'TEST', 'TRAIN', may be disregarded, but a trained SLADS model must be
present in: ./RESULTS/TRAIN/.

**Warning**: If training is enabled, any model already in
./RESULTS/TRAIN/ will be overwritten. Likewise, if testing or
implementation is enabled, then any data in ./RESULTS/TEST/ and/or
./RESULT/IMP/, respectively will be overwritten.

\-------\>ROOT\_DIR

|-------\>README

|-------\>CONFIG.py

|-------\>SLADS.py

|-------\>CODE

| |-------\>DEFS.py

| |-------\>EXPERIMENTAL.py

| |-------\>EXTERNAL.py

| |-------\>INTERNAL.py

| |-------\>TESTING.py

| |-------\>TRAINING.py

|-------\>INPUT

| |-------\>TEST

| | |-------\>TEST\_SAMPLE\_1

| | | |-------\>
mz***lowMZ1**\_**highMZ1***fn***sampleName***ns***numRows***.csv

| | | |-------\>
mz***lowMZ2**\_**highMZ2***fn***sampleName***ns***numRows***.csv

| | |-------\>TEST\_SAMPLE\_2

| | | |-------\>
mz***lowMZ1**\_**highMZ1***fn***sampleName***ns***numRows***.csv

| | | |-------\>
mz***lowMZ2**\_**highMZ2***fn***sampleName***ns***numRows***.csv

| |-------\>TRAIN

| | |-------\>TRAIN\_SAMPLE\_1

| | | |-------\>
mz***lowMZ1**\_**highMZ1***fn***sampleName***ns***numRows***.csv

| | | |-------\>
mz***lowMZ2**\_**highMZ2***fn***sampleName***ns***numRows***.csv

| | |-------\>TEST\_SAMPLE\_2

| | | |-------\>
mz***lowMZ1**\_**highMZ1***fn***sampleName***ns***numRows***.csv

| | | |-------\>
mz***lowMZ2**\_**highMZ2***fn***sampleName***ns***numRows***.csv

| |-------\>IMP

| | |-------\>
mz***lowMZ1**\_**highMZ1***fn***sampleName***ns***numRows***.csv

| | |-------\>
mz***lowMZ2**\_**highMZ2***fn***sampleName***ns***numRows***.csv

| | |-------\> UNLOCK

| | |-------\> LOCK

| | |-------\> DONE

|-------\>RESULTS

| |-------\>TEST

| | |-------\>Animations

| | |-------\>dataPrintout.csv

| | |-------\>mzResults

| | | |-------\>TEST\_SAMPLE\_1

| | | | |-------\> ***lowMZ1**\_**highMZ1***.png

| | | | |-------\> ***lowMZ2**\_**highMZ2***.png

| | | |-------\>TEST\_SAMPLE\_2

| | | | |-------\> ***lowMZ1**\_**highMZ1***.png

| | | | |-------\> ***lowMZ2**\_**highMZ2***.png

| | |-------\>testingAverageSSIM\_Percentage.csv

| | |-------\>testingAverageSSIM\_Percentage.png

| |-------\>TRAIN

| | |-------\>bestC.npy

| | |-------\>bestTheta.npy

| | |-------\>cValues.npy

| | |-------\>Images

| | | |-------N/A

| | |-------\>trainedModels.npy

| | |-------\>trainingSamples.p

\#====================================================================

\#====================================================================

**INSTALLATION**

\#====================================================================

This implementation of SLADS has functioned on Windows, Mac, and Linux
operating systems, with a clean Windows 10 installation described below.
The package versions do not necessarily need to match with those listed,
but should the program produce unexpected errors, installing a specific
version of a package might be able to resolve the issue.

**<span class="underline">Operating System</span>**

**Win. 10**: Updated as of Jan 1 2020

**Ubuntu**: 18.04

**Mac**: 10.13.6

**<span class="underline">System</span>**

**Python** 3.8.1

**pip** 19.3.1

**<span class="underline">Python Packages</span>**

**backcall**: 0.1.0

**colorama**: 0.4.3

**cycler**: 0.10.0

**decorator**: 4.4.1

**dill**: 0.3.1.1

**datetime**: 4.3

**glob3**: 0.0.1

**imageio**: 2.6.1

**IPython**: 7.11.1

**ipython-genutils**: 0.2.0

**jedi**: 0.15.2

**joblib**: 0.14.1

**kiwisolver**: 1.1.0

**pandas**: 0.25.3

**parso**: 0.5.2

**python-dateutil**: 2.8.1

**numpy**: 1.18.1

**matplotlib**: 3.1.2

**multiprocess**: 0.70.9

**natsort**: 6.2.0

**networkx**: 2.4

**opencv-python**: 4.1.2.30

**pickleshare**: 0.7.5

**pillow**: 7.0.0

**prompt-toolkit**: 3.0.2

**psutil**: 5.6.7

**pygments**: 2.5.2

**pyparsing**: 2.4.6

**pytz**: 2019.3

**PyWavelets**: 1.1.1

**ray**: 0.8.0

**scipy**: 1.4.1

**six**: 1.13.0

**scikit-image**: 0.16.2

**scikit-learn**: 0.22.1

**sklearn**: 0.0

**sobol**: 0.9

**sobol\_seq**: 0.1.2

**traitlets**: 4.3.3

**tqdm**: 4.41.1

**wcwidth**: 0.1.8

**zope.interface** 4.7.1

**<span class="underline">Installation on Mac/Linux</span>**

**Note**: These instructions have not been tested on a clean system
running only the operating systems specified, but should be expected to
function.

$ python --m pip install --upgrade pip

$ pip3 install opencv-python datetime glob3 IPython joblib pandas psutil
matplotlib pillow

ray scipy sobol sobol\_seq natsort multiprocess ray scikit-image sklearn
tqdm

**<span class="underline">Installation on Windows 10</span>**

**Note:** At this time, for Windows 10, the Linux subsystem must be
used, as the multiprocessing package **ray** has not been (nor is likely
to be in the near future) released for Windows.

Open **PowerShell** as an Administrator and run the following:

$ Enable-WindowsOptionalFeature -Online -FeatureName
Microsoft-Windows-Subsystem-

Linux

Restart when prompted and then open the **Microsoft Store**. Search for
Ubuntu and install the appropriate version. After launching the
**Ubuntu** program, wait for the installation to complete, then setup a
username and password as desired. Run the following commands
(interacting as directed):

$ sudo apt-get update

$ sudo apt-get install python3-pip

$ sudo apt-get install python3-opencv

$ pip3 install datetime glob3 IPython joblib pandas psutil matplotlib
pillow

ray scipy sobol sobol\_seq natsort multiprocess ray scikit-image sklearn
tqdm

Place the SLADS-\# relkease folder onto the Windows desktop. Then back
inside of **Ubuntu**, enter the following to move into the SLADS folder
replacing **username** and **versionNum** as appropriate:

$ cd /mnt/c/Users/**username**/Desktop/SLADS-**versionNum**

The configuration: CONFIG.py may be edited through Windows with the
editor of your choice, after which SLADS may be run using:

$ python3 SLADS.py

\#====================================================================

**TRAINING/TESTING PROCEDURE**

\#====================================================================

**<span class="underline">PRE-PROCESSING</span>**

Assuming the desired equipment intended for end application of SLADS,
outputs Thermo-Finnigan .RAW files, the data will need to be
pre-processed prior to training a SLADS model. Each data sample must be
aligned between rows using a simple linear interpolation scheme
according to the unique acquisition times. Each individual mz range of
interest must then be exported as a .csv file. For example, given a
singular sample: **sampleName**, with a number of rows: **numRows**
(corresponding to the number of acquired .RAW files, or number intended
for acquisition), and a set of mz ranges: **lowMZ1**-**highMZ1,
lowMZ2**-**highMZ2**, etc., a folder should be created which contains
files named according to the convention:

mz**lowMZ**\_**highMZ**fn**sampleName**ns**numRows**.csv

Please note that at this time in development, each mz value should be
specified with exactly 8 characters. Therefore, the following sample
parameters:

> **sampleName**: Slide1-Wnt-3
> 
> **numRows**: 72
> 
> **lowMZ1**: 454.8740
> 
> **highMZ1**: 454.8922
> 
> **lowMZ2**: 454.8844
> 
> **highMZ2**: 454.9026

should yield the following folder hierarchy:

> \-------\> Slide1-Wnt-3
> 
> |-------\> mz454.8740\_454.8922fnSlide1-Wnt-3ns72.csv
> 
> |-------\> mz454.8844\_454.9026fnSlide1-Wnt-3ns72.csv

Each of these folders may then be placed either into ./INPUT/TEST, or
./INPUT/TRAIN as desired. An example of the desired multichannel sample
input format is included in the ./EXAMPLE/ folder.

**<span class="underline">CONFIGURATION</span>**

**Warning**: Several of the parameters listed may either not implemented
at this time, or have been disabled; if this is the case it will be
noted in the comments within ./CONFIG.py.

All critical parameters for SLADS may be altered in ./CONFIG.py where:

**L0**: Specifies the overall program function(s) to be performed

**trainingModel**: Should a new SLADS model be generated

\- Uses data from ./INPUT/TRAIN/

**testingModel**: Should a SLADS model be evaluated

\- Uses data from ./INPUT/TEST/

**LOOCV**: Should Leave-One-Out Cross Validation be performed

\- Disables testingModel by default, uses data from ./INPUT/TRAIN/

**impModel**: Is a SLADS model being physically implemented

\- Uses data from ./INPUT/IMP/

**L1**: Specifies general model training parameters

**windowSize**: How should the sample be split for distortion
calculations

**stopPerc**: What percentage of pixels should be measured before
termination

**L2**: Specifies variables that should not typically be changed

**measurementPercs**: What sampling percentages should be used during
training

**cValues**: What possible c values should be tested for distortion
calculations

**animationGen**: Should animations be generated for
testing/implementation

**imageType**: Do the samples contain discrete (binary), or continuous
values

**findStopThresh**: Should a stopping threshold be automatically
determined

**percOfRD**: What percentage of RD should be expected; limits training
cost

**num**\_**threads**: How many CPU threads should be employed; automatic

**consoleRunning**: Is the program running in a console/terminal, or
notebook

**<span class="underline">RUN</span>**

After configuration, to run the program perform the following command in
the root directory:

$ python3 ./SLADS

**<span class="underline">RESULTS</span>**

All results will be placed in ./RESULTS/ (in the case of testing, at the
conclusion of a sample's scan) as follows:

**TRAIN**: Training results

**bestC.npy**: Determined optimal c value determined in training

**bestTheta.npy**: Best model corresponding with the determined best c
value

**cValues.npy**: List of possible c values that the best c value was
chosen from

**Images**: Empty directory for debug use; observation of training
convergence

**trainedModels.npy**: Trained models corresponding to each possible c
value

**TEST**: Testing results

**Animations**: Resultant visualizations and multimedia for test samples

**TEST\_SAMPLE\_1**: Folder of frames for a sample's scan

**TEST\_SAMPLE\_2**: Folder of frames for a sample's scan

**Videos**: Videos of final scan progression

**TEST\_SAMPLE\_1.av**i: Video of scan for sample

**TEST\_SAMPLE\_2.avi**: Video of scan for sample

**dataPrintout.csv**: Averaged final test results

**mzResults**: Final measurement results at specified mz ranges

**TEST\_SAMPLE\_1**: Folder of final measurements at mz ranges

**TEST\_SAMPLE\_2**: Folder of final measurements at mz Ranges

**testingAverageSSIM\_Percentage.csv**: Average SSIM progression results

**testingAverageSSIM\_Percentage.png**: Visualized average SSIM
progression

\#====================================================================

**OPERATIONAL PROCEDURE**

\#====================================================================

**Warning**: This section’s procedure has not been implemented at this
time. Below is a brief proposal of how SLADS may be easily integrated
with physical scanning equipment

**Note**: In order to use a SLADS model in a physical implementation,
the files resultant from the training procedure must be located within
'./RESULTS/TRAIN\_RESULTS/', particularly: bestC.npy and bestTheta.npy.

Prior to engaging the physical equipment, run SLADS with the
**impModel** variable enabled in the configuration file. All other
testing and training flags within **Parameters: L0**, should be
disabled. The program will then wait for a file: **LOCK** to be placed
within the ./INPUT/IMP/ folder; which when it appears will trigger the
program to read in any data saved into the same folder and produce a set
of points to scan, saved in a file: **UNLOCK**. SLADS will delete the
**LOCK** folder then, signalling the equipment that point selections
have been made and in preparation for the next acquisition iteration. As
with the training and testing datasets, it is expected that the data
will be given to SLADS in .csv files for each of the specified mz ranges
in accordance with the format mentioned in the **TRAINING/TESTING
PROCEDURE** section. When SLADS has reached its termination criteria it
will produce a different file: **DONE**, instead of: **UNLOCK**, to
signal the equipment that scanning has concluded.

\#====================================================================

**FAQ**

\#====================================================================
