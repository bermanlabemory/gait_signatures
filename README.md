# gait_signatures

#### This is code related to the paper: Taniel S. Winner, Michael C. Rosenberg, Trisha M. Kesar, Lena H. Ting, Gordon J. Berman. Discovering individual-specific ‘gait signatures’ from data-driven models of neuromechanical dynamics. Journal. 202X; TBD. 

## Data

#### PareticvsNonP_RNNData.csv: Data set containing sagittal plane joint angles of 5 able-bodied and 7 stroke survivors as they walked on a treadmill at 6 different speeds each for 15 seconds. (72 trials, 1500 sample/trial). The dataset was collected at Emory Rehab Hospital Motion Analyses Lab (PI Kesar). 

#### **Subjectslabels.csv**: Data set containing a list of subject identifiers corresponding to each trial. 

#### **Speedlabels.mat**: Data set containing a list of treadmill speeds (cm/s) corresponding to each trial. 


## Python Code  

#### **Gait Signatures Script 1: Train Model Architectures.ipynb** 
#### **Gait Signatures Script 2: Extract Internal Parameters.ipynb** 
#### **Gait Signatures Script 3: Generate Gait Signatures.ipynb** 

#### Code was executed on Google Colab. The specific Collab software versions tested are Python 3.8.16 and Tensorflow 2.9.2. on Google Colab’s standard GPU with high-RAM runtime (54.8 gigabytes). 

## Mathematical tools 

#### We make use of a collection of mathematical tools used in the Biologically Inspired Robotics and Dynamical Systems (BIRDS) lab at the University of Michigan. Most of these tools were either coded or invented to a large degree by the lab's founder, Shai Revzen. 

#### **fourierseries.py**: a general purpose FourierSeries class 
#### **phaser.py**: the Phaser algorithm from doi:10.1103/PhysRevE.78.051907 
#### **jac.py**: Jacobian functor (maps a function to its Jacobian) 
#### **odedp5.py**: order 5 ODE integrator with special facilities for hybrid systems 
#### **sde.py**: order 3 Stratonovic SDE integrator 
#### **fastode**: aggressively optimized version of the odedp5 integrator for very fast operation. Requires inner ODE function to be in C  
#### **util.py**, shaiutil.py and shai_util.py: helper functions 
