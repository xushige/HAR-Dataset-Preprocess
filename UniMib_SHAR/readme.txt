UniMiB SHAR: a new dataset for human
activity recognition using acceleration data
from smartphones

D. Micucci and M. Mobilio and P. Napoletano
-------------------------------------------------------------------------------
ABSTRACT: 
Smartphones, smartwatches, fitness trackers, and ad-hoc wearable devices are being increasingly used to monitor human activities. Data acquired by the hosted sensors are usually processed by machine-learning-based algorithms to classify human activities. The success of those algorithms mostly depends on the availability of training (labeled) data that, if made publicly available, would allow researchers to make objective comparisons between techniques. Nowadays, publicly available data sets are few, often contain samples from subjects with too similar characteristics, and very often lack of specific information so that is not possible to select subsets of samples according to specific criteria. In this article, we present a new smartphone accelerometer dataset designed for activity recognition. The dataset includes 11,771 activities performed by 30 subjects of ages ranging from 18 to 60 years. Activities are divided in 17 fine grained classes grouped in two coarse grained classes: 9 types of activities of daily living (ADL) and 8 types of falls. The dataset has been stored to include all the information useful to select samples according to different criteria, such as the type of ADL performed, the age, the gender, and so on. Finally, the dataset has been benchmarked with two different classifiers and with different configurations. The best results are achieved with k-NN classifying ADLs only, considering personalization, and with both windows of 51 and 151 samples.

-------------------------------------------------------------------------------
DATA:

The mat file that contains all the recorded data is "full_data.mat". 
Each row of the matrix contains the data of each subject annotated with the following fields:

"accelerometer data", "gender", "age", "height", "weight"

Each acceleration data contains the recorded activities: 

    'StandingUpFS'
    'StandingUpFL'
    'Walking'
    'Running'
    'GoingUpS'
    'Jumping'
    'GoingDownS'
    'LyingDownFS'
    'SittingDown'
    'FallingForw'
    'FallingRight'
    'FallingBack'
    'HittingObstacle'
    'FallingWithPS'
    'FallingBackSC'
    'Syncope'
    'FallingLeft'

For each activity, there are several trials (from 2 to 6) performed by each subjects.
For all the activities that have 2 trials, the first one has been recorded with the smartphone in the right pocket while the other has been recorded with the smartphone in the left pocket.
For all the activities that have 6 trials, the first three have been recorded with the smartphone in the right pocket while the others have been recorded with the smartphone in the left pocket.
Each activity record is made of 6 rows: the first three contain acceleration data along  x,y and z directions, the forth and fitfth row, the time instants and the sixth row the magnitudo of the raw signal. 

As additional materials, we make available the data used in the experiments presented in the paper.
The data are obtained by taking a window of 151 samples around a peak of the original signal higher than 1.5g with g being the gravitational acceleration. 
The data are separated by each experiment (please refer to the paper for further details):

AF-17: "acc_data.mat" and "acc_labels.mat"
AF-9: "adl_data.mat" and "adl_labels.mat"
AF-8: "fall_data.mat" and "fall_labels.mat"
AF-2: "two_classes_data.mat" and "two_classes_labels.mat"

the file "x_data.mat" contains for each row a set of three subsequent rows contains x,y,z accelerometer data.
The file "x_labels.mat" contains a row for each activity. Each row contains the label of the activity, the label of the subject who perfomed the activity and the number of trial.

The folder "split" contains the .mat of all the splits used for 5-fold cross-validation and 30-fold cross-validation.

The folder "results" contains the .mat of all the experiments presented in the paper.

All these files combined with the code included in the dataset allow to repeat the experiments presented in the paper.
-------------------------------------------------------------------------------
CODE

The code is written in Matlab and tested on a Ubuntu 14.04 machine with Matlab 2014b.
To repeat the experiments: open the matlab script "evall.m", change the variables "datapath", "splitpath" and "resultpath" in agreement with your local path. To repeat the same numbers of the paper, check if the original training/test splits are in the folder "./data/split/".

The results will be written in the folder "./data/results/"
-------------------------------------------------------------------------------
ACKNOWLEDGE:
If you use the dataset or the code, please cite this paper:

@article{micucci2017SHAR,
  title={UniMiB SHAR: a new dataset for human activity recognition using acceleration data from smartphones},
  author={Micucci, Daniela and Mobilio, Marco and Napoletano, Paolo},
  journal={arXiv preprint arXiv:1611.07688v2},
  year={2017}
}


