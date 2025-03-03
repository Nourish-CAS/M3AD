# M³AD: Multi Scale, Multi Mode, and Multi Process Time Series Anomaly Detection Framework
This repository contains the implementation of 'Revisiting Time Series Anomaly Detection: A Multi-Scale,
 Multi-Mode and Multi-Process Real-Time Framework'. We propose the M³AD framework to achieve effective, explainable, and robust time series anomaly detection.
## M³AD model architecture

<div align=center>
<img src=".\model.png" width="90%">
</div>

## Dataset
1.You can download the entire dataset from this link ([Click here to download the dataset](https://github.com/pseudo-Skye/Data-Smith/blob/master/TSAD%20Dataset/cleaned_dataset/cleaned_dataset.zip)).

2.run preprocess_data.py.

## Overview
Run train.py to train M3AD over the whole dataset.Actually, each of the 250 datasets has its own characteristics. We will compile a detailed training report for each dataset later. We recommend that you set 150 epochs for each dataset. Although this may result in a relatively long training time, we hope that you can achieve the accuracy reported in the paper as much as possible.

After the training of each dataset is completed, inference will be carried out immediately. Here, the inference is strictly performed in sequence according to the sliding window (corresponding to M³AD (All - process) in Table 5 of the paper). Later, we will provide parallel inference from a full-view perspective (corresponding to M³AD (Parallel Global Windows) in Table 5).

Please note that in Table 5, regarding the running time of MERLIN, we used (MinL = 50, MaxL = 300) for the search in UCR001 - UCR238, and enabled multi - threading. It took 660 minutes. However, the subsequent datasets were too large, and the time complexity of this algorithm increased sharply. So we used (MinL = 5, MaxL = 30) for UCR239 - UCR250, which took 594 minutes. Therefore, the total time spent was 1254 minutes. However, when we performed the MERLIN process after filtering out the abnormal regions, most of the searches were within (MinL = 5, MaxL = 300), and it only took 7.3 minutes.
