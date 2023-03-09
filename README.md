# Video_Feature_Extractor

## Overview

This repository contains the code for extracting features from videos using I3D, slowfast and other supported backbones.


## Data Preparation

### Thumos14

You can download the raw videos from [here](https://www.crcv.ucf.edu/data/THUMOS14/). The evaluation set is used for training, and the test set is used for testing. The annotations are available [here](https://www.crcv.ucf.edu/data/THUMOS14/).

You can use the given annotation files by me to extract the features, for which I removed some wrong annotations (history problem). You can see them in the datasets folder.

### ActivityNet

You can download the raw videos from [here](http://activity-net.org/download.html). Not supported yet.


### MultiThumos

This dataset is an extension of the original Thumos14 dataset. We have prepared the annotations for you


### Charades

This dataset is a large scale dataset for action recognition. You can download the raw videos from [here](https://allenai.org/plato/charades/). We use the GRB frames for feature extractions.

### Other datasets

If you want to use other datasets, you can prepare the annotations in the same format as the given ones. Then, you can write your own dataset class in the datasets folder.

## Extraction

We use 10 fps, every 4 frames for a feature for THUMOS14. You can change it in the config file.
Note that we use windows for feature extraction. The default window size is 64 for thumos14. You can change it in the config file.

