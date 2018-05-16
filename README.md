# *Brain2movie*: : Generating Image Sequence from Brain Signal via Temporal Detection and Generative Adversarial Network
This code is for Brain2movie: Generating Image Sequence from Brain Signal via Temporal Detection and Generative Adversarial Network

### Acknowledgement
This code is based on the belowing paper:

    Concetto Spampinato, Simone Palazzo, Isaak Kavasidis, Daniela Giordano, Nasim Souly and Mubarak Shah: 
    Deep Learning Human Mind for Automated Visual Classification, CVPR (Oral) 2017

### Contents
1. [Requirements: Hardware](#requirements-hardware)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Demo](#demo)

### Requirements: Hardware
* Linux Environment (Ubuntu)
* GPU with over 3GB (Trained with Nvidia GTX-1080Ti)

### Prerequisites
* Pytorch (Implemented with 0.4.0)
* Tensorflow
* CUDA 8.0 or Higher
* Numpy
* Matplotlib (Optional)

### Installation
Clone the code
```shell
git clone https://github.com/thecho7/Brain2movie.git
```
Dataset Download (TBI)

### Demo
1. (Beta) - **Classification & Image Generation**
```shell
python thecho7.py
```
The results are saved in result folder with data information.
ex) If the analyzed sequence is from 30 to 300, the folder will be created 'result/30_300/'.

The saved results are
1. Classification result
2. Generated images

2. (Beta) - **Detection**
```shell
python temporal_proposal.py
```


