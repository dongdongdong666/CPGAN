# CPGAN
The method of text-to-image
Pytorch implementation for reproducing CPGAN results in the paper CPGAN: Full-Spectrum Content-Parsing Generative Adversarial Networks for Text-to-image Synthesis by Jiadong Liang, Wenjie Pei, Feng Lu.

<img src="model_structure.jpg" width="900px" height="280px"/>

## Getting Started
### Installation

- Create  anaconda virtual environment

```bash
conda create -n CPGAN python=2.7
```

- Install PyTorch and dependencies from http://pytorch.org

```bash
conda install pytorch torchvision cudatoolkit=10.1
```
- PIP Install

```bash
pip install python-dateutil, easydict, pandas, torchfile, nltk, scikit-image, h5py, pyyaml
```

- Clone this repo:

```bash
https://github.com/dongdongdong666/CPGAN.git
cd CPGAN
```
- Download train2014-text.zip from [here](https://drive.google.com/file/d/1UBgUHYWSmDD1Gnja2K7ZCVuQTLR89PAf/view?usp=sharing) and unzip it to data/coco/text/

- Download memory features for each word from [here](https://drive.google.com/file/d/145fBRWbqTdQUFFtoOwGhA9TW_ZVLeZVx/view?usp=sharing) and put the feature to memory/

- Download Inception Enocder [here](https://drive.google.com/file/d/1i3TW5mOsXaqZqzfSeHIBzuxb6CL4BjvO/view?usp=sharing) and put the feature to memory/, Generator [here](https://drive.google.com/file/d/1nirpy1jI5_sh_b_Mnbw-I3SSI7K-5UE_/view?usp=sharing) and put the feature to memory/, Text Encoder [here](https://drive.google.com/file/d/1JO7NQM4JOHRoABxUqMYEQPDvs_w2lTJ8/view?usp=sharing) and put the feature to memory/ and put these models to models/
