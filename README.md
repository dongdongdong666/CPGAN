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
- Download train2014-text.zip from [here](https://drive.google.com/file/d/1CuW5ognTSkNbyx9TWoUFrgwqxZNk1cl0/view?usp=sharing) and unzip it to data/coco/text


