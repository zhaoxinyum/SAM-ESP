# Contour Field based Elliptical Shape Prior for the Segment Anything Model

## Project Overview
This paper introduces a novel elliptical shape prior for the Segment Anything Model based on contour field ([arXiv:2504.12556](https://arxiv.org/abs/2504.12556 "arXiv:2504.12556")).

## Installation Guide

### Requirements
The code requires python>=3.10, as well as torch>=2.5.1 and torchvision>=0.20.1
We use '[sam_vit_b_01ec64](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)' as our basic model. Please download this weight and then place it in the "parameterfault" folder.
### Installation Steps

```plaintext
# Clone the repository
git clone https://github.com/zhaoxinyum/SAM-ESP.git
cd SAM-ESP

# Create virtual environment and activate (optional but recommended)

# Install dependencies
pip install -r requirements.txt
```
### Code Introduction
The code of the SAM-ESP (SAM added the elliptical shape constraint module) structure is : `SAM_ESP.py`
The code of the esp module is:`esp_module.py`

###Training 
#### training the SAM-ESP model
```plaintext
python train.py --dataname YourData 
```
#### finetune the SAM
```plaintext
python SAMtrain.py --dataname YourData
```

### Evaluation
Please enter the correct path of the corresponding checkpoint when running
#### evaluate the SAM-ESP model
```plaintext
python Eva_Model.py --data_name YourData
```
#### evaluate the SAM-finetune model
```plaintext
python SAM_eval.py
```
The test results will be saved in `{yourcheckpointfolder/prediction}` folder

[//]: # (##Acknowledgments)

[//]: # (This repository is built upon ideas and implementations inspired by the paper "Contour flow constraint: Preserving)

[//]: # (global shape similarity for deep learning-based image segmentation". I would like to thank the authors)
