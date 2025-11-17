# PriorMCL
Code for "Prior-driven Multimodal Contrastive Learning with Fourier Cross-attention to Detect Aortic Dissection in 3D Non-Contrast CT"

 
## 🔍 Description

PriorMCL is an anatomy-prior-driven multi-modal contrastive learning framework. Its core innovations include: adopting a three-encoder network architecture combined with a Fourier-enhanced cross-attention mechanism to capture spatial anatomical features and frequency-domain patterns simultaneously; focusing on relevant anatomical regions through aortic segmentation and straightening preprocessing; and using only NC-CT during fine-tuning and inference.


## Getting Started
### Installation
```
pip install -r requirements.txt
```

### Step 1: Data Preprocessing for Aortic Straightening

Step 1.1 Segment the whole aorta using TotalSegmentator <https://github.com/wasserth/TotalSegmentator>
Step 1.2 Straighten the aorta using the 3D Slicer script, which is located in the `Preprocess` folder. 



### Step 2: Model Pretraining
Pretraining Entry File: `Pretrain/run_newsets_fftloss_attention.py`
> You can modify parameters (e.g., batch size, epochs) and dataset in `config_newsets_fftloss_attention.yaml` before running.



### Step3: Inference for AD Detection and Lumen Segmentation
AD detection:
```
python run_detect.py
```
Lumen Segmentation:
```
python run_seg.py
```

## Datasets
Prepare the dataset in the following format:
```
{
  {
  'NC-CT',
  'CE-CT',
  'Text',
  'label',
  'mask',
  },
  {
  ...
  },
...
}
```
When inference, only the NC-CT is required.

### Other Code
The other code is coming soon. We are currently working on tidying up the code to improve its readability and maintainability.


