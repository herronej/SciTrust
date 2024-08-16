# SciTrust: Evaluating the Trustworthiness of Large Language Models for Science


Table of Contents
=================

* [Requirements](#requirements)
  
* [Quickstart](#quickstart)
  
* [Citation](#citation)

## Requirements

### Environment 

* Python 3.12

### Environment Setup Instructions 

To set up the environment for this repository, please follow the steps below:

Step 1: Create and activate a Conda environment 

```bash
conda create -n scitrust python

conda activate scitrust
```

Step 2: Install PyTorch with CUDA

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Step 3: Install SciTrust and Python dependencies

```
pip install -e .
```

### Data & Models Setup

1. Create ```model``` folder in base directory
2. Download FORGE-L and Darwin-7b
3. Extract both models in ```model``` 

### Data and Code Description

The project data includes the following components:


## Quickstart 

Activate environment
```bash
conda activate scitrust
```

### Running SciTrust


To run inference use the ```scitrust-run``` command: 

```
scitrust-run --perspective <trustworthiness-perspective> --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>
```

#### Trustworthiness Perspectives

#### Candidate Models


#### Examples by Perspective 

##### Truthfulness

###### Misinformation

###### Logical Reasoning

###### Hallucination

###### Sycophancy

##### Adversarial Robustness
Coming soon

##### Scientific Ethics
Coming coon

# Citation
Coming soon
