# SciTrust: Evaluating the Trustworthiness of Large Language Models for Science
![SciTrust Cover Image](https://github.com/herronej/SciTrust/blob/main/cover_image.png)

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

```
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
```
conda activate scitrust
```

### Running SciTrust


To run inference use the ```scitrust-run``` command: 

```
scitrust-run --perspective <trustworthiness-perspective> --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>
```

To get performance results use the ```scitrust-eval``` command: 

```
scitrust-eval --perspective <trustworthiness-perspective> --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>
```

#### Supported Models w/ Flags

- Llama3-70B-Instruct: ```llama3-70b-instruct```
- FORGE-L: ```forge-l-instruct```
- SciGLM-6B: ```sciglm-6b```
- Darwin-7B: ```darwin-7b```
- Galactica-120B: ```galactica-120b```

#### Supported Datasets w/ Flags

- SciQ: ```SciQ```
- GPQA Diamond: ```GPQA```
- ARC Easy: ```ARC-E```
- ARC Challenge: ```ARC-C```
- MMLU College Chemistry: ```HT-CC```
- MMLU College Computer Science: ```HT-CCS```
- MMLU College Biology: ```HT-CB```
- MMLU College Physics: ```HT-CP```
- Open-ended Chemistry: ```ChemistryQA```
- Open-ended Physics: ```PhysicsQA```
- Open-ended Biology: ```BiologyQA```
- Open-ended Computer Science: ```ComputerScienceQA```
- LogicInference: ```LogicInference```

#### Examples by Perspective 

##### Truthfulness

###### Misinformation

```scitrust-run --perspective 'truthfulness' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```
```scitrust-eval --perspective 'truthfulness' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

###### Logical Reasoning

```scitrust-run --perspective 'logical_reasoning' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```
```scitrust-eval --perspective 'logical_reasoning' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

###### Hallucination

```scitrust-run --perspective 'hallucination' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```
```scitrust-eval --perspective 'hallucination' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```


###### Sycophancy

```scitrust-run --perspective 'sycophancy' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```
```scitrust-eval --perspective 'sycophancy' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```


##### Adversarial Robustness
Coming soon

##### Scientific Ethics
Coming coon

# Citation
Coming soon
