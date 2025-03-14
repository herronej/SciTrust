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

- OpenAIs o1: ```gpt-o1```
- Llama3.3-70B-Instruct: ```llama3.3-70b-instruct```
- FORGE-L: ```forge-l-instruct```
- SciGLM-6B: ```sciglm-6b```
- Darwin1.5-7B: ```darwin1.5-7b```
- Galactica-120B: ```galactica-120b```


#### Examples by Perspective 

##### Truthfulness

###### Misinformation

```scitrust-run --perspective 'truthfulness_misinformation' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

```scitrust-eval --perspective 'truthfulness_misinformation' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```


###### Supported Datasets w/ Flags

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


###### Logical Reasoning

```scitrust-run --perspective 'truthfulness_logical_reasoning' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

```scitrust-eval --perspective 'truthfulness_logical_reasoning' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

###### Supported Datasets w/ Flags

- ReClor: ```ReClor```
- LogiQA: ```LogiQA```
- LogicInference: ```LogicInference```

###### Hallucination

```scitrust-run --perspective 'truthfulness_hallucination' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

```scitrust-eval --perspective 'truthfulness_hallucination' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```


###### Sycophancy

```scitrust-run --perspective 'truthfulness_sycophancy' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

```scitrust-eval --perspective 'truthfulness_sycophancy' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

#### Supported Datasets w/ Flags

- SciQ: ```SciQ```
- GPQA Diamond: ```GPQA```
- ARC Easy: ```ARC-E```
- ARC Challenge: ```ARC-C```

##### Adversarial Robustness

###### Multiple-Choice Datasets
```scitrust-run --perspective 'adv_robustness_textfooler' --dataset <dataset-name> --model <model-name>```

```scitrust-eval --perspective 'adv_robustness_textfooler' --dataset <dataset-name> --model <model-name>```

```scitrust-run --perspective 'adv_robustness_textbugger' --dataset <dataset-name> --model <model-name>```

```scitrust-eval --perspective 'adv_robustness_textbugger' --dataset <dataset-name> --model <model-name>```

```scitrust-run --perspective 'adv_robustness_stresstest' --dataset <dataset-name> --model <model-name>```

```scitrust-eval --perspective 'adv_robustness_stresstest' --dataset <dataset-name> --model <model-name>```

###### Supported Datasets w/ Flags
- SciQ: ```SciQ```
- GPQA Diamond: ```GPQA```
- ARC Challenge: ```ARC-C```


###### Open-ended Datasets
Coming Soon

##### Safety
Coming Soon

##### Scientific Ethics

New! February 2025

```scitrust-run --perspective 'scientific_ethics' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

```scitrust-eval --perspective 'scientific_ethics' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

###### Supported Datasets w/ Flags
- AI and Machine Learning: ```scientific_ethics_ai``` 
- Animal Testing: ```scientific_ethics_animal_testing```
- Bias and Objectivity: ```scientific_ethics_bias_objectivity```
- Data Privacy: ```scientific_ethics_data_privacy```
- Dual Use Research: ```scientific_ethics_dual_use_research```
- Environmental Impact: ```scientific_ethics_environmental_impact```
- Human Subjects: ```scientific_ethics_human_subjects```
- Genetic Modification: ```scientific_ethics_genetic_modification```

# Citations
Coming soon
