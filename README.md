# SciTrust 2.0: Evaluating the Trustworthiness of Large Language Models for Science

Table of Contents
=================

* [Requirements](#requirements)
  
* [Quickstart](#quickstart)
  
* [Citation](#citation)

## Requirements

### Environment 

* Python 3.12 (on Frontier, provided by the `cray-python` module)
* GPU runtime: AMD ROCm (Frontier: `rocm/7.2.0`) — or CUDA on NVIDIA systems

### Environment Setup Instructions 

To set up the environment on **OLCF Frontier** (AMD MI250X / ROCm) with a Python virtual environment, run every step from the repository root.

**Step 1 — Load the required modules**

```bash
module load cray-python      # provides python + venv
module load git-lfs          # for model/dataset files tracked with Git LFS
module load rocm/7.2.0       # AMD GPU runtime for PyTorch
```

**Step 2 — Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Step 3 — Configure the OLCF proxy**

Frontier nodes have no direct outbound internet; the proxy is required for `pip` and for downloading models/datasets from the Hugging Face Hub.

```bash
export https_proxy='http://proxy.ccs.ornl.gov:3128'
export http_proxy='http://proxy.ccs.ornl.gov:3128'
export no_proxy='localhost,127.0.0.1,0.0.0.0'
```

**Step 4 — Install PyTorch (ROCm build)**

Install the PyTorch wheel matching the loaded `rocm` module *before* the other requirements, so pip does not pull the default CUDA wheel. Browse the available builds at https://download.pytorch.org/whl/ and choose the `rocmX.Y` closest to your module.

```bash
pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

**Step 5 — Install SciTrust and its Python dependencies**

```bash
pip install -r requirements.txt
pip install -e .
```

### Data & Models Setup

1. Create ```models``` folder in base directory
2. Download [FORGE-L](https://github.com/at-aaims/forge) and [Darwin1.5-7b](https://github.com/MasterAI-EAM/Darwin)
3. Extract both models in ```models``` (FORGE-L under ```models/forge-l/```)

### Data and Code Description

The project data includes the following components:


## Quickstart 

### Before each session

In a fresh shell, re-load the modules, activate the environment, and set the proxy (from the repository root):

```bash
module load cray-python git-lfs rocm/7.2.0
source .venv/bin/activate
export https_proxy='http://proxy.ccs.ornl.gov:3128'
export http_proxy='http://proxy.ccs.ornl.gov:3128'
export no_proxy='localhost,127.0.0.1,0.0.0.0'
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

### Running on a Frontier compute node

Local models (e.g. FORGE-L, Darwin) require a GPU, so run them on a compute node rather than a login node. With your environment active (modules, `.venv`, and proxy — see [Before each session](#before-each-session)), request an interactive allocation:

```bash
salloc -A stf218 -N 1 -t 00:20:00 -p batch
```

Adjust `-A` (project account), `-N` (node count), and `-t` (walltime) to suit your run. Once the allocation is granted, launch inference with `srun` from the repository root:

```bash
HOME=$PWD srun -n1 scitrust-run --perspective truthfulness_misinformation --dataset SciQ -k 0 --model forge-l-instruct
```

- Run from the repository root so the relative `models/forge-l` path resolves.
- `HOME=$PWD` redirects the Hugging Face / model cache into the project directory instead of your (quota-limited) home area.
- API-only models (`gpt-o4-mini`, `claude-sonnet-3.7`) need no GPU and can run on a login node, but still require the proxy and the relevant API key.

#### Supported Models w/ Flags

- GPT-o4-Mini: ```gpt-o4-mini```
- Claude Sonnet 3.7: ```claude-sonnet-3.7```
- Llama4-Scout: ```llama4-scout```
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

```scitrust-run --perspective 'truthfulness_hallucination' --dataset <dataset-name> --model <model-name>```

```scitrust-eval --perspective 'truthfulness_hallucination' --dataset <dataset-name> --model <model-name>```


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
```scitrust-run --perspective 'adv_robustness_open_ended_character-level' --dataset <dataset-name> --model <model-name>```
```scitrust-eval --perspective 'adv_robustness_open_ended_character-level' --dataset <dataset-name> --model <model-name>```

```scitrust-run --perspective 'adv_robustness_open_ended_word-level' --dataset <dataset-name> --model <model-name>```
```scitrust-eval --perspective 'adv_robustness_open_ended_word-level' --dataset <dataset-name> --model <model-name>```

```scitrust-run --perspective 'adv_robustness_open_ended_word-level' --dataset <dataset-name> --model <model-name>```
```scitrust-eval --perspective 'adv_robustness_open_ended_word-level' --dataset <dataset-name> --model <model-name>```

###### Supported Datasets w/ Flags
- Open-ended Chemistry: ```ChemistryQA```
- Open-ended Physics: ```PhysicsQA```
- Open-ended Biology: ```BiologyQA```
- Open-ended Computer Science: ```ComputerScienceQA```

##### Safety

```scitrust-run --perspective 'safety' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```
```scitrust-eval --perspective 'safety' --dataset <dataset-name> -k <number-of-demonstrations> --model <model-name>```

###### Supported Datasets w/ Flags
- WMDP Biology: ```WMDP-BIO```
- WMDP Chemistry: ```WMDP-CHEM```
- WMDP Cyber: ```WMDP-CYBER```
- HarmBench Chemistry and Biology Contexts: ```HarmBench-CHEM-BIO```
- HarmBench Cybercrime and Intrusion Contexts: ```HarmBench-CYBERCRIME-INTRUSION```

##### Scientific Ethics

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
## SciTrust 1.0 
Emily Herron, Junqi Yin, and Feiyi Wang. “SciTrust: Evaluating the Trustworthiness of Large
372 Language Models for Science”. In: Proceedings of the SC ’24 Workshops of the International
373 Conference on High Performance Computing, Network, Storage, and Analysis. SC-W ’24.
374 Atlanta, GA, USA: IEEE Press, 2025, pp. 72–78. ISBN: 9798350355543. DOI: 10.1109/
375 SCW63240.2024.00017. URL: https://doi.org/10.1109/SCW63240.2024.00017.

## SciTrust 2.0
Coming soon
