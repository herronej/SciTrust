import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm.auto import tqdm
from datasets import load_dataset
import pandas as pd
from sci_datasets import SciQDataset, GPQADataset, ARCDataset, HendrycksDataset, OpenBookQADataset, SciEthicsDataset, AdvDataset, QADataset
from logi_datasets import LogicInferenceDataset, ReClorDataset, LogiQADataset
import argparse
import time

def get_dataset(dimension, dataset_name, k=0, split=None):

    if dimension == 'truthfulness':
        if dataset_name == 'SciQ':
	        dataset = SciQDataset(k=k, split=split)

        elif dataset_name == 'GPQA':
    	    dataset = GPQADataset(k=k,  split=split)  #load_dataset("Idavidrein/gpqa", "gpqa_main")

        elif dataset_name == 'ARC-E':
            dataset = ARCDataset("Easy", k=k, split=split)

        elif dataset_name == 'ARC-C':
            dataset = ARCDataset("Challenge", k=k, split=split)

        elif dataset_name == 'HT-CC':
        	dataset = HendrycksDataset("CC", k=k, split=split)

        elif dataset_name == 'HT-CCS':
        	dataset = HendrycksDataset("CCS", k=k, split=split)

        elif dataset_name == 'HT-CM':
        	dataset = HendrycksDataset("CM", k=k, split=split)

        elif dataset_name == 'HT-CB':
        	dataset = HendrycksDataset("CB", k=k, split=split)

        elif dataset_name == 'HT-CP':
        	dataset = HendrycksDataset("CP", k=k, split=split)

        elif dataset_name == 'HT-S':
        	dataset = HendrycksDataset("S", k=k, split=split)

        elif dataset_name == 'OBQA':
            dataset = OpenBookQADataset(k=k, split=split)

        elif dataset_name == "ChemistryQA":
            dataset = QADataset("data/truthfulness_openended/chemistry_qa_chatgpt-4o.jsonl", split=split)

        elif dataset_name == "PhysicsQA":
            dataset = QADataset("data/truthfulness_openended/physics_qa_chatgpt-4o.jsonl", split=split)

        elif dataset_name == "BiologyQA":
            dataset = QADataset("data/truthfulness_openended/biology_qa_chatgpt-4o.jsonl", split=split)

        elif dataset_name == "ComputerScienceQA":
            dataset = QADataset("data/truthfulness_openended/computer_science_qa_chatgpt-4o.jsonl", split=split)

        else:
            print("Dataset {} not supported. Supported datasets: SciQ, GPQA, ARC-E, ARC-C, OBQA".format(dataset_name))

    elif dimension == "sycophancy":
        if dataset_name == 'SciQ':
            dataset = SciQDataset(k=k, split=split, sycophancy=True)

        elif dataset_name == 'ARC-E':
            dataset = ARCDataset("Easy", k=k, split=split, sycophancy=True)

        elif dataset_name == 'ARC-C':
            dataset = ARCDataset("Challenge", k=k, split=split, sycophancy=True)

        elif dataset_name == 'GPQA':
            dataset = GPQADataset(k=k, split=split, sycophancy=True)

        else:
            print("Dataset {} not supported. Supported datasets: SciQ, ARC-E, ARC-C, OBQA".format(dataset_name))
            exit()

    elif 'adv_robustness' in dimension:

        if dimension == 'adv_robustness_textfooler':
            if dataset_name == "SciQ":
                dataset = AdvDataset("data/adv_data/llama2-7b_textfooler_SciQ_0_0.json", split=split)
            elif dataset_name == "ARC-C":
                dataset = AdvDataset("data/adv_data/llama2-7b_textfooler_ARC-C_0_0.json", split=split)
            elif dataset_name == "GPQA":
                dataset = AdvDataset("data/adv_data/llama2-7b_textfooler_GPQA_0_0.json", split=split)
            else: 
                print("Dataset {} not supported. Supported datasets: SciQ, ARC-E, ARC-C, and GPQA.".format(dataset_name))
                exit()

        elif dimension == 'adv_robustness_textbugger':
            if dataset_name == "SciQ":
                dataset = AdvDataset("data/adv_data/llama2-7b_textbugger_SciQ_0_0.json", split=split)
            elif dataset_name == "ARC-C":
                dataset = AdvDataset("data/adv_data/llama2-7b_textbugger_ARC-C_0_0.json", split=split)
            elif dataset_name == "GPQA":
                dataset = AdvDataset("data/adv_data/llama2-7b_textbugger_GPQA_0_0.json", split=split)
            else:
                print("Dataset {} not supported. Supported datasets: SciQ, ARC-E, ARC-C, and GPQA.".format(dataset_name))
                exit()
        elif dimension == 'adv_robustness_stresstest':
            if dataset_name == "SciQ":
                dataset = AdvDataset("data/adv_data/llama2-7b_textbugger_SciQ_0_0.json", split=split)
            elif dataset_name == "ARC-C":
                dataset = AdvDataset("data/adv_data/llama2-7b_textbugger_ARC-C_0_0.json", split=split)
            elif dataset_name == "GPQA":
                dataset = AdvDataset("data/adv_data/llama2-7b_stresstest_GPQA_0_0.json", split=split)
            else:
                print("Dataset {} not supported. Supported datasets: SciQ, ARC-E, ARC-C, and OBQA.".format(dataset_name))
                exit()
        elif 'adv_robustness_open_ended' in dimension:
            attack_num = dimension.split("_")[-1]
            if int(attack_num) > 11:
                print("Attack not supported.")
                exit()
            if dataset_name == "ChemistryQA":
                dataset = QADataset("data/adv_data/chemistry_qa_chatgpt-4o_500_adv_{}.jsonl".format(attack_num), split=split)
            elif dataset_name == "ComputerScienceQA":
                dataset = QADataset("data/adv_data/computer_science_qa_chatgpt-4o_500_adv_{}.jsonl".format(attack_num), split=split)
            elif dataset_name == "BiologyQA":
                dataset = QADataset("data/adv_data/biology_qa_chatgpt-4o_500_adv_{}.jsonl".format(attack_num), split=split)
            elif dataset_name == "PhysicsQA":
                dataset = QADataset("data/adv_data/physics_qa_chatgpt-4o_500_adv_{}.jsonl".format(attack_num), split=split)
            elif dataset_name == "LogicInference":
                dataset = QADataset("data/adv_data/logicinference_oa_chatgpt-4o_500_adv_{}.jsonl".format(attack_num), split=split)
            else:
                print("Dataset {} not supported. Supported datasets: ChemistryQA, ComputerScienceQA, BiologyQA, PhysicsQA, and LogicInference.".format(dataset_name))
                exit()
        else:
            print("Attack not supported. Supported attacks: textbugger, textfooler, stresstest.")
            exit()

    elif "scientific_ethics" in dimension:

        if dimension == "scientific_ethics_full":
            dataset = SciEthicsDataset(k=k)

        elif dimension == "scientific_ethics_ai":
            dataset = SciEthicsDataset(subset='AI', k=k)

        elif dataset_name == 'scientific_ethics_animal_testing':
            dataset = SciEthicsDataset(subset='AT', k=k)

        elif dataset_name == 'scientific_ethics_bias_objectivity':
            dataset = SciEthicsDataset(subset='BO', k=k)

        elif dataset_name == 'scientific_ethics_data_privacy':
            dataset = SciEthicsDataset(subset='DP', k=k)

        elif dataset_name == 'scientific_ethics_dual_use_research':
            dataset = SciEthicsDataset(subset='DU', k=k)

        elif dataset_name == 'scientific_ethics_environmental_impact':
            dataset = SciEthicsDataset(subset='EI', k=k)

        elif dataset_name == 'scientific_ethics_human_subjects':
            dataset = SciEthicsDataset(subset='HS', k=k)
        else:
            print("Dataset {} not supported. Supported datasets: scientific_ethics_full, scientific_ethics_ai, scientific_ethics_animal_testing, scientific_ethics_bias_objectivity, scientific_ethics_data_privacy, scientific_ethics_dual_use_research, scientific_ethics_environmental_impact, scientific_ethics_human_subjects".format(dataset_name))
            exit()

    elif dimension == "logical_reasoning":

        if dataset_name == 'LogicInference':
            dataset = LogicInferenceDataset(k=k, split=split)

        elif dataset_name == 'ReClor':
            dataset = ReClorDataset(k=k)

        elif dataset_name == 'LogiQA':
            dataset = LogiQADataset(k=k)

        else:
            print("Dataset {} not supported. Supported datasets: LogicInference, ReClor, and LogiQA.".format(dataset_name))

    elif dimension == "hallucination":

        if dataset_name == "ChemistryQA":
            dataset = QADataset("data/truthfulness_openended/chemistry_qa_chatgpt-4o.jsonl", split=split)

        elif dataset_name == "PhysicsQA":
            dataset = QADataset("data/truthfulness_openended/physics_qa_chatgpt-4o.jsonl", split=split)

        elif dataset_name == "BiologyQA":
            dataset = QADataset("data/truthfulness_openended/biology_qa_chatgpt-4o.jsonl", split=split)

        elif dataset_name == "ComputerScienceQA":
            dataset = QADataset("data/truthfulness_openended/computer_science_qa_chatgpt-4o.jsonl", split=split)

        else:
            print("Dataset {} not supported. Supported datasets: ChemistryQA, PhysicsQA, BiologyQA, and ComputerScienceQA.".format(dataset_name))

    else:
        print("Error: Please enter a valid trustworthiness dimension. Supported options: truthfulness, sycophancy, adv_robustness, scientific_ethics, logical_reasoning, hallucination.")
        exit()

    return dataset

def append_record(record, filename):

    with open(filename, 'a') as f:
        json.dump(record, f)
        f.write('\n')

def generate_samples(batch, tokenizer, model, device, openended=False):
    gen_text_samples_batch = []
    #print('len(batch)', len(batch))
    if openended:
        max_new_tokens=300
    else:
        max_new_tokens=3

    for d in zip(batch[0], batch[1]):
        gen_text_samples = []
        for n in range(4):
            print('n', n)
            input_ids = tokenizer(d[0], return_tensors="pt").input_ids.to(device)
            #print('tokenizer.model_max_length', tokenizer.model_max_length)
            #print(input_ids.shape)
            gen_start_time = time.time()
            gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.7, max_new_tokens=max_new_tokens)
            gen_end_time = time.time()
            print("Generation Time", gen_end_time-gen_start_time)
            gen_text = tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:])[0]
            gen_text_samples.append(gen_text)
        sample_data = [d[0], d[1]] + gen_text_samples
        print('sample_data', sample_data)
        gen_text_samples_batch.append(sample_data)
    #print(len(gen_text_samples_batch))
    return gen_text_samples_batch

def save_checkpoint(batch_idx, output_path, checkpoint_path):
    checkpoint = {
        'batch_idx': batch_idx,
        'output_path': output_path
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint['batch_idx'], checkpoint['output_path']
    else:
        return 0, None

