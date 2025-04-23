import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm.auto import tqdm
from datasets import load_dataset
import pandas as pd
from .sci_datasets import * #SciQDataset, GPQADataset, ARCDataset, HendrycksDataset, OpenBookQADataset, SciEthicsDataset, AdvDataset, QADataset, WMDPDataset
from .logi_datasets import * #LogicInferenceDataset, ReClorDataset, LogiQADataset
import argparse
import time
from openai import OpenAI
import anthropic
#from google import genai

def get_dataset(perspective, dataset_name, k=0, split=None, use_cot=False, from_file=''):

    if perspective == 'truthfulness_misinformation':
        if dataset_name == 'SciQ':
	        dataset = SciQDataset(k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'GPQA':
    	    dataset = GPQADataset(k=k,  split=split, use_cot=use_cot)

        elif dataset_name == 'ARC-E':
            dataset = ARCDataset("Easy", k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'ARC-C':
            dataset = ARCDataset("Challenge", k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'HT-CC':
        	dataset = HendrycksDataset("CC", k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'HT-CCS':
        	dataset = HendrycksDataset("CCS", k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'HT-CM':
        	dataset = HendrycksDataset("CM", k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'HT-CB':
        	dataset = HendrycksDataset("CB", k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'HT-CP':
        	dataset = HendrycksDataset("CP", k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'HT-S':
        	dataset = HendrycksDataset("S", k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'OBQA':
            dataset = OpenBookQADataset(k=k, split=split)

        elif dataset_name == "ChemistryQA":
            dataset = QADataset("scitrust_datasets/truthfulness_open_ended/Chemistry_qa_rt2.jsonl", split=split, use_cot=use_cot)

        elif dataset_name == "PhysicsQA":
            dataset = QADataset("scitrust_datasets/truthfulness_open_ended/Physics_qa_rt2.jsonl", split=split, use_cot=use_cot)

        elif dataset_name == "BiologyQA":
            dataset = QADataset("scitrust_datasets/truthfulness_open_ended/Biology_qa_rt2.jsonl", split=split, use_cot=use_cot)

        elif dataset_name == "ComputerScienceQA":
            dataset = QADataset("scitrust_datasets/truthfulness_open_ended/Computer Science_qa_rt2.jsonl", split=split, use_cot=use_cot)

        elif dataset_name == "MaterialsScienceQA":
            dataset = QADataset("scitrust_datasets/truthfulness_open_ended/Materials Science_qa_rt2.jsonl", split=split, use_cot=use_cot)

        elif from_file != '':
            dataset = QADataset(from_file, split=split, use_cot=use_cot)

        else:
            print("Dataset {} not supported. Supported datasets: SciQ, GPQA, ARC-E, ARC-C, OBQA, ChemistryQA, PhysicsQA, BiologyQA, ComputerScienceQA, MaterialsScienceQA.".format(dataset_name))

    elif perspective == "truthfulness_sycophancy":
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

    elif 'adv_robustness' in perspective:

        if perspective == 'adv_robustness_textfooler':
            if dataset_name == "SciQ":
                dataset = AdvDataset("scitrust_datasets/adv_datasets/multiple_choice/llama2-7b_textfooler_SciQ_0_0.json", split=split)
            elif dataset_name == "ARC-C":
                dataset = AdvDataset("scitrust_datasets/adv_datasets/multiple_choice/llama2-7b_textfooler_ARC-C_0_0.json", split=split)
            elif dataset_name == "GPQA":
                dataset = AdvDataset("scitrust_datasets/adv_datasets/multiple_choice/llama2-7b_textfooler_GPQA_0_0.json", split=split)
            else: 
                print("Dataset {} not supported. Supported datasets: SciQ, ARC-E, ARC-C, and GPQA.".format(dataset_name))
                exit()

        elif perspective == 'adv_robustness_textbugger':
            if dataset_name == "SciQ":
                dataset = AdvDataset("scitrust_datasets/adv_datasets/multiple_choice/llama2-7b_textbugger_SciQ_0_0.json", split=split)
            elif dataset_name == "ARC-C":
                dataset = AdvDataset("scitrust_datasets/adv_datasets/multiple_choice/llama2-7b_textbugger_ARC-C_0_0.json", split=split)
            elif dataset_name == "GPQA":
                dataset = AdvDataset("scitrust_datasets/adv_datasets/multiple_choice/llama2-7b_textbugger_GPQA_0_0.json", split=split)
            else:
                print("Dataset {} not supported. Supported datasets: SciQ, ARC-E, ARC-C, and GPQA.".format(dataset_name))
                exit()
        elif perspective == 'adv_robustness_stresstest':
            if dataset_name == "SciQ":
                dataset = AdvDataset("scitrust_datasets/adv_datasets/multiple_choice/llama2-7b_textbugger_SciQ_0_0.json", split=split)
            elif dataset_name == "ARC-C":
                dataset = AdvDataset("scitrust_datasets/adv_datasets/multiple_choice/llama2-7b_textbugger_ARC-C_0_0.json", split=split)
            elif dataset_name == "GPQA":
                dataset = AdvDataset("scitrust_datasets/adv_datasets/multiple_choice/llama2-7b_stresstest_GPQA_0_0.json", split=split)
            else:
                print("Dataset {} not supported. Supported datasets: SciQ, ARC-E, ARC-C, and OBQA.".format(dataset_name))
                exit()
        elif 'adv_robustness_open_ended' in perspective:
            attack = perspective.split("_")[-1]
            if attack != 'character-level' and attack != 'word-level' and attack != 'sentence-level':
                print("Attack not supported.")
                exit()
            if dataset_name == "ChemistryQA":
                dataset = QADataset("scitrust_datasets/adv_datasets/open_ended/chemistry_oa_chatgpt-4o_500_adv_{}.jsonl".format(attack), split=split)
            elif dataset_name == "ComputerScienceQA":
                dataset = QADataset("scitrust_datasets/adv_datasets/open_ended/computer_science_oa_chatgpt-4o_500_adv_{}.jsonl".format(attack), split=split)
            elif dataset_name == "BiologyQA":
                dataset = QADataset("scitrust_datasets/adv_datasets/open_ended/biology_oa_chatgpt-4o_500_adv_{}.jsonl".format(attack), split=split)
            elif dataset_name == "PhysicsQA":
                dataset = QADataset("scitrust_datasets/adv_datasets/open_ended/physics_oa_chatgpt-4o_500_adv_{}.jsonl".format(attack), split=split)
            elif dataset_name == "MaterialsScienceQA":
                dataset = QADataset("scitrust_datasets/adv_datasets/open_ended/materials_science_oa_chatgpt-4o_500_adv_{}.jsonl".format(attack), split=split)
            elif dataset_name == "LogicInference":
                dataset = QADataset("scitrust_datasets/adv_datasets/open_ended/logicinference_oa_chatgpt-4o_500_adv_{}.jsonl".format(attack), split=split)
            else:
                print("Dataset {} not supported. Supported datasets: ChemistryQA, ComputerScienceQA, BiologyQA, PhysicsQA, and LogicInference.".format(dataset_name))
                exit()
        else:
            print("Attack not supported. Supported attacks: textbugger, textfooler, stresstest.")
            exit()

    elif 'safety' in perspective:

        if dataset_name == "WMDP-BIO":
            dataset = WMDPDataset('bio', split=split, k=k)
        elif dataset_name == "WMDP-CHEM":
            dataset = WMDPDataset('chem', split=split, k=k)
        elif dataset_name == "WMDP-CYBER":
            dataset = WMDPDataset('cyber', split=split, k=k)
        elif dataset_name == "HarmBench-CHEM-BIO":
            dataset = HarmBenchDataset(k=k, subset='chemical_biological', split=split)
        elif dataset_name == "HarmBench-CYBERCRIME-INTRUSION":
            dataset = HarmBenchDataset(k=k, subset='cybercrime_intrusion', split=split)
        else:
            print("Dataset {} not supported.")


    elif "scientific_ethics" in perspective:

        if dataset_name == "scientific_ethics_full":
            dataset = SciEthicsDataset(split=split, k=k)

        elif dataset_name == "scientific_ethics_ai":
            dataset = SciEthicsDataset(subset='AI', split=split, k=k)

        elif dataset_name == 'scientific_ethics_animal_testing':
            dataset = SciEthicsDataset(subset='AT', split=split, k=k)

        elif dataset_name == 'scientific_ethics_bias_objectivity':
            dataset = SciEthicsDataset(subset='BO', split=split, k=k)

        elif dataset_name == 'scientific_ethics_data_privacy':
            dataset = SciEthicsDataset(subset='DP', split=split, k=k)

        elif dataset_name == 'scientific_ethics_dual_use_research':
            dataset = SciEthicsDataset(subset='DU', split=split, k=k)

        elif dataset_name == 'scientific_ethics_environmental_impact':
            dataset = SciEthicsDataset(subset='EI', split=split, k=k)

        elif dataset_name == 'scientific_ethics_human_subjects':
            dataset = SciEthicsDataset(subset='HS', split=split, k=k)

        elif dataset_name == 'scientific_ethics_genetic_modification':
            dataset = SciEthicsDataset(subset='GM', split=split, k=k)

        else:
            print("Dataset {} not supported. Supported datasets: scientific_ethics_full, scientific_ethics_ai, scientific_ethics_animal_testing, scientific_ethics_bias_objectivity, scientific_ethics_data_privacy, scientific_ethics_dual_use_research, scientific_ethics_environmental_impact, scientific_ethics_human_subjects".format(dataset_name))
            exit()

    elif perspective == "truthfulness_logical_reasoning":

        if dataset_name == 'LogicInference':
            dataset = LogicInferenceDataset(k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'ReClor':
            dataset = ReClorDataset(k=k, split=split, use_cot=use_cot)

        elif dataset_name == 'LogiQA':
            dataset = LogiQADataset(k=k, split=split, use_cot=use_cot)

        else:
            print("Dataset {} not supported. Supported datasets: LogicInference, ReClor, and LogiQA.".format(dataset_name))

    elif perspective == "truthfulness_hallucination":

        if dataset_name == "ChemistryQA":
            dataset = QADataset("scitrust_datasets/truthfulness_open_ended/chemistry_qa_chatgpt-4o.jsonl", split=split)

        elif dataset_name == "PhysicsQA":
            dataset = QADataset("scitrust_datasets/truthfulness_open_ended/physics_qa_chatgpt-4o.jsonl", split=split)

        elif dataset_name == "BiologyQA":
            dataset = QADataset("scitrust_datasets/truthfulness_open_ended/biology_qa_chatgpt-4o.jsonl", split=split)

        elif dataset_name == "ComputerScienceQA":
            dataset = QADataset("scitrust_datasets/truthfulness_open_ended/computer_science_qa_chatgpt-4o.jsonl", split=split)

        else:
            print("Dataset {} not supported. Supported datasets: ChemistryQA, PhysicsQA, BiologyQA, and ComputerScienceQA.".format(dataset_name))

    else:
        print("Error: Please enter a valid trustworthiness perspective. Supported options: truthfulness, sycophancy, adv_robustness, scientific_ethics, logical_reasoning, hallucination.")
        exit()

    return dataset

def append_record(record, filename):

    with open(filename, 'a') as f:
        json.dump(record, f)
        f.write('\n')

def generate_samples(batch, tokenizer, model, device, openended=False, use_cot=False, n_samples=4):
    gen_text_samples_batch = []
    #print('len(batch)', len(batch))
    if openended and use_cot:
        max_new_tokens=600
    elif openended:
        max_new_tokens=300
    elif not openended and use_cot:
        max_new_tokens = 503
    else:
        max_new_tokens=3

    for d in zip(batch[0], batch[1]):
        gen_text_samples = []
        for n in range(n_samples):
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

def send_prompt_to_chatgpt(prompt, model, api_key, max_tokens):

    client = OpenAI(
        api_key=api_key,  # This is the default and can be omitted
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        #max_completion_tokens=max_tokens,
    )

    response = chat_completion.choices[0].message.content #requests.post(url, json=data, headers=headers)
    #if response.status_code == 200:
    return response

def send_prompt_to_gemini(prompt, api_key, max_tokens):

    client = genai.Client(api_key=api_key)

    chat_completion = client.models.generate_content(
        model="gemini-2.0-flash",
        #max_tokens=max_tokens,
        contents=prompt,
    )
    response = chat_completion.text
    #print(response)
    #exit()
    #if response.status_code == 200:
    return response

def send_prompt_to_claude(prompt, api_key, max_tokens):

    client = anthropic.Anthropic(api_key=api_key)

    chat_completion = client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    response = chat_completion.content[0].text
    #print(response)
    #exit()
    #if response.status_code == 200:
    return response

def generate_samples_from_api(batch, model_name, api_key, openended, use_cot, n_samples=4):

    #print("Num Samples", n_samples)

    gen_text_samples_batch = []
    #print('len(batch)', len(batch))
    if openended and use_cot:
        max_new_tokens=600
    elif openended:
        max_new_tokens=300
    elif not openended and use_cot:
        max_new_tokens=303
    elif model_name == 'claude-sonnet-3.7':
        max_new_tokens=300
    else:
        max_new_tokens=3

    for d in zip(batch[0], batch[1]):
        gen_text_samples = []
        for n in range(n_samples):
            print('n', n)
            if model_name == 'gpt-o1':
            	gen_text = send_prompt_to_chatgpt(d[0], 'o1', api_key, max_new_tokens)
            elif model_name == 'gpt-o3-mini':
                gen_text = send_prompt_to_chatgpt(d[0], 'o3-mini', api_key, max_new_tokens)
            elif model_name == 'claude-sonnet-3.7':
                gen_text = send_prompt_to_claude(d[0], api_key, max_new_tokens)
            elif model_name == 'gemini-2.0-pro': 
                gen_text = send_prompt_to_gemini(d[0], api_key, max_new_tokens)
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

