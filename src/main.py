import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm.auto import tqdm
from datasets import load_dataset
import pandas as pd
from .sci_datasets import SciQDataset, GPQADataset, ARCDataset, HendrycksDataset, OpenBookQADataset, SciEthicsDataset
from .logi_datasets import LogicInferenceDataset, ReClorDataset, LogiQADataset
import argparse
import time
from .utils import get_dataset, append_record, generate_samples, save_checkpoint, load_checkpoint
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Found cuda")
else:
    device = torch.device("cpu")
    print("Couldn't find cuda")


def main():

    #print(args)
    print('parsing args')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('-k', type=int, default=None)
    parser.add_argument('--answer_format', type=str, default='S')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--dimension', type=str, default=None)

    args = parser.parse_args()
    print('#devices =', torch.cuda.device_count())
    print('available =', torch.cuda.is_available())


    openended_datasets = ['ChemistryQA', "BiologyQA", "ComputerScienceQA", "PhysicsQA", 'LogicInference']

    dataset_name = args.dataset
    model_name = args.model #'forge'
    k = args.k
    restart = args.restart
    split = args.split
    dimension = args.dimension
    openended = dataset_name in openended_datasets
    #device = args.device

    dataset = get_dataset(dimension, dataset_name, k=k, split=split)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    output_path = "outputs/{}_{}_{}_{}.json".format(dimension, model_name, dataset_name, k)
    checkpoint_path = "checkpoints/chkpt_{}_{}_{}_{}_{}.json".format(dimension, model_name, dataset_name, k, split)
    features = ['x', 'y', 'gen1', 'gen2', 'gen3', 'gen4']#, 'gen5']


    generation_data = []
    if model_name == 'llama3-70b-instruct':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", device_map='auto')
    elif model_name == 'forge-l-instruct':
        model_path = '../models/forge-l-instruct-base1'
        from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
        model = GPTNeoXForCausalLM.from_pretrained(model_path, device_map='auto')
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_path)
    elif model_name == 'sciglm-6b':
        from transformers import AutoTokenizer, AutoModel
        model_path = 'zd21/SciGLM-6B'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
    elif model_name == "darwin-7b":
        from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
        model_path = "../models/darwin-7b"
        if device=="cuda":
            torch_type = torch.float16
        else:
            torch_type = torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_path, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
        model = LlamaForCausalLM.from_pretrained(model_path, load_in_8bit=False, torch_dtype=torch_type, device_map='auto')
    elif model_name == 'galactica-120b':
        # Load model directly
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-120b")
        model = AutoModelForCausalLM.from_pretrained("facebook/galactica-120b", device_map='auto')
    else:
        print("Model name {} invalid. Supported models: llama3-70b-instruct, forge-l-instruct, sciglm-6b, darwin-7b, galactica-120b".format(model_name))
        exit()

    print(torch.cuda.device_count())

    #load checkpoint
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)

    if restart:
    	start_idx = load_checkpoint(checkpoint_path)[0]
    else:
        start_idx = 0

    start = time.time()
    #for batch_idx, batch in enumerate(tqdm(data_loader), start=start_idx):
    for batch_idx, batch in enumerate(tqdm(data_loader)):

        if batch_idx < start_idx:
            continue

        gen_text_samples_batch = generate_samples(batch, tokenizer, model, device, openended)
        for sample_data in gen_text_samples_batch:
            append_record(dict(zip(features, sample_data)), output_path)
        save_checkpoint(batch_idx+1, output_path, checkpoint_path)  # Save checkpoint after each batch
        end = time.time()
        print('batch inference time', end - start)
        #exit()

if __name__ == '__main__':

    main(args)
