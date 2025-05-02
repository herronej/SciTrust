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
from .utils import get_dataset, append_record, generate_samples, generate_samples_from_api, save_checkpoint, load_checkpoint
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Found cuda")
else:
    device = torch.device("cpu")
    print("Couldn't find cuda")


'''rank = int(os.environ["RANK"])
device = torch.device(f"cuda:{rank}")
torch.distributed.init_process_group("nccl", device_id=device)

local_rank = os.getenv("LOCAL_RANK")
device_string = "cuda:" + str(local_rank)
print("device_string", device_string)
'''
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
    parser.add_argument('--perspective', type=str, default=None)
    parser.add_argument('--use_cot', action='store_true')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--from_file', type=str, default=None)

    args = parser.parse_args()
    print('#devices =', torch.cuda.device_count())
    print('available =', torch.cuda.is_available())


    openended_datasets = ['ChemistryQA', "BiologyQA", "ComputerScienceQA", "PhysicsQA", "MaterialsScienceQA", 'LogicInference', "HarmBench-CHEM-BIO", "HarmBench-CYBERCRIME-INTRUSION", "AstroMLQADataset"]

    dataset_name = args.dataset
    model_name = args.model #'forge'
    k = args.k
    restart = args.restart
    split = args.split
    perspective = args.perspective
    from_file = args.from_file
    openended = (dataset_name in openended_datasets) or (from_file != None)
    use_cot = args.use_cot
    api_key = args.api_key
    #device = args.device

    print("openended", openended)
    print("use_cot", use_cot)

    dataset = get_dataset(perspective, dataset_name, k=k, split=split, use_cot=use_cot, from_file=from_file)

    if (model_name == 'gpt-o3-mini' or model_name == "gpt-o1" or model_name == 'claude-sonnet-3.7' or model_name == 'gemini-2.0-pro') and (openended and (perspective == 'truthfulness_misinformation' or perspective == 'truthfulness_hallucination')):
        dataset = dataset[:100]

    print("Dataset Length:", len(dataset))

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if from_file != None:
        output_path = "outputs/{}_{}_{}_{}_{}.json".format(perspective, model_name, from_file, k, use_cot)
        checkpoint_path = "checkpoints/chkpt_{}_{}_{}_{}_{}_{}.json".format(perspective, model_name, from_file, k, split, use_cot)
    else:
        output_path = "outputs/{}_{}_{}_{}_{}.json".format(perspective, model_name, dataset_name, k, use_cot)
        checkpoint_path = "checkpoints/chkpt_{}_{}_{}_{}_{}_{}.json".format(perspective, model_name, dataset_name, k, split, use_cot)
    features = ['x', 'y', 'gen1', 'gen2', 'gen3', 'gen4']#, 'gen5']


    generation_data = []
    if model_name == 'llama3.1-405b-instruct':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "meta-llama/Llama-3.1-405B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, tp_plan="auto") #device_map='auto')
    elif model_name == 'llama3.3-70b-instruct':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "meta-llama/Meta-Llama-3.3-70B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    elif model_name == 'llama3-8b':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "meta-llama/Meta-Llama-3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    elif model_name == 'forge-l-instruct':
        model_path = '/lustre/orion/proj-shared/stf218/junqi/chathpc/forge-l-instruct-base1' #'models/forge-l-instruct-base1'
        from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
        model = GPTNeoXForCausalLM.from_pretrained(model_path, device_map='auto')
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_path)
    elif model_name == 'sciglm-6b':
        from transformers import AutoTokenizer, AutoModel
        model_path = 'zd21/SciGLM-6B'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
    elif model_name == "darwin1.5-7b":
        from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
        model_path = "models/darwin1_v2"
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
    elif "astro" in model_name.lower():
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained('AstroMLab/'+model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained('AstroMLab/'+model_name)
    elif not (model_name != 'gpt-o3-mini' or model_name != 'gpt-o1' or model_name != 'claude-sonnet-3.7' or model_name != 'gemini-2.0-pro'):
        print("Model name {} invalid. Supported models: gpt-o1, claude-sonnet-3.7, gemini-2.0-pro, llama3-70b-instruct, forge-l-instruct, sciglm-6b, darwin-7b, galactica-120b".format(model_name))
        exit()

    print(torch.cuda.device_count())

    #load checkpoint
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if restart:
    	start_idx = load_checkpoint(checkpoint_path)[0]
    else:
        start_idx = 0

    start = time.time()
    #for batch_idx, batch in enumerate(tqdm(data_loader), start=start_idx):
    for batch_idx, batch in enumerate(tqdm(data_loader)):

        if batch_idx < start_idx:
            continue
        if  model_name == 'gpt-o3-mini' or model_name == "gpt-o1" or model_name == 'claude-sonnet-3.7' or model_name == 'gemini-2.0-pro':
            print("openended", openended, "perspective", perspective)
            if openended and (perspective == 'truthfulness_misinformation' or perspective == 'truthfulness_hallucination'):
                gen_text_samples_batch = generate_samples_from_api(batch, model_name, api_key, openended, use_cot, n_samples=4)
            else:
                gen_text_samples_batch = generate_samples_from_api(batch, model_name, api_key, openended, use_cot, n_samples=4)
            #gen_text_samples_batch = generate_samples_from_api(batch, model_name, api_key, openended, use_cot)
        else:
            #print("openended", openended, "perspective", perspective, "")
            gen_text_samples_batch = generate_samples(batch, tokenizer, model, device, openended, use_cot, n_samples=4)
        for sample_data in gen_text_samples_batch:
            append_record(dict(zip(features, sample_data)), output_path)
        save_checkpoint(batch_idx+1, output_path, checkpoint_path)  # Save checkpoint after each batch
        end = time.time()
        print('batch inference time', end - start)
        #exit()

if __name__ == '__main__':

    main(args)
