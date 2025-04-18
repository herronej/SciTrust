#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import torch
import json
from torch.utils.data import Dataset
import random
#from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
import pandas as pd
import requests
import os
import numpy as np

class LogicInferenceDataset(Dataset):
    def __init__(self, k=0, split=0, use_cot=False):

        self.k = k
        self.use_cot = use_cot
        #dataset = load_dataset('KK04/LogicInference_OA')
        dataset = load_from_disk("/lustre/orion/stf218/scratch/1eh/Trustworthiness-Scientific-LLMs/data/LI")
        df_pandas = dataset['train'].to_pandas().sample(n=500, random_state=0)
        if split != None:
            df_pandas = np.array_split(df_pandas, 100)[split]

        self.data, self.labels = self.preprocess(df_pandas)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        return data, labels

    def get_labels(self):
        return self.labels

    def preprocess(self, df):

      df_first_k = df.iloc[:self.k]
      df = df.iloc[self.k:]

      prompt = """{}\n{}\n"""
      if self.use_cot:
          prompt_final = """{} Include your reasoning steps in the format of [Reasoning Steps] your reasoning steps [End]. Use this exact format. """
      else:
          prompt_final = """{}"""

      output_data = []
      labels = []
      for idx, item in df.iterrows(): #item in input_data[self.k:]:
          #print(item)
          all_shots_str = '' #'The following are multiple choice questions (with answers).'
          correct_answer_letter = ''
          for k_i in range(self.k+1):

              if k_i < self.k:
                  curr_item = df_first_k.iloc[k_i]
                  #print(curr_item)
              else:
                  curr_item = item

              completion_str = curr_item['INSTRUCTION']

              if k_i == self.k:
                  #labels.append(item['correct_answer'])
                  all_shots_str += prompt_final.format(completion_str)
              else:
                  all_shots_str += prompt.format(completion_str, curr_item['RESPONSE']+'\n') #correct_answer_letter)

          output_data.append(all_shots_str)
          labels.append(curr_item['RESPONSE'])

      return output_data, labels


class ReClorDataset(Dataset):
    def __init__(self, k=0, use_cot=False, split=None):

        self.k = k
        self.use_cot = use_cot

        df_pandas = pd.read_json('reclor_data/train.json') #dataset['train'].to_pandas()

        if split != None:
            df_pandas = np.array_split(df_pandas, 100)[split]

        self.data, self.labels = self.preprocess(df_pandas)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        return data, labels

    def get_labels(self):
        return self.labels

    def preprocess(self, df):

      df_first_k = df.iloc[:self.k]
      df = df.iloc[self.k:]

      prompt_base = """Answer and explain the following multiple-choice question. 
            Give only one answer, either A, B, C, D, or E, but not more than one, and always give an answer.
            Provide a detailed explanation for why this answer is correct.
            Return your answer as a json string in the following format:
            {}
            Strictly adhere to this output format.
            Question: {}"""

      output_format = """{"ANSWER": "[The choice you decide to choose]", "EXPLANATION": "[Provide a valid explanation for the answer mentioned in ANSWER]"}"""

      prompt = prompt_base + """Answer: {}"""
      prompt_final = prompt_base

      output_data = []
      labels = []
      for idx, item in df.iterrows(): #item in input_data[self.k:]:
          all_shots_str = '' #'The following are multiple choice questions (with answers).'
          correct_answer_letter = ''
          for k_i in range(self.k+1):

              if k_i < self.k:
                  curr_item = df_first_k.iloc[k_i]
                  #print(curr_item)
              else:
                  curr_item = item

              #multiple_choice = ''
              #random_choices = random.sample(range(4), 4)
              completion_str = curr_item['question'] + '\n'
              choices = ['(A)', '(B)', '(C)', '(D)', '(E)']
              for i_c, choice in enumerate(curr_item['answers']):
                  line = choices[i_c] + ' ' #choice + ': '
                  line += str(choice) + '\n'
                  completion_str += line

              if k_i == self.k:
                  #labels.append(item['correct_answer'])
                  all_shots_str += prompt_final.format(output_format, curr_item['context'], completion_str)
              else:
                  all_shots_str += prompt.format(output_format, curr_item['context'], completion_str, choices[curr_item['label']]) #correct_answer_letter)

          output_data.append(all_shots_str)
          labels.append(choices[curr_item['label']])

      '''print(output_data[0])
      print(labels[0])
      exit()'''
      return output_data, labels


class LogiQADataset(Dataset):
    def __init__(self, k=5, use_cot = False, split=None):

        self.k = k

        self.use_cot = use_cot

        dataset = load_dataset('lucasmccabe/logiqa', split='train')

        df_pandas = dataset.to_pandas()

        if split != None:
            df_pandas = np.array_split(df_pandas, 100)[split]

        self.data, self.labels = self.preprocess(df_pandas)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        return data, labels

    def get_labels(self):
        return self.labels

    def preprocess(self, df):

      df_first_k = df.iloc[:self.k]
      df = df.iloc[self.k:]


      prompt_base = """Answer and explain the following multiple-choice question. 
            Give only one answer, either A, B, C, D, or E, but not more than one, and always give an answer.
            Provide a detailed explanation for why this answer is correct.
            Return your answer as a json string in the following format:
            {}
            Strictly adhere to this output format.
            Question: {}"""

      output_format = """{"ANSWER": "[The choice you decide to choose]", "EXPLANATION": "[Provide a valid explanation for the answer mentioned in ANSWER]"}"""

      prompt = prompt_base + """Answer: {}"""
      prompt_final = prompt_base



      choices = ['(A)', '(B)', '(C)', '(D)', '(E)']
      output_data = []
      labels = []
      for idx, item in df.iterrows(): #item in input_data[self.k:]:
          all_shots_str = '' #'The following are multiple choice questions (with answers).'
          correct_answer_letter = ''
          for k_i in range(self.k+1):

              if k_i < self.k:
                  curr_item = df_first_k.iloc[k_i]
                  #print(curr_item)
              else:
                  curr_item = item

              #multiple_choice = ''
              #random_choices = random.sample(range(4), 4)
              completion_str = curr_item['query'] + '\n'

              for i_c, choice in enumerate(curr_item['options']):
                  line = choices[i_c] + ' ' #'' #choice + ': '
                  line += str(choice) + '\n'
                  completion_str += line

              if k_i == self.k:
                  #labels.append(item['correct_answer'])
                  all_shots_str += prompt_final.format(output_format, curr_item['context'], completion_str)
              else:
                  all_shots_str += prompt.format(output_format, curr_item['context'], completion_str, choices[curr_item['correct_option']]) #correct_answer_letter)

          output_data.append(all_shots_str)
          labels.append(choices[curr_item['correct_option']])

      '''print(output_data[0])
      print(labels[0])
      exit()'''
      return output_data, labels


