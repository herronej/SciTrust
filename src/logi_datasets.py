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
    def __init__(self, k=5, split=0):

        self.k = k
        #ds = load_dataset('KK04/LogicInference_OA') 
        #ds.save_to_disk("data/LI")
        dataset = load_from_disk("data/LI") #load_dataset('KK04/LogicInference_OA', split='train') #Dataset.from_file("data/LogicInference_OA.hf/train/data-00000-of-00001.arrow")#'KK04/LogicInference_OA', split='train')
        df_pandas = dataset['train'].to_pandas()
        if split != None:
            df_pandas = np.array_split(df_pandas, 1000)[split]

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

              #multiple_choice = ''
              #random_choices = random.sample(range(4), 4)
              '''completion_str = curr_item[''] + '\n'

              for choice in ['A', 'B', 'C', 'D']:
                  line = '' #choice + ': '
                  line += str(curr_item[choice]) + '\n'
                  completion_str += line'''

              completion_str = curr_item['INSTRUCTION']

              if k_i == self.k:
                  #labels.append(item['correct_answer'])
                  all_shots_str += prompt_final.format(completion_str)
              else:
                  all_shots_str += prompt.format(completion_str, curr_item['RESPONSE']+'\n') #correct_answer_letter)

          output_data.append(all_shots_str)
          labels.append(curr_item['RESPONSE'])

      #print(output_data[0])
      #print(labels[0])
      #print(df.shape)
      #exit()
      return output_data, labels


class ReClorDataset(Dataset):
    def __init__(self, k=5):

        self.k = k

        #dataset = load_dataset('voidful/ReClor')

        df_pandas = pd.read_json('reclor_data/train.json') #dataset['train'].to_pandas()
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

      prompt = """\nWhat is the correct answer to this question:\n{}\nQuestion: {}Answer: {}"""
      prompt_final = """\nWhat is the correct answer to this question:\n{}\nQuestion: {}Answer:"""

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
                  all_shots_str += prompt_final.format(curr_item['context'], completion_str)
              else:
                  all_shots_str += prompt.format(curr_item['context'], completion_str, choices[curr_item['label']]) #correct_answer_letter)

          output_data.append(all_shots_str)
          labels.append(choices[curr_item['label']])

      '''print(output_data[0])
      print(labels[0])
      exit()'''
      return output_data, labels


class LogiQADataset(Dataset):
    def __init__(self, k=5):

        self.k = k

        dataset = load_dataset('lucasmccabe/logiqa', split='train') #[{}%:{}%]'.format(split*10, split*10+10))

        df_pandas = dataset.to_pandas()
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

      prompt = """\nWhat is the correct answer to this question:\n{}\nQuestion: {}Answer: {}"""
      prompt_final = """\nWhat is the correct answer to this question:\n{}\nQuestion: {}Answer:"""
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
                  all_shots_str += prompt_final.format(curr_item['context'], completion_str)
              else:
                  all_shots_str += prompt.format(curr_item['context'], completion_str, choices[curr_item['correct_option']]) #correct_answer_letter)

          output_data.append(all_shots_str)
          labels.append(choices[curr_item['correct_option']])

      '''print(output_data[0])
      print(labels[0])
      exit()'''
      return output_data, labels


'''class ARBDataset(Dataset):
    def __init__(self, subset="math", k=5):

        self.k = k
        self.answer_format = answer_format

        response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/math/numerical")

        response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/math/symbolic")

        response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/math/prooflike")

        response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/mcatReading/val")

        response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/physics/numerical/noimg")

        response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/physics/symbolic/noimg")

        response = requests.get("https://advanced-reasoning-benchmark.netlify.app/api/lib/mcatScience/val/noimg")

        

        data = response.json()

        dataset = load_dataset("Idavidrein/gpqa", 'gpqa_main')

        df_pandas = dataset['train'].to_pandas()

        self.data, self.labels = self.preprocess(df_pandas)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        return data, labels

    def get_labels(self):
        return self.labels
'''
