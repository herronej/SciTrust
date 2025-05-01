import torch
import json
from torch.utils.data import Dataset
import random
from tqdm.auto import tqdm
from datasets import load_dataset
import pandas as pd
import requests
import os
import numpy as np

class AstroMLMCDataset(Dataset):
    def __init__(self, sycophancy=False, k=0, split=None, use_cot=False):

        self.k = k
        self.sycophancy = sycophancy
        self.use_cot = use_cot

        dataset = load_dataset('AstroMLab/araa-mcq-gemini-1.5-generated-v2-temp-0', split='train')

        df_pandas = dataset.to_pandas()
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

      prompt = """What is the correct answer to this question? Answer A, B, C, or D.
			Question: {question}
			Options:
			A: {a}
			B: {b}
			C: {c}
			D: {d}
            Answer:
			"""

      #output_format = """{"ANSWER": "[The choice you decide to choose]", "EXPLANATION": "[Provide a valid explanation for the answer mentioned in ANSWER]"}"""


      output_data = []
      labels = []
      for idx, item in df.iterrows(): 

          final_prompt = prompt.format(question=item['question'], a=item['A'], b=item['B'], c=item['C'], d=item['D'])#, output_format=output_format)

          output_data.append(final_prompt)
          labels.append(item['correct'])

      '''print(output_data[0])
      print(labels[0])
      exit()'''
      return output_data, labels

class AstroMLQADataset(Dataset):
    def __init__(self, split=None, use_cot=False):

        self.k = 0
        self.use_cot = use_cot

        dataset = load_dataset('AstroMLab/araa-qa-gemini-1.5-generated-v2', split='train')
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

      prompt = """You are an expert in general astrophysics. Your task is to answer the following question on astrophysics, sourced from a dataset. The question is:
            Question: {question}
            Determine your answer using your knowledge of astrophysics.
            Ensure your answer is thorough and clearly articulated.
            """

      output_data = []
      labels = []
      for idx, item in df.iterrows():

          final_prompt = prompt.format(question=item['question'])

          output_data.append(final_prompt)
          labels.append(item['answer'])

      '''print(output_data[0])
      print(labels[0])
      exit()'''
      return output_data, labels


