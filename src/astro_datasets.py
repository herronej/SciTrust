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

        # 1) load and (optionally) split the raw dataframe
        ds = load_dataset(
            'AstroMLab/araa-mcq-gemini-1.5-generated-v2-temp-0',
            split='train'
        )
        df = ds.to_pandas()
        if split is not None:
            df = np.array_split(df, 1000)[split]

        # 2) carve off first k for demos, remainder for actual examples
        self.df_demos = df.iloc[:self.k].reset_index(drop=True)
        self.df_items = df.iloc[self.k:].reset_index(drop=True)

        # 3) build prompts + labels
        self.data, self.labels = self._preprocess()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

    def _format_one(self, row):
        """Generate the formatted prompt (ending just before the model’s answer)."""

        prompt = (
            f"What is the correct answer to this question? "
            f"Answer only A, B, C, or D.\n"
            f"Question: {row['question']}\n"
            f"Options:\n"
            f"A: {row['A']}\n"
            f"B: {row['B']}\n"
            f"C: {row['C']}\n"
            f"D: {row['D']}\n"
            f"The correct answer is: "
        )
        return prompt

    def _format_two(self, row):
        """Generate the formatted prompt (ending just before the model’s answer)."""

        prompt = (
            f"What is the correct answer to this question? "
            f"Answer with the full text of the correct option.\n"
            f"Question: {row['question']}\n"
            f"Options:\n"
            f"A: {row['A']}\n"
            f"B: {row['B']}\n"
            f"C: {row['C']}\n"
            f"D: {row['D']}\n"
            f"The correct answer is: "
        )
        return prompt

    def _format_three(self, row):
        """Generate the formatted prompt (ending just before the model’s answer)."""

        prompt = (
            f"What is the correct answer to this question? "
            f"Please explain your reasoning step by step before providing your answer. "
            f"Answer with the full text of the correct option (not just the letter).\n"
            f"Question: {row['question']}\n"
            f"Options:\n"
            f"A: {row['A']}\n"
            f"B: {row['B']}\n"
            f"C: {row['C']}\n"
            f"D: {row['D']}\n"
            f"Let's think step by step.\n"
            f"Reasoning: "
        )
        return prompt

    def _format_four(self, row):
        """Generate the formatted prompt (ending just before the model’s answer)."""

        prompt = f"""
        You are an expert in general astrophysics. Your task is to answer and explain the following multiple-choice question (MCQ) on astrophysics, sourced from a dataset

        The question is:
        **Question**: {row['question']}

        Options:
        - **A**: {row['A']}
        - **B**: {row['B']}
        - **C**: {row['C']}
        - **D**: {row['D']}

        Determine the correct answer using your astrophysics knowledge and provide a detailed explanation for why this answer is correct. 
        Ensure your explanation is thorough, clearly articulating your thought process based on astrophysical principles.

        Output format: 
        {{
            "ANSWER": "[The choice you decide to choose]",
            "EXPLANATION": "[Provide a valid explanation for the answer mentioned in "ANSWER]"
        }}

        Give only one answer, either A, B, C or D, but not more than one, and always give an answer.

        Provide your response in valid JSON format only. Begin your output with the JSON structure immediately, without any preceding text. Strictly adhere to the specified output format.
        """

        return prompt

    def _preprocess(self):
        # build the few‑shot prefix
        demo_prompts = []
        for _, shot in self.df_demos.iterrows():
            p = self._format_four(shot)
            # append the known answer for each demo
            p += shot['correct']
            demo_prompts.append(p)

        few_shot_prefix = "\n\n".join(demo_prompts) if demo_prompts else ""

        # now build the actual dataset
        out_prompts, out_labels = [], []
        for _, ex in self.df_items.iterrows():
            base = self._format_four(ex)
            full = f"{few_shot_prefix}\n\n{base}" if few_shot_prefix else base
            out_prompts.append(full)
            out_labels.append(ex['correct'])

        return out_prompts, out_labels

class AstroOlympiadMCDataset(Dataset):
    def __init__(self, sycophancy=False, k=0, split=None, use_cot=False):
        self.k = k
        self.sycophancy = sycophancy
        self.use_cot = use_cot

        print("AstroOlympiadMCDataset")
        exit()

        # 1) load and (optionally) split the raw dataframe
        '''ds = load_dataset(
            'AstroMLab/araa-mcq-gemini-1.5-generated-v2-temp-0',
            split='train'
        )'''
        df = pd.read_json("/scratch/1eh/AstroML/SciTrust/Olympiad_MCQ.jsonl", lines=True) #pd.read_csv("Olympiad_MCQ.csv", sep=";")#, header=True)

        # 2) carve off first k for demos, remainder for actual examples
        self.df_demos = df.iloc[:self.k].reset_index(drop=True)
        self.df_items = df.iloc[self.k:].reset_index(drop=True)

        # 3) build prompts + labels
        self.data, self.labels = self._preprocess()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

    def _format_one(self, row):
        """Generate the formatted prompt (ending just before the model’s answer)."""

        prompt = f"""
        You are an expert in general astrophysics. Your task is to answer and explain the following multiple-choice question (MCQ) on astrophysics, sourced from a dataset

        The question is:
        **Question**: {row['question']}

        Options:
        - **A**: {row['A']}
        - **B**: {row['B']}
        - **C**: {row['C']}
        - **D**: {row['D']}
        - **E**: {row['E']}

        Determine the correct answer using your astrophysics knowledge and provide a detailed explanation for why this answer is correct. 
        Ensure your explanation is thorough, clearly articulating your thought process based on astrophysical principles.

        Output format: 
        {{
            "ANSWER": "[The choice you decide to choose]",
            "EXPLANATION": "[Provide a valid explanation for the answer mentioned in "ANSWER]"
        }}

        Give only one answer, either A, B, C or D, but not more than one, and always give an answer.

        Provide your response in valid JSON format only. Begin your output with the JSON structure immediately, without any preceding text. Strictly adhere to the specified output format.
        """
        return prompt

    def _preprocess(self):
        # build the few‑shot prefix
        demo_prompts = []
        for _, shot in self.df_demos.iterrows():
            p = self._format_one(shot)
            # append the known answer for each demo
            p += shot['answer']
            demo_prompts.append(p)

        few_shot_prefix = "\n\n".join(demo_prompts) if demo_prompts else ""

        # now build the actual dataset
        out_prompts, out_labels = [], []
        for _, ex in self.df_items.iterrows():
            if len(ex['images']) == 0: 
                base = self._format_one(ex)
                full = f"{few_shot_prefix}\n\n{base}" if few_shot_prefix else base
                out_prompts.append(full)
                out_labels.append(ex['answer'])

        print(len(out_prompts), len(out_labels))
        exit()
        return out_prompts, out_labels

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

        print('len(data)', len(data))
        print("len(labels)", len(labels))

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

      print(output_data[0])
      print(labels[0])
      #exit()
      '''exit()'''
      return output_data, labels


