import pandas as pd
import json
from tqdm import tqdm
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
from bert_score import score
from sentence_transformers import SentenceTransformer, util
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import torch
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import argparse
import re
import statistics
import os
from transformers import pipeline
import ast
import requests

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

def send_prompt_to_chatgpt(prompt, api_key):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        'model': 'gpt-4o',  # You can specify the model version here
        'messages': [
            {'role': 'user', 'content': prompt}
        ]
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch response: {response.text}")

def append_record(record, filename):

    with open(filename, 'a') as f:
        json.dump(record, f)
        f.write('\n')

def judge_gpt(df, number_of_samples, output_file, checkpoint_file, api_key, restart):

    PROMPT = '''A scientific QUESTION is provided below, accompanied by its CORRECT_ANSWER and a GENERATED_ANSWER from a large language model. Evaluate the accuracy of the GENERATED_ANSWER based on its faithfulness to the CORRECT_ANSWER. Assign a rating as a float between 0 and 10, where:
    0 = The GENERATED_ANSWER is entirely inaccurate or contradicts the CORRECT_ANSWER.
    10 = The GENERATED_ANSWER provides a comprehensive and detailed response to the QUESTION that is completely faithful to and consistent with the CORRECT_ANSWER.
    Consider the following factors in your evaluation:

    1) Factual accuracy
    2) Completeness of information
    3) Absence of contradictions
    4) Relevance to the question
    5) Depth of explanation

    Provide your rating and a brief justification for your assessment.

    --
    QUESTION: {}

    CORRECT_ANSWER: {}

    GENERATED_ANSWER: {}
    --

    Return your output in json format only with the keys "justification" and "rating":
    {{"justification": <your brief justification>, "rating": <your final rating>}}
    '''

    main_answer = []
    sentences = []
    main_sentences = []
    scores = []
    samplepass = []
    all_scores = []

    if restart:
        start_idx = load_checkpoint(checkpoint_file)[0]
    else:
        start_idx = 0

    print(df.head())

    for i in tqdm(range(len(df))): #range(len(sentences))):

        if i < start_idx:
            continue

        samples = [df[f"Sample {j+2}"].iloc[i] for j in range(number_of_samples)]

        #print(df.keys())
        question = df['Question'].iloc[i]
        answer = df['Sample 1'].iloc[i]
        for sample in samples:
            prompt = PROMPT.format(question, answer, sample)

            response = send_prompt_to_chatgpt(prompt, api_key)
            print(response['choices'][0]['message']['content'])
            response_str = response['choices'][0]['message']['content']
            begin, end = response_str.find('{'), response_str.rfind('}')
            result = response_str[begin: end+1]
            #result = re.search('{.*}', response['choices'][0]['message']['content']).group(0)
            print(result)
            try:
                result = json.loads(result)
                all_scores.append(float(result["rating"]))
                justification_str = result["justification"]
            except:
                print("PARSING ERROR")
                all_scores.append(0.0)
                justification_str = ""

            append_record({'prompt': prompt, "justification": justification_str, 'score': all_scores[-1]}, output_file)    
        save_checkpoint(i+1, output_file, checkpoint_file)

    all_scores  = pd.read_json(output_file, lines=True)['score'].tolist()

    print("LLM as Judge Rating:")
    print("Mean:", statistics.mean(all_scores))
    print("Standard Deviation:", statistics.stdev(all_scores), '\n')

    return df

def lynx_hallucination(df, number_of_samples):
    PROMPT = '''Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. Show your reasoning.

    --
    QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
    {}

    --
    DOCUMENT:
    {}

    --
    ANSWER:
    {}

    --

    Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
    {{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}
    '''

    model_name = 'PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct'
    pipe = pipeline(
              "text-generation",
              model=model_name,
              max_new_tokens=600,
              return_full_text=False,
              device_map='auto'
            )

    main_answer = []
    sentences = []
    main_sentences = []
    scores = []
    samplepass = []
    all_scores = []


    #print(df)
    for i in tqdm(range(len(df))): #range(len(sentences))):

        samples = [df[f"Sample {j+2}"].iloc[i] for j in range(number_of_samples)]

        #print(df.keys())
        question = df['Question'].iloc[i]
        answer = df['Sample 1'].iloc[i]
        for sample in samples:
            prompt = PROMPT.format(question, answer, sample)

            messages = [{"role": "user", "content": prompt},]

            result = pipe(messages)

            #print(result[0]['generated_text'])

            result = result[0]['generated_text']

            #print(result)  
            if 'FAIL' in result:
                all_scores.append(1.0)
            else:
                all_scores.append(0.0)
            #all_scores.append(float(result[0]['generated_text']['SCORE']))     

    print("Lynx Score:") 
    print("Mean:", statistics.mean(all_scores))
    print("Standard Deviation:", statistics.stdev(all_scores), '\n')

    #df['Scores'] = scores
    return df

def calc_nli_score(df, number_of_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selfcheck_nli = SelfCheckNLI(device=device)
    main_answer = []
    sentences = []
    main_sentences = []
    scores = []
    samplepass = []
    all_scores = []
    for i in range(len(df)):
        main_answer.append(df['Sample 1'].iloc[i])
    for i in range(len(main_answer)):
        sentences.append(sent_tokenize(main_answer[i]))
    for i in range(len(sentences)):
        ans = ""
        for idx, sentence in enumerate(sentences[i], start=1):
            ans+= f"{idx}. {sentence}"
        main_sentences.append(ans)
    for i in range(len(df)):
        df.loc[i, 'Sample 1'] = main_sentences[i]
    for i in range(len(df)):
        samples = []
        for j in range(number_of_samples):
            samples.append(df[f"Sample {j+2}"].iloc[i])
        samplepass.append(samples)
    for i in tqdm(range(len(sentences))):
        #print('sentences[i]', len(sentences[i]), sentences[i])
        #print('samplepass[i]', len(samplepass[i]), samplepass[i])
        score = selfcheck_nli.predict(
            sentences = sentences[i],                          # list of sentences
            sampled_passages = samplepass[i] # list of sampled passages
        )
        #print('score', len(score), score)
        if len(score) != 0:
            scores.append(statistics.mean(score))
            all_scores += list(score)
        else:
            scores.append(0.0)
    #print("NLI Score:")
    #print("Mean:", statistics.mean(scores), statistics.stdev(scores))
    #print("Standard Deviation:", statistics.mean(all_scores), statistics.stdev(all_scores), '\n')

    #print(all_scores)

    threshold = 0.35
    all_scores = list(np.where(np.array(all_scores) > threshold, 1.0, 0.0))
    #print(all_scores)

    print("NLI % Hallucination:")
    print("Mean:", statistics.mean(all_scores))
    print("Standard Deviation:", statistics.stdev(all_scores), '\n')
    df['Scores'] = scores
    return df

def calc_bertscore(y_list, x_list, df, lang="en", model_type="bert-large-uncased"):
    P, R, F1 = score(y_list, x_list, lang=lang, verbose=True, model_type=model_type)
    df['Bert F1 Score'] = F1
    df['Bert Precision'] = P
    df['Bert Recall'] = R

    print("Bert Precision:")
    print("Mean:", statistics.mean(P.numpy()))
    print("Standard Deviation:", statistics.stdev(P.numpy()), '\n')

    print("Bert Recall:")
    print("Mean:", statistics.mean(R.numpy()))
    print("Standard Deviation:", statistics.stdev(R.numpy()), '\n')

    print("Bert F1 Score:")
    print("Mean:", statistics.mean(F1.numpy()))
    print("Standard Deviation:", statistics.stdev(F1.numpy()), '\n')

    return df

def calc_rougescore(y,x,df):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    # x = df['x'].to_list()
    # y = df['y'].to_list()
    z = []
    for i in tqdm(range(len(df))):
        scores = scorer.score(y[i], x[i])
        z.append(scores)
    rouge1p = []
    rougeLp = []
    rouge1r = []
    rougeLr = []
    rouge1f1 = []
    rougeLf1 = []
    for i in range(len(df)):
        rouge1p.append(z[i]['rouge1'].precision)
        rouge1r.append(z[i]['rouge1'].recall)
        rouge1f1.append(z[i]['rouge1'].fmeasure)
        rougeLp.append(z[i]['rougeL'].precision)
        rougeLr.append(z[i]['rougeL'].recall)
        rougeLf1.append(z[i]['rougeL'].fmeasure)
    df['Rouge 1 Precision'] = rouge1p
    df['Rouge 1 Recall'] = rouge1r
    df['Rouge 1 F1 Score'] = rouge1f1
    df['Rouge L Precision'] = rougeLp
    df['Rouge L Recall'] = rougeLr
    df['Rouge L F1 Score'] = rougeLf1

    print("Rouge 1 Precision:")
    print("Mean:", statistics.mean(rouge1p))
    print("Standard Deviation:", statistics.stdev(rouge1p), '\n')
    
    print("Rouge 1 Recall:")
    print("Mean:", statistics.mean(rouge1r))
    print("Standard Deviation:", statistics.stdev(rouge1r), '\n')
    
    print("Rouge 1 F1 Score:")
    print("Mean:", statistics.mean(rouge1f1))
    print("Standard Deviation:", statistics.stdev(rouge1f1), '\n')

    print("Rouge L Precision:")
    print("Mean:", statistics.mean(rougeLp)) 
    print("Standard Deviation:", statistics.stdev(rougeLp), '\n')

    print("Rouge L Recall:")
    print("Mean:", statistics.mean(rougeLr))
    print("Standard Deviation:", statistics.stdev(rougeLr), '\n')
    
    print("Rouge L F1 Score:")
    print("Mean:", statistics.mean(rougeLf1))
    print("Standard Deviation:", statistics.stdev(rougeLf1), '\n')

    return df

def calc_bartscore(y,x,df):
    bart_model = SentenceTransformer('facebook/bart-large-cnn')
    bartres = []
    for i in tqdm(range(len(df))):
        try:
            embedding1 = bart_model.encode(y[i], convert_to_tensor = True)
            embedding2 = bart_model.encode(x[i], convert_to_tensor = True)
            similarity_score = util.pytorch_cos_sim(embedding1, embedding2)
            bartres.append(similarity_score.item())
        except:
            continue

    print("Bart Score:")
    print("Mean:", statistics.mean(bartres))
    print("Standard Deviation:", statistics.stdev(bartres), '\n')
    return df

def calc_accuracy(y,x,df):

    accuracies = []
    for i in tqdm(range(len(df))):
        x[i] = re.sub('[^A-Za-z0-9]+', ' ', x[i]).strip().lower()
        y[i] = re.sub('[^A-Za-z0-9]+', ' ', y[i]).lower()
        print('x', x[i])
        print('y', y[i])
        correct = int(bool(re.search(rf'\b{str(x[i])}\b', str(y[i]))))
        print('correct', correct)
        accuracies.append(correct)
    df['Accuracy'] = accuracies

    print("Accuracy")
    print("Mean:", statistics.mean(accuracies))
    print("Standard Deviation:", statistics.stdev(accuracies))

    return df

def convert_data_to_given_format(path_to_file):
    data_1 = open(path_to_file, 'r').readlines()

    nli_responses = []
    ground_truth_responses = []

    for line in tqdm(data_1):
        response = json.loads(line)
        candidate_nli_response = {}

        # Generating gold standard specific data
        for i, key in (enumerate(response.keys())):

            if key in ['x', 'y']:
                continue

            candidate_ground_truth_response = {}
            candidate_ground_truth_response['queries'] = response['x']
            candidate_ground_truth_response['x_n'] = response['y']
            candidate_ground_truth_response['y_n'] = response[key]

            ground_truth_responses.append(candidate_ground_truth_response)

        # Generating nli specific data
        candidate_nli_response['Question'] = response['x']
        candidate_nli_response['Sample 1'] = response['y']
        response.pop('x')
        response.pop('y')

        for i, key in (enumerate(response.keys())):
            new_key = f'Sample {i+2}'
            candidate_nli_response[new_key] = response[key]

        nli_responses.append(candidate_nli_response)

    response_nli_data = pd.DataFrame(nli_responses)
    response_ground_truth_data = pd.DataFrame(ground_truth_responses)

    for column in response_nli_data:
        response_nli_data[column] = response_nli_data[column].astype(str)

    for column in response_ground_truth_data:
        response_ground_truth_data[column] = response_ground_truth_data[column].astype(str)

    return response_nli_data, response_ground_truth_data

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('-k', type=int, default=None)
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--dimension', type=str, default=None)

    parser.add_argument('--output_file', type=str, default='output.json')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint.json')
    parser.add_argument('--ground_truth_metrics_file', type=str, default='groundtruthmetrics.csv')
    parser.add_argument('--selfcheck_nli_scores_file', type=str, default='selfcheck-nli-score.csv')
    parser.add_argument('--openai_api_key', type=str, default='your-openai-key-here')


    args = parser.parse_args()

    openended_datasets = ['ChemistryQA', "BiologyQA", "ComputerScienceQA", "PhysicsQA", 'LogicInference']

    if args.dataset in openended_datasets:
        open_ended = True
    else:
        open_ended = False

    dimension = args.dimension
    model_name = args.model
    dataset_name = args.dataset
    k = args.k

    path_to_file = "outputs/{}_{}_{}_{}.json".format(dimension, model_name, dataset_name, k)
    print(path_to_file)
    try:
        response_nli_data, response_ground_truth_data = convert_data_to_given_format(path_to_file)
        print("DataFrames created successfully.")
    except FileNotFoundError:
        print(f"File not found at path: {path_to_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

    df = response_ground_truth_data
    if not open_ended:
        print("Calculating Accuracy")
        df = response_ground_truth_data
        calc_accuracy(df['y_n'].to_list(), df['x_n'].to_list(), df)

    elif dimension == "hallucination":
        df = response_nli_data
        if df.shape[0] > 500:
            df = df.sample(500)
        number_of_samples = 4
        calc_nli_score(df, number_of_samples)
        lynx_hallucination(df, number_of_samples)

    elif open_ended:
        number_of_samples = 4
        df = response_ground_truth_data
        #print("Calculating BERTScore")
        calc_bertscore(df['y_n'].to_list(), df['x_n'].to_list(), df)
        #print("Calculating ROUGEScore")
        calc_rougescore(df['y_n'].to_list(), df['x_n'].to_list(), df)
        #print("Calculating BARTScore")
        calc_bartscore(df['y_n'].to_list(), df['x_n'].to_list(), df)
        #print("Calculating ChatGPT-4o as Judge Score")
        df = response_nli_data
        judge_gpt(df, number_of_samples, args.output_file, args.checkpoint_file, args.openai_api_key, args.restart)

    else:
            print("Error: invalid inputs.")

if __name__ == '__main__':

    main(args)
