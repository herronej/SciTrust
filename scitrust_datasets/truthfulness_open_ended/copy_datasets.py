import pandas as pd
import json

df = pd.read_json("../../../Trustworthiness-Scientific-LLMs/data/truthfulness_openended/physics_qa_chatgpt-4o.jsonl", lines=True)

df = df.drop(columns=["difficulty"])
df = df.rename(columns={"explanation": "justification"})

df.to_json("physics_qa_chatgpt-4o.jsonl",
           orient="records",
           lines=True)
