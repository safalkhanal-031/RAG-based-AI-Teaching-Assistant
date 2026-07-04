import pandas as pd
import numpy as np
import requests
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3:567m",
        "input" : text_list
    })

    embedding = r.json()['embeddings'] 
    return embedding

jsons = os.listdir("jsons")
chunk_id = 0
my_dict = []

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    texts = [c["text"] for c in content["chunks"]]
    embeddings = create_embedding(texts)
    for i, chunk in enumerate(content["chunks"]):
        chunk["Chunk_id"] = chunk_id
        chunk["Embedding"] = embeddings[i]
        chunk_id += 1
        my_dict.append(chunk)

df = pd.DataFrame.from_records(my_dict)
joblib.dump(df, "embeddings.joblib")


