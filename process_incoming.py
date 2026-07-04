import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3:567m",
        "input" : text_list
    })

    embedding = r.json()['embeddings'] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model" : "gemma3",
        "prompt" : prompt,
        "stream": False
    })

    response = r.json()
    return response

df = joblib.load("embeddings.joblib")


incomming_querry = input("Ask a question: ")
question_embedding = create_embedding([incomming_querry])[0]
    
similarities = cosine_similarity(np.vstack(df["Embedding"]), [question_embedding]).flatten()
#print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
#print(max_indx)
new_df = df.loc[max_indx]
#print(new_df[["number","title", "text"]])

prompt = f'''I am teaching web development using Sigma web development course. Here are video chunks containing video title, video number,start time in second,end time in seconds, the text at that time:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
____________________________________________________________________________________________________________________________
{incomming_querry}
User asked this question related to video chunks, you have to answer where and how much content is taughted where (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course.
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)