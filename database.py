import faiss
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

def build_index(vectors: np.array, dimension):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

def save_index(index, filepath:str):
    faiss.write_index(index, filepath)


def load_index(filepath:str):
    return faiss.read_index(filepath)

def retrieve(quary:str, top_k=5, *args):

    return None

def transform(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        return cls_vector

def build(text):
    arrays = []
    for line in text:
        vector = transform(line)
        arrays.append(vector)
    # index = build_index(arrays, 0)
    # distances, indices = index.search(transform("what is examples"))
    # print(distances)
    # print(indices)
    index = faiss.IndexFlatL2(768)
    index.add(arrays[0])
    index.add(arrays[1])
    index.add(arrays[2])
    index.add(arrays[3])

    d,i = index.search(transform("what is examples"), 2)
    print(d)
    print(i)



text = [
    "Example is a word",
    "Example is from English",
    "Do I know examples",
    "I love the base"
]

build(text)