from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import os,sys
from pathlib import Path
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
from util import Configure

# BERT_pretrained_path=pathlib.Path(curr_path) / pathlib.Path("BERT_pretrained")
# model_name=Configure.get("Embedding", "SBERT_model")

model_path=Path(curr_path)/"Embedding_models"/Configure.get("Embedding", "SBERT_model")

model = SentenceTransformer(str(model_path))


def sentence_embeddings(sentences:list=[],normalization=True):
    
    sentence_embeddings = model.encode(sentences=sentences,normalize_embeddings=normalization,show_progress_bar=False)
    return sentence_embeddings # np.array shape = [number of sentences, dim of vector]
