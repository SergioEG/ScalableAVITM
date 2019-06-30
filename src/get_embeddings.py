"""
Created on Mon Jun 10 16:46:07 2019

Code for generating Document Embeddings with Flair. This one in particular 
uses RNN and a hidden layer size of 512.

@authors: Sergio Elola (sergioelola@hotmail.com) AND
          Pablo Martínez Olmos (pamartin@ing.uc3m.es)
"""

# Make sure to have flair, torch and pandas installed

import numpy as np
from pandas import read_csv
import pickle
    
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings, Sentence

data = read_csv('data/abcnews-date-text.csv', error_bad_lines=False)
documents = data[['headline_text']].values.reshape(-1).tolist()
# documents = list(pickle.load(open( "./corpus/df_proyectosFECYT.pkl", "rb" ) )['LEMAS_UC3M'])

glove_embedding = WordEmbeddings('glove')
document_embeddings = DocumentRNNEmbeddings([glove_embedding], hidden_size=512)
embeddings = []

count = 0

try: 
    for document in documents:
        count += 1
        sentence = Sentence(document)
    
        document_embeddings.embed(sentence)
    
        embeddings.append(sentence.get_embedding().tolist())
        
        if (count % 1000 == 0): print(count)

finally: # In case an error occurs before finish, we store previous results
    embedings_array = np.array(embeddings)
    np.save("embeds_abcnews_512_2.npy", embedings_array)