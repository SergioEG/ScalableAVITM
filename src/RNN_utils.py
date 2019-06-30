
"""
Grouping all parsers specifications here

@authors: Sergio Elola (sergioelola@hotmail.com) AND
          Pablo Martínez Olmos (pamartin@ing.uc3m.es)
"""

import argparse
import pickle
from pandas import read_csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import logging
logging.getLogger().setLevel(logging.INFO)

def parse_args():
    desc = 'Variational Inference for Topic Models based on Dense Layers'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=200, help='Size of the batches')
    parser.add_argument('--epochs',type=int,default=10, 
                        help='Number of epochs of the simulations')
    parser.add_argument('--train', type=int,default=1, help='Training model flag')
    parser.add_argument('--verbose', type=int,default=1, help='Verbose option flag')
    parser.add_argument('--save_it', type=int,default=1000, 
                        help='Save variables every save_it iterations')
    parser.add_argument('--restore', type=int,default=0, 
                        help='To restore session, to keep training or evaluation') 
    parser.add_argument('--save_file', type=str, default='abcnews', 
                        help='Save file name (will be extended with other conf. parameters)')
    parser.add_argument('--corpus', type=str, default='abcnews', 
                        help='Corpus. Instructions to read each corpus are given in utils.py')
    parser.add_argument('--ntopics', type=int, default=10, help='Number of topics')   
    parser.add_argument('--display', type=int, default=1, 
                        help='If verbose, print results every display epochs')    
    parser.add_argument('--nlayers', type=int, default=2, 
                        help='Number of dense layers for the inference network')
    parser.add_argument('--hidden_list','--list', nargs='+', 
                        help='List of hidden units per layer',default=100) 
                        #If only one number, same number of units perlayer!
    parser.add_argument('--rate', type=float,default=0.005, help='Learning rate')
    
    parser.add_argument('--vocabulary_size', type=int,default=10000, help='Vocabulary size')
    
    parser.add_argument('--alpha', type=float,default=1.0, help='Dirich hiper.')
    
    return parser.parse_args()



def get_corpus(corpus_name,nwords):
 
    bow = None
    dictionary = None
    
    logging.info(f"Getting corpus and embeds of {corpus_name}")
    
    if(corpus_name == 'FECYT'): 
        
        documents = list(pickle.load(open( "data/df_proyectosFECYT.pkl", "rb" ) )['LEMAS_UC3M'])
        
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=nwords, stop_words='english')        

        bow = vectorizer.fit_transform(documents)
        
        dictionary = vectorizer.get_feature_names()   
    
        embeds = np.load("data/embeds_FECYT_256_LSTM.npy")

    if(corpus_name == 'abcnews'):
        
        data = read_csv('Data/abcnews-date-text.csv', error_bad_lines=False)
        
        documents = data[['headline_text']].values.reshape(-1).tolist()
        
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=nwords, stop_words='english')
        
        bow = vectorizer.fit_transform(documents)
        
        dictionary = vectorizer.get_feature_names()
        
        embeds = np.load("data/embeds_abcnews_512.npy")
        
    
    logging.info(f"Corpus and embeds obtained")
    return documents, bow,dictionary, embeds

        
def display_topics(lda_matrix, dictionary, no_top_words):
    for topic_idx, topic in enumerate(lda_matrix):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([dictionary[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def next_batch(idx_vector,index_batch,batch_size):
    return idx_vector[index_batch*batch_size:(index_batch*batch_size+batch_size)]
    