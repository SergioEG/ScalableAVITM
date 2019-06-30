"""

Main File. Code inspired on 

https://github.com/akashgit/autoencoding_vi_for_topic_models.git

@authors: Sergio Elola (sergioelola@hotmail.com) AND
          Pablo Martínez Olmos (pamartin@ing.uc3m.es)
"""

'''--------------Libraries---------------'''
import numpy as np
import pandas as pd
import RNN_utils
import RNN_neural_lda
import os

import logging
logging.getLogger().setLevel(logging.INFO)


#%%
'''--------------Parsing and Configurations---------------'''
args = RNN_utils.parse_args()
if args is None:
    logging.info("NO ARGS")
    exit()
print(args, '\n')
#%%

## Creating folder to save TF variables

save_file = args.corpus + '_ntopic_' + str(args.ntopics)   \
+ '_nwords_' + str(args.vocabulary_size)
            
if not os.path.exists('./Saved_Results_Prod/' + save_file):
    os.makedirs('./Saved_Results_Prod/' + save_file)
    
# NOTE: Changed named as it was too large
network_file_name='./Saved_Results_Prod/' + save_file + '/' + save_file + '.ckpt'
beta_file_name='./Saved_Results_Prod/' + save_file + '/' + save_file +'.npz'  


## Loading corpus (this would be replace by direct minibatch access in future versions)

original_documents, bow, dictionary, embeds = RNN_utils.get_corpus(args.corpus,args.vocabulary_size)

n_samples_tr = np.shape(bow)[0]

## Creating the VAE-Topic class

if(type(args.hidden_list) == int):
    
    hid_list =[args.hidden_list for i in range(args.nlayers)]

else:
    hid_list=[int(x) for x in args.hidden_list]
    
    
network_params = dict(nlayers = args.nlayers,
                          hidden=hid_list,
                          input_dim = args.vocabulary_size,
                          embedding_dim = embeds.shape[1],
                          out_dim = args.ntopics,
                          net_file = network_file_name,
                          beta_file = beta_file_name)




nlda_object = RNN_neural_lda.nvlda(network_params,dictionary,learning_rate=args.rate,
                   batch_size=args.batch_size,alpha=args.alpha,train=args.train,
                   restore=args.restore,epochs=args.epochs,verbose=args.verbose,
                   display=args.display,save_it=args.save_it)

# Training
nlda_object.variational_inference(bow, embeds)

# Getting topic proportions
topics,loglik = nlda_object.topic_proportions(bow, embeds)


# Displaying topics
nlda_object.display_topics(20)


# GET TOPIC PROPORTIONS OF EXAMPLE SPECIFIC DOCS
doc_nums = [10,17,28,79,120,168,196,204,292]

topic_props = topics[doc_nums,:]
df_topic_props = pd.DataFrame(a, index=doc_nums)

docum_text = [original_documents[i] for i in doc_nums]
df_docum_text = pd.DataFrame(c, index=doc_nums)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_docum_text)
    print(df_topic_props.round(decimals=3))
