"""
Created on Tuesday Jul 24 2018

VAE for topic model class

Code inspired in 

https://github.com/akashgit/autoencoding_vi_for_topic_models.git

@authors: Sergio Elola (sergioelola@hotmail.com) AND
          Pablo Martínez Olmos (pamartin@ing.uc3m.es)
"""


import numpy as np
import tensorflow as tf
import time
from scipy.sparse import csr_matrix
import sys

class nvlda_graph(object):

    def __init__(self, network_params, transfer_fct=tf.nn.softplus, 
                 kinit=tf.contrib.layers.xavier_initializer(),
                 learning_rate=0.002, batch_size=200,alpha=1.0):


        self.network_params = network_params
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.kinit = kinit
        '''----------------Inputs----------------'''
        # Bag of Words
        self.bow_batch = tf.placeholder(tf.float32, [self.batch_size, network_params['input_dim']])
        
        # Document embedding
        self.doc_embeds = tf.placeholder(tf.float32, [self.batch_size, network_params['embedding_dim']])
        
        self.keep_prob = tf.placeholder(tf.float32)

        '''-------Constructing Laplace Approximation to Dirichlet Prior--------------'''
        self.h_dim = int(network_params['out_dim'])
        self.a = alpha*np.ones((1 , self.h_dim)).astype(np.float32)
        self.muprior = tf.constant((np.log(self.a).T-np.mean(np.log(self.a),1)).T)
        self.varprior = tf.constant(  ( ( (1.0/self.a)*( 1 - (2.0/self.h_dim) ) ).T +
                                ( 1.0/(self.h_dim*self.h_dim) )*np.sum(1.0/self.a,1) ).T  )

        # Create autoencoder network
        self._create_VAE()
        self._create_loss_optimizer()

        
    def _recognition_network(self):
        
        #Network_architecture['nlayers'] dense layers
        #Activation indicated in tf.nn.softplus
        #Dropout before the last layer
    
        
        h_list = []
        
        h_list.append(tf.layers.dense(self.doc_embeds,units=self.network_params['hidden'][0],
            activation=self.transfer_fct,kernel_initializer=self.kinit,
            name="layer_0",reuse=None,trainable=True))
        
        for i in range(1,self.network_params['nlayers']-1):
            
            name_layer = "layer_" + str(i)
            
            h_list.append(tf.layers.dense(h_list[i-1],units=self.network_params['hidden'][i],
                activation=self.transfer_fct,kernel_initializer=self.kinit,
                name=name_layer,reuse=None,trainable=True))
            
        layer_dropout = tf.nn.dropout(h_list[-1], self.keep_prob) 
        
        z_mean = tf.contrib.layers.batch_norm(
                tf.layers.dense(inputs=layer_dropout,units=self.network_params['out_dim'],
                                activation=None, kernel_initializer=self.kinit,
                                name='layer_mean',reuse=None,trainable=True))
        
        z_log_sigma = tf.contrib.layers.batch_norm(
                tf.layers.dense(inputs=layer_dropout,units=self.network_params['out_dim'],
                                activation=None,kernel_initializer=self.kinit,
                                name='layer_log_sigma',reuse=None,trainable=True))

        return (z_mean, z_log_sigma)
    
    
    
    
    def _create_VAE(self):
        
        # Encoder
        
        self.z_mean,self.z_log_sigma = self._recognition_network()
    
        eps = tf.random_normal((self.batch_size, self.network_params['out_dim']), 
                               0, 1,dtype=tf.float32)
        
        self.samples = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma)), eps))
        
        self.norm_samples = tf.nn.softmax(self.samples)
        
        self.sigma = tf.exp(self.z_log_sigma)
        
        # Decoder
        # Beta matrix regularized by batch_normalization layer
        
        self.beta = tf.get_variable(name='beta',
                                    shape=[self.network_params['out_dim'],self.network_params['input_dim']],
                                    initializer=self.kinit,trainable=True)
        
        self.norm_beta = tf.nn.softmax(self.beta)
        
        self.layer_dropout_z_x = tf.nn.dropout(self.norm_samples,self.keep_prob)
        
        self.x_reconstr_mean = \
            tf.matmul(self.layer_dropout_z_x,tf.nn.softmax(tf.contrib.layers.batch_norm(self.beta)))+1e-10
            
        
    def _create_loss_optimizer(self):
 
        self.reconstr_loss = tf.reduce_mean(tf.reduce_sum(self.bow_batch * tf.log(self.x_reconstr_mean),1))
        
        self.latent_loss = tf.reduce_mean(0.5*( tf.reduce_sum(tf.div(self.sigma,self.varprior),1)+tf.reduce_sum(
                tf.multiply(tf.div((self.muprior - self.z_mean),self.varprior),(self.muprior - self.z_mean)),1) 
            - self.h_dim +tf.reduce_sum(tf.log(self.varprior),1)  - tf.reduce_sum(self.z_log_sigma,1)))
            
            
        self.cost = self.reconstr_loss - self.latent_loss   
        
    
        
        self.optim = \
        tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.99).minimize(-1*self.cost)
   
    def partial_fit(self, bow, embeds, session):
        _, cost,lr,KL = session.run((self.optim, self.cost,self.reconstr_loss,self.latent_loss),
                            feed_dict={self.bow_batch: bow, self.doc_embeds: embeds, self.keep_prob: .75})
        return cost,lr,KL
    
    
    def topic_prop(self, bow, embeds, session):
        
        # theta is the topic proportion vector 
        
        theta_,loglik = session.run([self.norm_samples,self.reconstr_loss],
                            feed_dict={self.bow_batch: bow, self.doc_embeds: embeds, self.keep_prob: 1.0})
        return theta_,loglik          
            
            
    def get_topic_matrix(self,session):
        
        return session.run(self.norm_beta,feed_dict={})


class nvlda(object):
    
    
    def __init__(self,network_params,dictionary,learning_rate=0.002,batch_size=200,alpha=1.0,
                 train=1,restore=0,epochs=10,verbose=1,display=1,save_it = 50):
        
        self.batch_size = batch_size
        self.network_params = network_params
        self.learning_rate = learning_rate
        self.save_it = save_it
        self.alpha = alpha
        self.dictionary = dictionary
        self.train = train
        self.restore = restore
        self.epochs = epochs
        self.verbose = verbose
        self.display = display
        self.net_file = self.network_params['net_file']
        self.beta_file = self.network_params['beta_file']

        
        self.graph = tf.Graph()
        # Creating computational graph
        
        with self.graph.as_default():
            self.vae_graph =nvlda_graph(self.network_params,learning_rate=self.learning_rate,
                                        batch_size=self.batch_size,alpha=self.alpha)  
    
    # Training method
    
    def variational_inference(self, bow, embeds):
       
        self.ntrain = bow.shape[0]
       
        with tf.Session(graph=self.graph) as session:
           
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
                
            if(self.restore == 1):
                saver.restore(session, self.net_file)
                
                self.topic_matrix = np.load(self.beta_file)['beta']
                
                print("Model restored.")
                
            else:
                print('Initizalizing Variables ...')
                tf.global_variables_initializer().run()
                
            if(self.train == 1):
            
                n_batches = int(np.floor(self.ntrain/self.batch_size))
                start_time_lda_vae = time.time()
                
                print('Training the model ...')
                
                for epoch in range(self.epochs):
                    
                    random_perm = np.random.permutation(range(self.ntrain))
                    
                    avg_loss = 0.
                    avg_KL = 0.
                    avg_elbo = 0.
                    
                    for i in range(n_batches):
                        
                        
                        idx_batch = self.next_batch(random_perm,i)   
            
                        # Fit training using batch data
                        cost,lr,KL = self.vae_graph.partial_fit(csr_matrix.todense(bow[idx_batch,:]),
                                        embeds[idx_batch,:], session) 
                       
                        # Compute average loss
                        avg_loss += lr / self.ntrain * self.batch_size
                        avg_KL += KL / self.ntrain * self.batch_size
                        avg_elbo += cost / self.ntrain * self.batch_size
                        
                        if np.isnan(avg_elbo):
                            print ('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                            sys.exit()                    
                            
                    if (self.verbose == 1 and epoch%self.display==0):
                        
                        elapsed_time = time.time()-start_time_lda_vae
                        print("Elapsed time %4.4f, Epoch %d, elbo %4.4f, loss %4.4f, KL %4.4f" 
                          %(elapsed_time,epoch,avg_elbo,avg_loss,avg_KL))    
                    
                    if (epoch%self.save_it==0 and self.save_it != -1):
                    
                        saver.save(session, self.net_file) 
            
                        # Save topic_matrix
                        
                        self.topic_matrix = self.vae_graph.get_topic_matrix(session)                  
            
                        np.savez(self.beta_file,beta = self.topic_matrix)
                        
                        
                        
                print('Training Finished. Saving results ...')                     
                
                saver.save(session, self.net_file) 
            
                # Save topic_matrix
            
                self.topic_matrix = self.vae_graph.get_topic_matrix(session)
                
                np.savez(self.beta_file, beta = self.topic_matrix)
                
                
                
            session.close()
            
            
    def topic_proportions(self,bow_matrix, embeds):
        
        #We assume up to batch_size documents to compute their proportions
        
        with tf.Session(graph=self.graph) as session:
        
            saver = tf.train.Saver()
            
            saver.restore(session, self.net_file)
            
            n_batches = int(np.floor(self.ntrain/self.batch_size))

            order = np.arange(bow_matrix.shape[0])

            topic_prop = np.zeros([bow_matrix.shape[0],self.network_params['out_dim']])

            loglik = 0.0            
            
            for i in range(n_batches):
                
                                
                idx_batch = self.next_batch(order,i)     
                
                bow_batch = np.zeros([self.batch_size,self.network_params['input_dim']])
                embed_batch = np.zeros([self.batch_size,self.network_params['embedding_dim']])
                #We complete with zeros (last batch only)
            
                bow_batch[:idx_batch.shape[0],:] = csr_matrix.todense(bow_matrix[idx_batch,:])  
                embed_batch[:idx_batch.shape[0],:] = embeds[idx_batch,:]
                
                topic_prop[idx_batch,:],ll = self.vae_graph.topic_prop(bow_batch,embed_batch,session)
                
                loglik += ll / self.ntrain * self.batch_size

            session.close()
            
            
        return topic_prop,loglik

    # Next mini-batch index generator
    
    def next_batch(self,idx_vector,index_batch):
        return idx_vector[index_batch*self.batch_size:(index_batch*self.batch_size+self.batch_size)]
    
    # Display dominant words per topic
    
    def display_topics(self,nwords):
        
        for topic_idx, topic in enumerate(self.topic_matrix):
            print ("Topic %d:" % (topic_idx))
            print (" ".join([self.dictionary[i] for i in topic.argsort()[:-nwords - 1:-1]]))

               

                    
                    
                    