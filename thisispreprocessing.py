#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 01:11:23 2023

@author: lucia

this program:
    pulls .csv documents into pandas df (https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial)
    creates document for gensim model
    trains LDA model
    reuploads model
    
"""

# 0. setup
import pandas as pd  # For data handling
from time import time

import spacy  # For preprocessing
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

# Install spacy-transformers
#pip install spacy[transformers]

# Download the traditional spacy english language large model
#python -m spacy download en_core_web_lg 

# Download the spacy transformer (roberta-base) english model
#python -m spacy download en_core_web_trf



# 1. create dataframe from csv corpus

df = pd.read_csv('/Users/lucia/Documents/schul/04 year4/thesis/quickass e-flux corpus - absolutely this one.csv')
print(df.head())


# 2. cleaning

#load spacy model, using roBERTa transformers (more accurate than _sm or _lg)
nlp = spacy.load('en_core_web_trf') 


def cleaning(doc):
    #creates a list of tokens in doc that
    #are not named entities, are alpha, and 
    #are not stop words, then lemmatizes
    txt=[]
    for token in doc:
        if token.ent_iob_=='O':
            if token.is_stop==False:
                if token.text.isalpha()==True:
                    txt.append(token.lemma_)
    #return txt
    return ' '.join(txt)


#call cleaning on docs in df
t = time()
txt = [cleaning(doc) for doc in nlp.pipe(df['text'], batch_size=500)]
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))



# put cleaned data in new clean df
df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
df_clean.shape
df_clean.to_csv('/Users/lucia/Documents/schul/04 year4/thesis/cleancorpus_nlptrfpls_bigrams.csv')



# ---- bigrams
from gensim.models.phrases import Phrases, Phraser
sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]

df_bigram = pd.DataFrame({'cleaned w bigrams': sentences})
df_bigram.to_csv('/Users/lucia/Documents/schul/04 year4/thesis/cleancorpus_nlptrf_bigrams.csv')

workdf = df_bigram.copy()
print(workdf.head())
print(df_bigram.head())


for listitem in workdf['cleaned w bigrams']:
    for word in listitem:
        if word in ["work", "artist", "exhibition", 'work', 'artist', 'exhibition']:
                listitem.remove(word)
                
print(workdf)
workdf.to_csv('/Users/lucia/Documents/schul/04 year4/thesis/cleancorpus_workdf.csv')
 
                
 

#df_clean.to_csv('/Users/lucia/Documents/schul/04 year4/thesis/cleancorpus_nlptrfpls.csv')


# 3. build and train model
from time import time
from gensim.models import Word2Vec
#model = Word2Vec(sentences=sentences)
# or # model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
#model.save("word2vec.model")

#create model, not initialized with sentences
w2v_model = Word2Vec(min_count=2, 
                     window=2,
                    vector_size=300,
                    sample=6e-5, 
                    alpha=0.03, 
                    min_alpha=0.0007, 
                    negative=0)

#build model, initialized w sentences
t = time()
w2v_model.build_vocab(sentences, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

#train initialized model
t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

#save model
w2v_model.save("word2vec.model")


# 4. test model
w2v_model.wv.most_similar(positive=["art"])
w2v_model.wv.most_similar(positive=["collaboration"])
w2v_model.wv.most_similar(positive=["viewer"])
w2v_model.wv.most_similar(positive=["place"])
w2v_model.wv.most_similar(positive=["medium"])
w2v_model.wv.most_similar(positive=["institution"])
w2v_model.wv.most_similar(positive=["house"])
w2v_model.wv.most_similar(positive=["existence"])
w2v_model.wv.most_similar(positive=["ecology"])
w2v_model.wv.most_similar(positive=["museum"])


w2v_model.wv.doesnt_match(['museum', 'exhibition', 'war'])
w2v_model.wv.most_similar(positive=["war", "beauty"], negative=["painting"], topn=3)




# 5. visualize

from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

x_vals, y_vals, labels = reduce_dimensions(w2v_model)

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')

    else:
        plot(data, filename='word-embedding-plot.html')
        


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly

plot_function(x_vals, y_vals, labels)






# --- scratch material

#create list of named entities
#items = [x.text for x in tokens.ents]


# create list of words
#tokens = nlp(''.join(str(df.text.tolist())))
    
#create list of non named entities
#nonents = [x.text for x in tokens if x.ent_iob_=='O']

#ents_remove = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['text'])
#nonalpha_remove = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['text'])

   #txt = [token in doc if token.ent_iob_=='O']
   
'''          
        
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.text for token in doc if token.ent_iob_=='O']
    for token in txt:
        token.lemma_ for token in doc if not token.is_stop
    txt = [token.lemma_ for token in doc if not token.is_stop]
    
    txt = [token in txt if x.isalpha!=False ]
    

    return ' '.join(txt)

#nonalpha_remove = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['text'])
'''




