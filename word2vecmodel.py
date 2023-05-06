#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:34:53 2023

@author: lucia

data preprocessed w Preprocessed.py
word2vec

https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-download-auto-examples-tutorials-run-word2vec-py
https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial

"""

import multiprocessing
from time import time
from gensim.models import Word2Vec
from preprocessing import sentences

cores = multiprocessing.cpu_count() # Count the number of cores in a computer


w2v_model = Word2Vec(min_count=20,
                     window=2,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)

#
t = time()
w2v_model.build_vocab(preprocessing.sentences, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

#train model
t = time()
w2v_model.train(preprocessing.sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))