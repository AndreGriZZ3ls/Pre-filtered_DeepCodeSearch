#!/usr/bin/env python

__author__    = "Andre Warnecke"
__copyright__ = "Copyright (c) 2022, André Warnecke"

#Citation for DeepCS, which is called/executed in the context of this program:
'''bibtex
@inproceedings{gu2018deepcs,
  title={Deep Code Search},
  author={Gu, Xiaodong and Zhang, Hongyu and Kim, Sunghun},
  booktitle={Proceedings of the 2018 40th International Conference on Software Engineering (ICSE 2018)},
  year={2018},
  organization={ACM}
}
'''

import os
import sys
import math
import codecs
import numpy as np
from tqdm import tqdm

from DeepCSKeras import data_loader
from DeepCSKeras import configs
from DeepCSKeras.utils import convert, revert, normalize

class IndexCreator:
    def __init__(self, args, conf = None):
        self.data_path    = args.data_path
        self.dataset_path = args.data_path + args.dataset + '/'
        self.data_params  = conf.get('data_params', dict())
        self.index_type   = args.index_type
        self.dataset      = args.dataset
        self.index_dir    = args.index_dir
        self.methname_vocab   = data_loader.load_pickle(self.dataset_path + conf['data_params']['vocab_methname'])
        self.token_vocab      = data_loader.load_pickle(self.dataset_path + conf['data_params']['vocab_tokens'])
        self.chunk_size       = 2000000

    def replace_synonyms(self, word):
        word = ' ' + word + ' '
        word = word.replace(' read ', 'load').replace(' write', 'store').replace('save', 'store').replace(' dump', 'store')
        word = word.replace('object', 'instance').replace(' quit', 'exit').replace('terminate', 'exit').replace(' leave', 'exit')
        word = word.replace('pop ', 'delete').replace('remove', 'delete').replace('begin', 'start').replace('run ', 'execute')
        word = word.replace(' halt', 'stop').replace('restart', 'continue').replace('append', 'add').replace('push ', 'add')
        word = word.replace('null ', 'none').replace('method', 'function').replace('concat', 'combine').replace(' break ', 'exit')
        return word.replace(' implements ', 'extends').replace('runnable', 'executable').strip()

    def load_data(self):
        assert os.path.exists(self.dataset_path + self.data_params['use_methname']), f"Method names of real data not found."
        assert os.path.exists(self.dataset_path + self.data_params['use_tokens']),   f"Tokens of real data not found."
        #methname_indices = None #####
        #token_indices    = None #####
        methname_indices = data_loader.load_hdf5(self.dataset_path + self.data_params['use_methname'], 0, -1)
        token_indices    = data_loader.load_hdf5(self.dataset_path + self.data_params['use_tokens'],   0, -1)
        if   self.index_type == "word_indices": return methname_indices, token_indices
        elif self.index_type == "inverted_index":
            print("Translating methname and token word indices back to natural language...   Please wait.")
            inverted_methname_vocab = dict((v, k) for k, v in self.methname_vocab.items())
            inverted_token_vocab    = dict((v, k) for k, v in self.token_vocab.items())
            fm = lambda lst: [inverted_methname_vocab.get(i, 'UNK') for i in lst]
            ft = lambda lst: [inverted_token_vocab.get(   i, 'UNK') for i in lst]
            methnames = list(map(fm, methname_indices))
            tokens    = list(map(ft, token_indices))
            return methnames, tokens
            
        #print(type(self.methname_vocab.items()))
        #print(self.methname_vocab.items())

    def safe_index(self, index):
        if self.index_type == "word_indices": return
        index_path = self.data_path + self.index_dir + '/'
        index_file = self.index_type + '.pkl'
        #os.makedirs(index_path, exist_ok = True)
        assert os.path.exists(index_path + index_file), (
                              f"File for index storage not found. Please create an (empty) file named {index_file} in {index_path}")
        data_loader.save_pickle(index_path + index_file, index)

    def load_index(self):
        if self.index_type == "word_indices": return self.load_data()
        index_path = self.data_path + self.index_dir + '/'
        index_file = self.index_type + '.pkl'
        assert os.path.exists(index_path + index_file), f"Index file {index_file} not found at {index_path}"
        return data_loader.load_pickle(index_path + index_file)
                    
    def add_to_index(self, index, lines, stopwords):
        print("Adding lines to the index...   Please wait.")
        for i, line in enumerate(tqdm(lines)):
            for word in line:
                if word in stopwords: continue
                word = self.replace_synonyms(word)
                if word not in index:
                    index[word] = [i]
                else:
                    index[word].append(i)

    def create_index(self, stopwords):
        if self.index_type == "word_indices": print("Nothing to be done."); return
        methnames, tokens = self.load_data()
        index = dict()
        if self.index_type == "inverted_index":
            self.add_to_index(index, methnames, stopwords)
            self.add_to_index(index, tokens   , stopwords)
        #items = list(index.items())
        #for i in range(0, 10):
        #print(items[0])
        self.safe_index(index)
