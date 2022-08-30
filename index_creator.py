#!/usr/bin/env python

__author__    = "Andre Warnecke"
__copyright__ = "Copyright © 2022, André Warnecke"

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
        self.data_path   = args.data_path + args.dataset + '/'
        self.data_params = conf.get('data_params', dict())
        self.index_type  = args.index_type
        self.dataset     = args.dataset
        self.filtered_dataset = args.filtered_dataset
        self.methname_vocab   = data_loader.load_pickle(self.data_path + conf['data_params']['vocab_methname'])
        self.token_vocab      = data_loader.load_pickle(self.data_path + conf['data_params']['vocab_tokens'])
        self.chunk_size       = 2000000

    def load_data(self):
        assert os.path.exists(self.data_path + self.data_params['use_methname']), f"Method names of real data not found."
        assert os.path.exists(self.data_path + self.data_params['use_tokens']),   f"Tokens of real data not found."
        #methname_indices = None #####
        #token_indices    = None #####
        methname_indices = data_loader.load_hdf5(self.data_path + self.data_params['use_methname'], 0, -1)
        token_indices    = data_loader.load_hdf5(self.data_path + self.data_params['use_tokens'],   0, -1)
        if   self.index_type == "word_indices": return methname_indices, token_indices
        elif self.index_type == "inverted_index":
            methnames, tokens = [], []
            for index in methname_indices:
                methnames.append(revert(self.methname_vocab, index))
            for index in token_indices:
                tokens.append(   revert(self.token_vocab,    index))
            """print(type(methname_indices[0]))
            for elem in methnames:
                print(elem)
            for elem in tokens:
                print(elem)"""
            return methnames, tokens
            
        #print(type(self.methname_vocab.items()))
        #print(self.methname_vocab.items())

    def safe_index(self):
        if self.index_type == "word_indices": return
        os.makedirs(self.data_path, exist_ok = True)
        # TODO save to file

    def load_index(self):
        if self.index_type == "word_indices": return self.load_data()

    def create_index(self):
        if self.index_type == "word_indices": print("Nothing to be done."); return
        methnames, tokens = self.load_data()
        
        self.safe_index(index)

# Working code to load codebase (raw methods):
"""codes = codecs.open(self.data_path + self.data_params['use_codebase'], encoding='utf8', errors='replace').readlines()
codebase = []
#for i in tqdm(range(0,len(codes), chunk_size)):
for i in tqdm(range(0, self.chunk_size, self.chunk_size)):
    codebase.append(codes[i : i + self.chunk_size])""" #
