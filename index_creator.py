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
from collections import Counter

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
        self.apiseq_vocab     = data_loader.load_pickle(self.dataset_path + conf['data_params']['vocab_apiseq'])
        self.chunk_size       = 2000000

    def replace_synonyms(self, word):
        word = ' ' + word + ' '
        word = word.replace(' read ', 'load').replace(' write', 'store').replace('save', 'store').replace(' dump', 'store')
        word = word.replace('object', 'instance').replace(' quit', 'exit').replace('terminate', 'exit').replace(' leave', 'exit')
        word = word.replace('pop ', 'delete').replace('remove', 'delete').replace('begin', 'start').replace('run ', 'execute')
        word = word.replace(' halt', 'stop').replace('restart', 'continue').replace('append', 'add').replace('push ', 'add')
        word = word.replace('null ', 'none').replace('method', 'function').replace('concat', 'combine').replace(' break ', 'exit')
        word = word.replace(' for ', 'loop').replace(' foreach ', 'loop').replace(' while ', 'loop').replace(' iterat ', 'loop')
        word = word.replace('tinyint ', 'int').replace(' smallint ', 'int').replace(' bigint ', 'int').replace(' shortint ', 'int')
        word = word.replace('longint ', 'int').replace(' byte ', 'int').replace(' long ', 'int').replace(' short ', 'int')
        word = word.replace('integer ', 'int').replace(' double ', 'float').replace(' long ', 'float').replace(' decimal ', 'float')
        word = word.replace('real ', 'float')
        return word.replace(' implements ', 'extends').replace('runnable', 'executable').replace(' array ', '[]').replace(' arrays ', '[]').strip()

    def load_data(self):
        assert os.path.exists(self.dataset_path + self.data_params['use_methname']), f"Method names of real data not found."
        assert os.path.exists(self.dataset_path + self.data_params['use_tokens']),   f"Tokens of real data not found."
        assert os.path.exists(self.dataset_path + self.data_params['use_apiseq']),   f"API sequences of real data not found."
        methname_indices = data_loader.load_hdf5(self.dataset_path + self.data_params['use_methname'], 0, -1)
        token_indices    = data_loader.load_hdf5(self.dataset_path + self.data_params['use_tokens'],   0, -1)
        apiseq_indices   = data_loader.load_hdf5(self.dataset_path + self.data_params['use_apiseq'],   0, -1)
        if   self.index_type == "word_indices": return methname_indices, token_indices
        elif self.index_type == "inverted_index":
            print("Translating methname, token and api sequence word indices back to natural language...   Please wait.")
            inverted_methname_vocab = dict((v, k) for k, v in self.methname_vocab.items())
            inverted_token_vocab    = dict((v, k) for k, v in self.token_vocab.items())
            inverted_apiseq_vocab   = dict((v, k) for k, v in self.apiseq_vocab.items())
            fm = lambda lst: [inverted_methname_vocab.get(i, 'UNK') for i in lst]
            ft = lambda lst: [inverted_token_vocab.get(   i, 'UNK') for i in lst]
            fa = lambda lst: [inverted_apiseq_vocab.get(  i, 'UNK') for i in lst]
            methnames = list(map(fm, methname_indices))
            tokens    = list(map(ft, token_indices))
            apiseqs   = list(map(fa, apiseq_indices))
            return methnames, tokens, apiseqs

    def safe_index(self, index):
        if self.index_type == "word_indices": return
        index_path = self.data_path + self.index_dir + '/'
        index_file = self.index_type + '.pkl'
        #os.makedirs(index_path, exist_ok = True)
        assert os.path.exists(index_path + index_file), (
                              f"File for index storage not found. Please create an (empty) file named {index_file} in {index_path}")
        data_loader.save_pickle(index_path + index_file, index)
        print(f"Index successfully saved to: {index_path}{index_file}")

    def load_index(self):
        if self.index_type == "word_indices": 
            methnames, tokens, irrelevant = self.load_data()
            return methnames, tokens
        index_path = self.data_path + self.index_dir + '/'
        index_file = self.index_type + '.pkl'
        assert os.path.exists(index_path + index_file), f"Index file {index_file} not found at {index_path}"
        print(f"Loading index from: {index_path}{index_file}")
        return data_loader.load_pickle(index_path + index_file)
                    
    def add_to_index(self, index, lines, stopwords):
        print("Adding lines to the index...   Please wait.")
        if stopwords:
            f = lambda word: bool(word in stopwords)
        else:
            f = lambda word: bool(word != '[]')
        for i, line in enumerate(tqdm(lines)):
            for word in line:
                #if       stopwords and word in stopwords: continue
                #elif not stopwords and word != '[]':      continue
                if map(f, word): continue
                word = self.replace_synonyms(word)
                if word in index:
                    #index[word].append(i)
                    index[word][i] += 1
                else:
                    #index[word] = [i]
                    cnt = Counter()
                    cnt[i] += 1
                    index[word] = cnt

    def create_index(self, stopwords):
        if self.index_type == "word_indices": print("Nothing to be done."); return
        methnames, tokens, apiseqs = self.load_data()
        index = dict()
        if self.index_type == "inverted_index":
            self.add_to_index(index, methnames, stopwords)
            self.add_to_index(index, tokens   , stopwords)
            self.add_to_index(index, apiseqs  , None)
            number_of_code_fragments = len(methnames)
            for line_counter in tqdm(index.keys()):
                lines = list(line_counter.keys()) # deduplicated list of those code fragments
                idf   = math.log10(number_of_code_fragments / len(lines)) # idf = log10(N/df)
                for line_nr in lines:
                    line_counter[line_nr] = idf * math.log(1 + line_counter[line_nr]) # tf-idf = idf * log10(1 + tf)
        
        self.safe_index(index)
