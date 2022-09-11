#!/usr/bin/env python

"""This program is able to create an index for given data that are compatible with DeepCS (a open source code search engine
   utilizing a deep learning neural net) and use this index to pre-select search candidates, increasing DeepCS performance."""

__author__    = "Andre Warnecke"
__copyright__ = "Copyright (c) 2022, Andre Warnecke"

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
import argparse
import operator
import traceback
from tqdm import tqdm
from collections import Counter
from nltk.stem import PorterStemmer

from index_creator import IndexCreator
import index_creator

from DeepCSKeras import data_loader, configs, models
from DeepCSKeras import main as deepCS_main
from DeepCSKeras.utils import convert, revert


def parse_args():
    parser = argparse.ArgumentParser("Generate Index or perform pre-filtered deep code search")
    parser.add_argument("--index_dir",  type=str, default="indices",      help="index directory")
    parser.add_argument("--dataset",    type=str, default="github",       help="dataset name")
    parser.add_argument("--data_path",  type=str, default='./DeepCSKeras/data/',       help="working directory")
    parser.add_argument("--model",      type=str, default="JointEmbeddingModel",       help="DeepCS model name")
    parser.add_argument("--mode", choices=["create_index","search"], default='search', help="The mode to run:"
                        " The 'create_index' mode constructs an index of specified type on the desired dataset; "
                        " The 'search' mode filters the dataset according to given query and index before utilizing "
                        " DeepCS with a trained model to search pre-selected for the K most relevant code snippets.")
    parser.add_argument("--index_type", choices=["word_indices","inverted_index"], default="inverted_index", help="Type of index "
                        " to be created or used: The 'word_indices' mode utilizes parts of the dataset already existing for DeepCS "
                        " (simple but not usable for more accurete similarity measurements. For each meaningful word the "
                        " 'inverted_index' stores IDs of code fragment that contain it. ")
    parser.add_argument("--similarity_mode", choices=["lexical","idf","tf_idf"], default='tf_idf', help="The metric used for "
                        " similarity calculation between query and code fragments: The 'lexical' similarity mode measures "
                        " the amount of words that query and code fragment have in common (rather simple and inaccurate). "
                        " 'idf' stands for inverted document frequency and measures the informativeness of each word "
                        " (incompatible with word_indices as index type).")
    parser.add_argument("--less_memory_mode", action="store_true", default=False, help="If active the program will load some files "
                        "just (partial) pre-filtered after each query input instead of complete in the beginning (slower).")
    return parser.parse_args()

if __name__ == '__main__':
    args            = parse_args()
    config          = getattr(configs, 'config_' + args.model)()
    data_path       = args.data_path + args.dataset + '/'
    index_type      = args.index_type
    similarity_mode = args.similarity_mode
    less_memory     = args.less_memory_mode
    methname_vocab  = data_loader.load_pickle(data_path + config['data_params']['vocab_methname'])
    token_vocab     = data_loader.load_pickle(data_path + config['data_params']['vocab_tokens'])
    indexer         = IndexCreator(args, config)
    stopwords       = set("a,about,after,also,an,and,another,are,around,as,at,be,because,been,before,being,between,both,but,by,came,can,come,could,did,do,does,each,every,get,got,had,has,have,he,her,here,him,himself,his,how,into,it,its,just,like,make,many,me,might,more,most,much,must,my,never,no,now,of,on,only,other,our,out,over,re,said,same,see,should,since,so,some,still,such,take,than,that,the,their,them,then,there,these,they,this,those,through,to,too,under,up,use,very,want,was,way,we,well,were,what,when,where,which,who,will,with,would,you,your".split(','))

    _codebase_chunksize = 2000000
    n_threads = 8 # number of threads for parallelization of less performance intensive program parts

    if args.mode == 'create_index':
        indexer.create_index(stopwords)

    elif args.mode == 'search':
        engine = deepCS_main.SearchEngine(args, config)

        ##### Define model ######
        model = getattr(models, args.model)(config) # initialize the model
        model.build()
        model.summary(export_path = f"./output/{args.model}/")
        
        optimizer = config.get('training_params', dict()).get('optimizer', 'adam')
        model.compile(optimizer = optimizer)
        
        assert config['training_params']['reload'] > 0, "Please specify the number of the optimal epoch checkpoint in config.py"
        engine.load_model(model, config['training_params']['reload'], f"./DeepCSKeras/output/{model.__class__.__name__}/models/")
        if not less_memory:
            full_code_reprs = data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], -1)
            full_codebase   = data_loader.load_codebase(  data_path + config['data_params']['use_codebase'], -1)
        vocab = data_loader.load_pickle(data_path + config['data_params']['vocab_desc'])
        
        if index_type == "word_indices":
            methnames, tokens = indexer.load_index()
        else:
            index = indexer.load_index()
            codebase, codereprs = [], []
            number_of_code_fragments = len(codecs.open(data_path + config['data_params']['use_codebase'], encoding='utf8',errors='replace').readlines())
        while True:
            try:
                query     =     input('Input Query: ')
                n_results = int(input('How many results? '))
            except Exception:
                print("Exception while parsing your input:")
                traceback.print_exc()
                break
            #engine._code_reprs = full_code_reprs
            query = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
            query_list = list(set(query.split(' ')) - stopwords)
            len_query_without_stems = len(query_list)
            porter = PorterStemmer()
            tmp = []
            for word in query_list:
                word_stem = porter.stem(word)
                if word != word_stem:
                    tmp.append(porter.stem(word)) # include stems of query words
            query_list += tmp
            query_list = [indexer.replace_synonyms(w) for w in query_list]
            print(f"Query without stop words and possibly with replaced synonyms and added word stems: {query_list}")
            result_line_numbers = set()
            print("Processing...  Please wait.")
            if index_type == "word_indices":
                query_index_for_methnames = set([methname_vocab.get(w, 0) for w in query_list]) # convert user input to word indices
                query_index_for_tokens    = set([token_vocab.get(   w, 0) for w in query_list])
                min_common = len_query_without_stems / 2 + len_query_without_stems % 2
                for i in range(0, len(methnames)):
                    if len(query_index_for_methnames & set(methnames[i])) >= min_common:
                    #if not query_index_for_methnames.isdisjoint(methnames[i]):
                        result_line_numbers.add(i)
                for i in range(0, len(tokens)):
                    if len(query_index_for_tokens & set(tokens[i])) >= min_common:
                    #if not query_index_for_tokens.isdisjoint(tokens[i]):
                        result_line_numbers.add(i)
                
            elif index_type == "inverted_index":
                result_line_lists = []
                cnt, cnt_tf = Counter(), Counter()
                for word in query_list:
                    if word in index: # for each word of the processed query that the index contains: ...
                        result_line_lists.append(index[word]) # ... add the list of code fragments containing that word.
                
                for line_list in tqdm(result_line_lists): # iterate the code fragment list of each found query word:
                    if similarity_mode == 'tf_idf':
                        for line_nr in line_list:
                            cnt_tf[line_nr] += 1 # count occurences of the query word in each of its code fragments
                        lines = list(cnt_tf.keys()) # deduplicated list of those code fragments
                        idf   = math.log10(number_of_code_fragments / len(lines)) # idf = log10(N/df)
                        print(f"##### IDF = {idf}")
                        for line_nr in lines:
                            cnt[line_nr] += idf * math.log10(1 + cnt_tf[line_nr]) # tf-idf = idf * log10(1 + tf)
                        cnt_tf.clear() # clear temporary counter for the next query word
                    #################################################if i == 0: print(cnt.items())
                    elif similarity_mode == 'idf':
                        lines = list(set(line_list)) # deduplicated list of code fragments
                        idf   = math.log10(number_of_code_fragments / len(lines)) # idf = log10(N/df)
                        for line_nr in lines:
                            cnt[line_nr] += idf
                    else:
                        for line_nr in list(set(line_list)): # iterate deduplicated list of code fragments
                            cnt[line_nr] += 1
                #result_line_numbers, irrelevant = zip(*cnt.most_common(n_results))
                ##################################################################################################################
                #result_line_numbers, irrelevant = zip(*cnt.most_common(10000 + 100 * n_results))
                result_line_numbers, irrelevant = zip(*cnt.most_common(11 * n_results))
                #print(irrelevant)
                ##################################################################################################################
            
            result_line_numbers = list(result_line_numbers)
            
            
            #print(result_line_numbers)
            full_methname_indices = data_loader.load_hdf5(data_path + config['data_params']['use_methname'], 0, -1)
            full_token_indices    = data_loader.load_hdf5(data_path + config['data_params']['use_tokens'],   0, -1)
            print(f"############################# len(methname_indices): {len(full_methname_indices)}")
            print(f"############################# len(token_indices): {len(full_token_indices)}")
            print(f"############################# len(full_codebase): {len(full_codebase)}")
            #token_indices    = data_loader.load_hdf5_lines(data_path + config['data_params']['use_tokens'], result_line_numbers[0:10])
            #token_indices    = data_loader.load_hdf5_lines(data_path + config['data_params']['use_tokens'], [*range(0, 10, 1)])
            token_indices    = data_loader.load_hdf5_lines(data_path + config['data_params']['use_tokens'], [4098478])
            inverted_token_vocab    = dict((v, k) for k, v in token_vocab.items())
            ft = lambda lst: [inverted_token_vocab.get(   i, 'UNK') for i in lst]
            #print(f"First 10 tokens: {list(map(ft, token_indices))}")
            print(f"token 4098478: {list(map(ft, token_indices))}")
            #methname_indices    = data_loader.load_hdf5_lines(data_path + config['data_params']['use_methname'], [*range(0, 10, 1)])
            methname_indices    = data_loader.load_hdf5_lines(data_path + config['data_params']['use_methname'], [4098478])
            inverted_methname_vocab    = dict((v, k) for k, v in methname_vocab.items())
            fm = lambda lst: [inverted_methname_vocab.get(   i, 'UNK') for i in lst]
            #print(f"First 10 methnames: {list(map(fm, methname_indices))}")
            print(f"methname 4098478: {list(map(fm, methname_indices))}")
            
            
            print(f"Number of pre-filtered possible results: {len(result_line_numbers)}")
            if less_memory:
                engine._code_reprs = data_loader.load_code_reprs_lines(data_path + config['data_params']['use_codevecs'], result_line_numbers, n_threads)
                engine._codebase   = data_loader.load_codebase_lines(  data_path + config['data_params']['use_codebase'], result_line_numbers, n_threads)
            else:
                #print(f"########## {type(result_line_numbers[0])}")
                #result_line_numbers = list(result_line_numbers)
                f = operator.itemgetter(*result_line_numbers)
                codebase_lines = list(f(full_codebase))
                vector_lines   = list(f(full_code_reprs))
                #print(f"First 10 codebase_lines elements: {codebase_lines[0:10]}")
                chunk_size     = math.ceil(len(result_line_numbers) / n_threads)
                
                for i in range(0, len(codebase_lines), chunk_size):
                    codebase.append(codebase_lines[i:i + chunk_size])
                    codereprs.append( vector_lines[i:i + chunk_size])
                """inverted_vocab    = dict((v, k) for k, v in vocab.items())
                fv = lambda lst: [inverted_vocab.get(   i, 'UNK') for i in lst]
                print(f"Top 10 results codereprs {list(map(fv, codereprs[0]))}")"""
                engine._code_reprs = codereprs
                engine._codebase   = codebase
                #print(codebase)
                #print(f"Top 10 codebase elements: {codebase[0]}")
                print(f"codebase elements around 4098478: {full_codebase[4098470:4098485]}")
                """print(len(codebase))
                print(len(codebase[0]))
                print(len(codereprs))
                print(len(codereprs[0]))"""
            deepCS_main.search_and_print_results(engine, model, vocab, query, n_results, )
