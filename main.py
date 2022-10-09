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
import io
import sys
import math
import time
#import glob
import shutil
import codecs
import argparse
import operator
import traceback
import numpy as np
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
    """parser.add_argument("--similarity_mode", choices=["lexical","tf_idf"], default='tf_idf', help="The metric used for "
                        " similarity calculation between query and code fragments: The 'lexical' similarity mode measures "
                        " the amount of words that query and code fragment have in common (rather simple and inaccurate). "
                        " 'tf_idf' combines term frequency and inverted document frequency (both logarithmically damped) "
                        " to weighten the informativeness of each word and measure the overall quality of match (best known "
                        " metric but more time consuming; incompatible with word_indices as index type).")"""
    parser.add_argument("--memory_mode", choices=["performance","less_memory","database"], default="database", help="'performance': "
                        " [fastest, very memory intensive] All data are loaded just one time at program start and kept in memory  "
                        " for fast access. 'less_memory': [slower] The program will just load pre-filtered elements of some files "
                        " after each query input instead of loading them completely in the beginning. The entire index including "
                        " tf-idf weight counter objects is cept in memory. 'database': [slowest, least memory usage]  ") # TODO: complete
    return parser.parse_args()
   
'''def generate_sublist(list, indices):
    g = lambda lst: (lst[i] for i in indices)
    yield g(list)
    
def chunk_of_iter(iterable, chunk_size):
    chunks = [iter(iterable)] * chunk_size
    return zip(*chunks)'''

if __name__ == '__main__':
    args        = parse_args()
    config      = getattr(configs, 'config_' + args.model)()
    data_path   = args.data_path + args.dataset + '/'
    index_type  = args.index_type
    memory_mode = args.memory_mode
    indexer     = IndexCreator(args, config)
    stopwords   = set("a,about,after,also,an,and,another,are,around,as,at,be,because,been,before,being,between,both,but,by,came,can,create,come,could,did,do,does,each,every,from,get,got,had,has,have,he,her,here,him,himself,his,how,in,into,it,its,just,like,make,many,me,might,more,most,much,must,my,never,no,now,of,on,only,other,our,out,over,re,said,same,see,should,since,so,some,still,such,take,than,that,the,their,them,then,there,these,they,this,those,through,to,too,under,unk,UNK,up,use,very,want,was,way,we,well,were,what,when,where,which,who,will,with,would,you,your".split(','))
    n_threads   = 8 # number of threads for parallelization of less performance intensive program parts
    _codebase_chunksize = 2000000
    tf_idf_threshold    = 2.79 
    
    data_loader.eval_to_db()
    '''

    if args.mode == 'create_index':
        indexer.create_index(stopwords)

    elif args.mode == 'search':
        try:
            shutil.rmtree('__pycache__')
            print('Info: Cleared index_creator cache.')
        except FileNotFoundError:
            print('Info: index_creator cache is not present --> nothing to be cleared.')
            pass
        except:
            print("Exception while trying to clear cache directory '__pycache__'! \n Warning: Cache not cleared. --> Time measurements will be distorted!")
            traceback.print_exc()
            pass
            
        try:
            shutil.rmtree('DeepCSKeras/__pycache__')
            print('Info: Cleared DeepCSKeras cache.')
        except FileNotFoundError:
            print('Info: DeepCSKeras cache is not present --> nothing to be cleared.')
            pass
        except:
            print("Exception while trying to clear cache directory 'DeepCSKeras/__pycache__'! \n Warning: Cache not cleared. --> Time measurements will be distorted!")
            traceback.print_exc()
            pass
            
        ##### Initialize DeepCS search engine and model ######
        engine = deepCS_main.SearchEngine(args, config)
        model  = getattr(models, args.model)(config) # initialize the model
        model.build()
        model.summary(export_path = f"./output/{args.model}/")
        optimizer = config.get('training_params', dict()).get('optimizer', 'adam')
        model.compile(optimizer = optimizer)
        assert config['training_params']['reload'] > 0, "Please specify the number of the optimal epoch checkpoint in config.py"
        engine.load_model(model, config['training_params']['reload'], f"./DeepCSKeras/output/{model.__class__.__name__}/models/")
        #####
        porter = PorterStemmer()
        vocab  = data_loader.load_pickle(data_path + config['data_params']['vocab_desc'])
        
        if memory_mode == "performance":
            full_code_reprs = data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], -1)
            #full_code_reprs = np.array(data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], -1))
            #full_codebase   = np.array(data_loader.load_codebase(  data_path + config['data_params']['use_codebase'], -1))
            full_codebase   = data_loader.load_codebase(  data_path + config['data_params']['use_codebase'], -1)
        
        if index_type == "word_indices":
            methname_vocab  = data_loader.load_pickle(data_path + config['data_params']['vocab_methname'])
            token_vocab     = data_loader.load_pickle(data_path + config['data_params']['vocab_tokens'])
            methnames, tokens = indexer.load_index()
        else:
            index = indexer.load_index()  # TODO: Just load data specified by querywords --> see: Reading (and selecting) data in a table -> Table.where()
        
        while True:
            """file_list = glob.glob('__pycache__/*.pyc')
            if not file_list: print('Info: index_creator cache is not present --> nothing to be cleared.')
            for file in file_list:
                try:
                    os.remove(file)
                    print('Info: Cleared index_creator cache.')
                except:
                    print(f"Exception while trying to clear cache file '{file}'! \n Warning: Cache not cleared. --> Time measurements will be distorted!")
                    traceback.print_exc()
                    pass
            file_list = glob.glob('DeepCSKeras/__pycache__/*.pyc')
            if not file_list: print('Info: DeepCSKeras cache is not present --> nothing to be cleared.')
            for file in file_list:
                try:
                    os.remove(file)
                    print('Info: Cleared a DeepCSKeras cache file.')
                except:
                    print(f"Exception while trying to clear cache file '{file}'! \n Warning: Cache not cleared. --> Time measurements will be distorted!")
                    traceback.print_exc()
                    pass"""
            
            codebase, codereprs, tmp = [], [], []
            result_line_numbers = set()
            ##### Get user input ######
            try:
                query     =     input('Input query: ')
                n_results = int(input('How many results? '))
            except Exception:
                print("Exception while parsing your input: ")
                traceback.print_exc()
                break
            start        = time.time()
            start_proc   = time.process_time()
            max_filtered = max(1000, 50 * n_results)
            min_filtered = max(500,  25 * n_results)
            ##### Process user query ######
            query = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
            query_list = list(set(query.split(' ')) - stopwords)
            #len_query_without_stems = len(query_list)
            for word in query_list:
                word_stem = porter.stem(word)
                if word != word_stem and word_stem not in stopwords:
                    tmp.append(porter.stem(word)) # include stems of query words
            query_list.extend(tmp)
            query_list = [indexer.replace_synonyms(w) for w in query_list]
            print(f"Query without stopwords and possibly with replaced synonyms as well as added word stems: {query_list}")
            #####
            #print("Processing...  Please wait.")
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
                result_line_numbers = list(result_line_numbers)
                
            elif index_type == "inverted_index":
                #result_line_lists = []
                """result_line_counters = []"""
                #cnt, cnt_tf = Counter(), Counter()
                cnt = Counter()
                for word in query_list:
                    if word in index: # for each word of the processed query that the index contains: ...
                        #result_line_lists.append(index[word]) # ... add the list of code fragments containing that word.
                        """result_line_counters.append(index[word]) # ... add the list of code fragments containing that word."""
                        cnt += Counter(dict(index[word].most_common(max_filtered))) # sum tf-idf values for each identical line and merge counters in general '''
                '''print('Time to sum the tf-idf counters:  {:5.3f}s'.format(time.time()-start))'''
                """#for line_list in tqdm(result_line_lists): # iterate the code fragment list of each found query word:
                for line_counter in tqdm(result_line_counters): # iterate the code fragment counters of each found query word:
                    if similarity_mode == 'tf_idf':
                        #for line_nr in line_list:
                        #    cnt_tf[line_nr] += 1 # count occurrences of the query word in each of its code fragments
                        #lines = list(cnt_tf.keys()) # deduplicated list of those code fragments
                        lines = list(line_counter.keys()) # deduplicated list of those code fragments
                        idf   = math.log10(number_of_code_fragments / len(lines)) # idf = log10(N/df)
                        for line_nr in lines:
                        #    cnt[line_nr] += idf * math.log(1 + cnt_tf[line_nr]) # tf-idf = idf * log10(1 + tf); sum values for the same line
                            cnt[line_nr] += idf * math.log(1 + line_counter[line_nr]) # tf-idf = idf * log10(1 + tf); sum values for the same line
                        #cnt_tf.clear() # clear temporary counter for the next query word
                    else: # lexical similarity:
                        #for line_nr in list(set(line_list)): # iterate deduplicated list of code fragments
                        #    cnt[line_nr] += 1
                        cnt += line_counter"""
                ##################################################################################################################
                #result_line_numbers, values = zip(*cnt.most_common(10000 + 100 * n_results))
                #result_line_numbers, values = zip(*cnt.most_common(100 * n_results))
                '''result_line_numbers, values = zip(*cnt.most_common(max_filtered))
                #threshold = values[0]
                #last_threshold_index = 1 + max(idx for idx, val in enumerate(list(values)) if val == threshold)
                last_threshold_index = 1 + max(idx for idx, val in enumerate(list(values)) if val >= tf_idf_threshold)
                #for i in range(0, 1000):
                #    print(values[i])
                result_line_numbers = list(result_line_numbers)
                if last_threshold_index >= min_filtered:
                    result_line_numbers = result_line_numbers[:last_threshold_index]
                else:
                    result_line_numbers = result_line_numbers[:min_filtered] '''
            '''print('Time to calculate most relevant lines:  {:5.3f}s'.format(time.time()-start))'''
            '''print(f"Number of pre-filtered possible results: {len(result_line_numbers)}")
            
            chunk_size = math.ceil(len(result_line_numbers) / max(10, n_results))
            #chunk_size = n_results
            if memory_mode != "performance":
                engine._code_reprs = data_loader.load_code_reprs_lines(data_path + config['data_params']['use_codevecs'], result_line_numbers, chunk_size)
                engine._codebase   = data_loader.load_codebase_lines(  data_path + config['data_params']['use_codebase'], result_line_numbers, chunk_size)
            else:
                f = operator.itemgetter(*result_line_numbers)
                codebase_lines = list(f(full_codebase))
                #codebase_lines = map(full_codebase.__getitem__, result_line_numbers)
                #codebase_lines = full_codebase[result_line_numbers]
                #vector_lines   = list(f(full_code_reprs))
                #vector_lines   = map(full_code_reprs.__getitem__, result_line_numbers)
                vector_lines   = full_code_reprs[result_line_numbers]
                #codebase_lines = list(codebase_lines)
                #vector_lines   = list(vector_lines)
                for i in range(0, len(result_line_numbers), chunk_size):
                #for chunk in chunk_of_iter(codebase_lines, chunk_size):
                    codebase.append(codebase_lines[i:i + chunk_size])
                    #codebase.append(chunk)
                    codereprs.append( vector_lines[i:i + chunk_size])
                #for chunk in chunk_of_iter(vector_lines, chunk_size):
                    #codereprs.append(chunk)
                engine._code_reprs = codereprs
                engine._codebase   = codebase
            deepCS_main.search_and_print_results(engine, model, vocab, query, n_results, )
            print('Total time:  {:5.3f}s  <<<<<<<<<<<<<'.format(time.time()-start))
            print('System time: {:5.3f}s'.format(time.process_time()-start_proc)) '''
