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
import itertools
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
    parser.add_argument("--index_dir",  type=str, default="indices",       help="index directory")
    parser.add_argument("--dataset",    type=str, default="codesearchnet", help="dataset name")
    parser.add_argument("--data_path",  type=str, default='./DeepCSKeras/data/',       help="working directory")
    parser.add_argument("--model",      type=str, default="JointEmbeddingModel",       help="DeepCS model name")
    parser.add_argument("--mode", choices=["create_index","search","populate_database","evaluate"], default='search', 
                        help="The mode to run: 'create_index' mode constructs an index of specified type on the desired dataset; "
                        " 'search' mode filters the dataset according to given query and index before utilizing "
                        " DeepCS with a trained model to search pre-selected for the K most relevant code snippets; "
                        " 'populate_database' mode adds data to the database (for one time use only!); "
                        " 'evaluate' mode evaluates the filter (false negatives).")
    parser.add_argument("--index_type", choices=["word_indices","inverted_index"], default="inverted_index", help="Type of index "
                        " to be created or used: The 'word_indices' mode [not recommended at all] utilizes parts of the dataset "
                        " already existing for DeepCS to work (simple but not usable for more accurete similarity measurements. "
                        " For each meaningful word the 'inverted_index' stores IDs and tf-idf weights of code fragment that contain it. ")
    parser.add_argument("--memory_mode", choices=["performance","vecs_and_code","vecs_and_index","vecs","code","nothing"], 
                        default="nothing", help="'performance': [fastest, overly memory intensive, not recommended] All data "
                        " are loaded just one time at program start and kept in memory for fast access. 'vecs_and_code': "
                        " [insignificantly slower, less memory usage] Vectors and raw code are loaded at program start and kept in "
                        " memory; for each query just necessary index items including counter objects are loaded from "
                        " disk very fast. 'vecs': [reasonably slower, quite less memory usage, recommended] Vectors are kept "
                        " in memory; for each query just pre-filtered elements of raw code and index are loaded. 'code':   "
                        " 'nothing': [slowest, least memory usage]  ") # TODO: complete
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
    stopwords   = set("a,about,after,also,an,and,another,are,around,as,at,be,because,been,before,being,between,both,but,by,came,can,come,could,did,do,does,each,every,from,get,got,had,has,have,he,her,here,him,himself,his,how,in,into,it,its,just,like,make,many,me,might,more,most,much,must,my,never,no,now,of,on,only,other,our,out,over,re,said,same,see,should,since,so,some,still,such,take,than,that,the,their,them,then,there,these,they,this,those,through,to,too,under,unk,UNK,up,use,very,want,was,way,we,well,were,what,when,where,which,who,will,with,would,you,your".split(','))
    n_threads   = 8 # number of threads for parallelization of less performance intensive program parts
    _codebase_chunksize = 2000000
    tf_idf_threshold    = 2.79 
    

    if args.mode == 'populate_database':
        #data_loader.data_to_db(data_path, config)
        #print('Info: Populating the database was sucessful.')
        index = indexer.load_index()
        for word in index.keys():
            index[word] = Counter(dict(sorted(index[word].items(), key=lambda x: (-x[1], x[0]))))
        data_loader.save_index(index_type, index, data_path)
        data_loader.save_pickle(data_path + index_type + '.pkl', index)
        #data_loader.codebase_to_sqlite(data_path + config['data_params']['use_codebase'], data_path + 'sqlite.db')
        print('Nothing done.')
    
    elif args.mode == 'create_index':
    #if args.mode == 'create_index':
        indexer.create_index(stopwords)

    elif args.mode == 'evaluate':
        e = 0
        eval_dict = data_loader.load_pickle(data_path + 'eval_filter.pkl')
        queries   = list(eval_dict.keys())
        #line_nrs  = list(eval_dict.values().keys())
        #scores    = list(eval_dict.values().values())
        index     = indexer.load_index()
        n_results = 10
        porter    = PorterStemmer()
        max_filtered = max(1000, 50 * n_results)
        min_filtered = max(500,  25 * n_results)
        global_cnt   = Counter()
        result_path  = data_path + 'eval_results.txt'
        if os.path.exists(result_path):
            os.remove(result_path)
        result_file  = io.open(result_path, "a", encoding='utf8', errors='replace')
        
        for query in queries:
            query_lines  = list(eval_dict[query].keys())
            query_scores = list(eval_dict[query].values())
            ##### Process user query ######
            tmp   = []
            query_proc = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
            query_list = list(set(query_proc.split(' ')) - stopwords)
            """for word in query_list:
                word_stem = porter.stem(word)
                if word != word_stem and word_stem not in stopwords:
                    tmp.append(word_stem) # include stems of query words"""
            for i in range(0, len(query_list)):
                query_list[i] = porter.stem(query_list[i])
            query_list.extend(tmp)
            query_list = [indexer.replace_synonyms(w) for w in query_list]
            query_list = list(set(query_list))
            print(f"Query without stopwords and possibly with replaced synonyms as well as added word stems: {query_list}")
            cnt, query_cnt = Counter(), Counter()
            for word in query_list:
                if word in index: # for each word of the processed query that the index contains: ...
                    #cnt += Counter(dict(index[word].most_common(max_filtered))) # sum tf-idf values for each identical line and merge counters in general 
                    cnt += Counter(dict(itertools.islice(index[word].items(), max_filtered))) # sum tf-idf values for each identical line and merge counters in general 
            #result_line_numbers, values = zip(*cnt.most_common(max_filtered))
            result_line_numbers, values = zip(*itertools.islice(sorted(cnt.items(), key=lambda x: (-x[1], x[0])), max_filtered))
            last_threshold_index = 1 + max(idx for idx, val in enumerate(list(values)) if val >= tf_idf_threshold)
            result_line_numbers = list(result_line_numbers)
            if last_threshold_index >= min_filtered:
                result_line_numbers = result_line_numbers[:last_threshold_index]
            else:
                result_line_numbers = result_line_numbers[:min_filtered]
            print(f"Number of pre-filtered possible results: {len(result_line_numbers)}")
            result_line_numbers = set(result_line_numbers)
            
            for s, line in enumerate(query_lines):
                score = query_scores[s]
                if line in result_line_numbers:
                    query_cnt["found_{}".format(score)] += 1
                query_cnt["total_{}".format(score)] += 1
            
            global_cnt += query_cnt
            e += 1
            result_file.write(f"{e}&{query}&{query_cnt['found_3']} / {query_cnt['total_3']}&{query_cnt['found_2']} / {query_cnt['total_2']}&{query_cnt['found_1']} / {query_cnt['total_1']}\\\\\n")
        result_file.write(f"&Insgesamt&{global_cnt['found_3']} / {global_cnt['total_3']}&{global_cnt['found_2']} / {global_cnt['total_2']}&{global_cnt['found_1']} / {global_cnt['total_1']}\\\\\n")
        result_file.close()
    
    elif args.mode == 'search':
        """try:
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
            pass"""
        
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
        
        if memory_mode in ["performance","vecs_and_code","vecs","vecs_and_index"]: 
            full_code_reprs = data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], -1)
            #full_code_reprs = np.array(data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], -1))
        if memory_mode in ["performance","vecs_and_code","code"]: 
            #full_codebase   = np.array(data_loader.load_codebase(  data_path + config['data_params']['use_codebase'], -1))
            full_codebase   = data_loader.load_codebase(  data_path + config['data_params']['use_codebase'], -1)
        
        if index_type == "word_indices":
            methname_vocab  = data_loader.load_pickle(data_path + config['data_params']['vocab_methname'])
            token_vocab     = data_loader.load_pickle(data_path + config['data_params']['vocab_tokens'])
            methnames, tokens = indexer.load_index()
        elif memory_mode in ["performance","vecs_and_index"]:
            index = indexer.load_index()
        
        while True:
            tmp = []
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
            """for word in query_list:
                word_stem = porter.stem(word)
                if word != word_stem and word_stem not in stopwords:
                    tmp.append(word_stem) # include stems of query words"""
            for i in range(0, len(query_list)):
                query_list[i] = porter.stem(query_list[i])
            query_list.extend(tmp)
            query_list = [indexer.replace_synonyms(w) for w in query_list]
            query_list = list(set(query_list))
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
                if memory_mode in ["performance","vecs_and_index"]:
                    for word in query_list:
                        if word in index: # for each word of the processed query that the index contains: ...
                            #cnt += Counter(dict(index[word].most_common(max_filtered))) # sum tf-idf values for each identical line and merge counters in general 
                            cnt += Counter(dict(itertools.islice(index[word].items(), max_filtered))) # sum tf-idf values for each identical line and merge counters in general 
                else:
                    for counter in data_loader.load_index_counters(index_type, query_list, data_path, max_filtered):
                        cnt += counter # sum tf-idf values for each identical line and merge counters in general 
                #print('Time to sum the tf-idf counters:  {:5.3f}s'.format(time.time()-start))
                ##################################################################################################################
                #result_line_numbers, values = zip(*cnt.most_common(max_filtered))
                result_line_numbers, values = zip(*itertools.islice(sorted(cnt.items(), key=lambda x: (-x[1], x[0])), max_filtered))
                last_threshold_index = 1 + max(idx for idx, val in enumerate(list(values)) if val >= tf_idf_threshold)
                #for i in range(0, 1000):
                #    print(values[i])
                result_line_numbers = list(result_line_numbers)
                if last_threshold_index >= min_filtered:
                    result_line_numbers = result_line_numbers[:last_threshold_index]
                else:
                    result_line_numbers = result_line_numbers[:min_filtered]
                result_line_numbers.sort()
            #print('Time to calculate most relevant lines:  {:5.3f}s'.format(time.time()-start))
            print(f"Number of pre-filtered possible results: {len(result_line_numbers)}")
            
            chunk_size = math.ceil(len(result_line_numbers) / max(10, n_results))
            #chunk_size = n_results
            if memory_mode in ["performance","vecs_and_code","vecs","vecs_and_index"]:
                vector_lines = full_code_reprs[result_line_numbers]
                #vector_lines = [full_code_reprs[line] for line in result_line_numbers]
                engine._code_reprs = [vector_lines[i:i + chunk_size] for i in range(0, len(result_line_numbers), chunk_size)]
            else:
                engine._code_reprs = data_loader.load_code_reprs_lines(data_path + config['data_params']['use_codevecs'], result_line_numbers, chunk_size)
            if memory_mode in ["performance","vecs_and_code","code"]:
                #f = operator.itemgetter(*result_line_numbers)
                #codebase_lines = list(f(full_codebase))
                codebase_lines = [full_codebase[line] for line in result_line_numbers]
                #codebase_lines = full_codebase[result_line_numbers]
                engine._codebase = [codebase_lines[i:i + chunk_size] for i in range(0, len(result_line_numbers), chunk_size)]
            else:
                engine._codebase = data_loader.load_codebase_lines(data_path + 'sqlite.db', result_line_numbers, chunk_size) # database
                #engine._codebase = data_loader.load_codebase_lines(data_path + config['data_params']['use_codebase'], result_line_numbers, chunk_size)
            print('DeepCS start time: {:5.3f}s  <<<<<<<<<<<<<'.format(time.time() - start))
            deepCS_main.search_and_print_results(engine, model, vocab, query, n_results, data_path, config['data_params'])
            print('Total time:  {:5.3f}s  <<<<<<<<<<<<<'.format(time.time() - start))
            print('System time: {:5.3f}s'.format(time.process_time() - start_proc))
