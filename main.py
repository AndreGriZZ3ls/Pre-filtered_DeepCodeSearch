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
import re
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
import fileinput
import numpy as np
from tqdm import tqdm
from statistics import mean
from collections import Counter
from nltk.stem import PorterStemmer

from index_creator import IndexCreator
import index_creator

from DeepCSKeras import data_loader, configs, models
from DeepCSKeras import main as deepCS_main
from DeepCSKeras.utils import convert, revert


def parse_args():
    parser = argparse.ArgumentParser("Generate Index or perform pre-filtered deep code search")
    parser.add_argument("--dataset",    type=str, default="codesearchnet", help="dataset name")
    parser.add_argument("--data_path",  type=str, default='./DeepCSKeras/data/',       help="working directory")
    parser.add_argument("--model",      type=str, default="JointEmbeddingModel",       help="DeepCS model name")
    parser.add_argument("--mode", choices=["create_index","search","populate_database","eval","eval_filter"], default='search', 
                        help="The mode to run: 'create_index' mode constructs an index of specified type on the desired dataset; "
                        " 'search' mode filters the dataset according to given query and index before utilizing "
                        " DeepCS with a trained model to search pre-selected for the K most relevant code snippets; "
                        " 'populate_database' mode adds data to the database (for one time use only!); "
                        " 'evaluate' mode evaluates the filter (false negatives).")
    parser.add_argument("--index_type", choices=["word_indices","inverted_index"], default="inverted_index", help="Type of index "
                        " to be created or used: The 'word_indices' mode [not recommended at all] utilizes parts of the dataset "
                        " already existing for DeepCS to work (simple but not usable for more accurete similarity measurements. "
                        " For each meaningful word the 'inverted_index' stores IDs and tf-idf weights of code fragment that contain it. ")
    parser.add_argument("--memory_mode", choices=["performance","vecs_and_code","vecs_and_index","vecs","code_and_index","code","index","nothing"], 
                        default="nothing", help="'performance': [fastest, overly memory intensive, not recommended] All data "
                        " are loaded just one time at program start and kept in memory for fast access. 'vecs_and_code': "
                        " [insignificantly slower, less memory usage] Vectors and raw code are loaded at program start and kept in "
                        " memory; for each query just necessary index items including counter objects are loaded from "
                        " disk very fast. 'vecs': [reasonably slower, quite less memory usage, recommended] Vectors are kept "
                        " in memory; for each query just pre-filtered elements of raw code and index are loaded. 'code':   "
                        " 'nothing': [slowest, least memory usage]  ") # TODO: complete
    return parser.parse_args()
    

if __name__ == '__main__':
    args         = parse_args()
    config       = getattr(configs, 'config_' + args.model)()
    data_path    = args.data_path + args.dataset + '/'
    index_type   = args.index_type
    memory_mode  = args.memory_mode
    index_in_mem = memory_mode in ["performance","vecs_and_index","code_and_index","index"]
    vecs_in_mem  = memory_mode in ["performance","vecs_and_code","vecs_and_index","vecs"]
    code_in_mem  = memory_mode in ["performance","vecs_and_code","code_and_index","code"]
    indexer      = IndexCreator(args, config)
    stopwords    = set("a,about,after,also,an,and,another,any,are,around,as,at,awt,be,because,been,before,being,best,between,both,but,by,came,can,come,could,did,do,does,each,every,final,got,had,has,have,he,her,here,him,himself,his,how,if,in,into,io,it,its,java,javax,just,lang,like,many,me,might,more,most,much,must,my,never,net,no,now,on,only,other,our,out,over,override,private,protected,public,re,return,said,same,see,should,since,so,some,static,still,such,take,than,that,the,their,them,then,there,these,they,this,those,through,throw,throws,too,under,unk,UNK,up,use,util,very,void,want,was,way,we,well,were,what,when,where,which,who,will,with,would,you,your".split(','))
    pattern1     = re.compile(r'[^\[\]a-zA-Z \n\r]+')
    pattern2     = re.compile(r' \w? +')
    n_threads    = 8 # number of threads for parallelization
    _codebase_chunksize = 2000000
    tf_idf_threshold    = 1.0 #2.79 # 2.00
    

    if args.mode == 'populate_database':
        #data_loader.data_to_db(data_path, config)
        #print('Info: Populating the database was sucessful.')
        """index = indexer.load_index()
        for word in index.keys():
            index[word] = Counter(dict(sorted(index[word].items(), key=lambda x: (-x[1], x[0]))))
        data_loader.save_index(index_type, index, data_path)
        data_loader.save_pickle(data_path + index_type + '.pkl', index)"""
        #data_loader.codebase_to_sqlite(data_path + config['data_params']['use_codebase'], data_path + 'sqlite.db')
        #data_loader.index_to_sqlite(index_type, data_path + index_type + '.pkl', data_path + 'sqlite.db')
        #indexer.process_raw_code()
        """porter = PorterStemmer()
        to_stem = ".replace(' read ', 'load').replace(' write', 'store').replace('save', 'store').replace(' dump', 'store')\
        .replace('object', 'instance').replace(' quit', 'exit').replace('terminate', 'exit').replace(' leave', 'exit')\
        .replace(' pop ', 'delet').replace('remov', 'delet').replace(' trim ', 'delet').replace(' strip ', 'delet')\
        .replace(' halt', 'stop').replace('restart', 'continue').replace('push ', 'add')\
        .replace('null ', 'none').replace('method', 'function').replace('concat ', 'combine').replace(' break ', 'exit')\
        .replace(' for ', 'loop').replace(' foreach ', 'loop').replace(' while ', 'loop').replace(' iterat ', 'loop')\
        .replace('integer ', 'int').replace('tinyint ', 'int').replace(' smallint ', 'int').replace(' bigint ', 'int')\
        .replace(' shortint ', 'int').replace('longint ', 'int').replace(' byte ', 'int').replace(' long ', 'int').replace(' short ', 'int')\
        .replace(' double ', 'float').replace(' long ', 'float').replace(' decimal ', 'float').replace(' whitespace ', 'space')\
        .replace('real ', 'float').replace(' array ', '[]').replace(' arrays ', '[]').replace(' arr ', '[]').replace(' fastest ', 'fast')\
        .replace(' define ', 'create').replace(' declare ', 'create').replace(' init ', 'create').replace(' construct ', 'create')\
        .replace(' make ', 'create').replace(' boolean ', 'bool').replace('begin', 'start').replace('run ', 'execute')\
        .replace(' initialize ', 'create').replace(' initialized ', 'create').replace(' initializing ', 'create').replace(' initi ', 'create')\
        .replace(' enumerate ', 'enum').replace(' enumerated ', 'enum').replace(' enumeration ', 'enum').replace(' website ', 'web')\
        .replace(' speed ', 'fast').replace(' vertex ', 'node').replace(' arc ', 'edge').replace(' math ', 'calc').replace(' determine ', 'calc')\
        .replace(' equality ', 'compare').replace(' equals ', 'compare').replace(' equal ', 'compare').replace(' ensure ', 'check')\
        .replace(' should ', 'check').replace(' test ', 'check').replace(' is ', 'check')\
        .replace(' initiate ', 'create').replace(' implements ', 'extends').replace('runnable', 'executable')"
        for word in re.findall(r'[a-z]+', to_stem):
            if word == 'replace': continue
            to_stem = to_stem.replace(word, porter.stem(word))
        print(to_stem)"""
        print('Nothing done.')
    
    elif args.mode == 'create_index':
    #if args.mode == 'create_index':
        indexer.create_index(stopwords)

    elif args.mode in ["eval","eval_filter"]:
        e = 0
        if args.mode  == "eval":
            source_file = io.open(data_path + 'eval_difference.txt', "r", encoding='utf8', errors='replace')
            queries     = source_file.readlines()
            source_file.close()
            amount_diff, mean_sims, mean_sims_pf = [], [], []
            #out_path    = data_path + 'search_results.txt'
            out_path_pf = data_path + 'search_results_filtered.txt'
            #if os.path.exists(out_path   ): os.remove(out_path)
            if os.path.exists(out_path_pf): os.remove(out_path_pf)
            #out_file    = io.open(out_path   , "a", encoding='utf8', errors='replace')
            out_file_pf = io.open(out_path_pf, "a", encoding='utf8', errors='replace')
            
            engine = deepCS_main.SearchEngine(args, config)
            model  = getattr(models, args.model)(config) # initialize the model
            model.build()
            model.summary(export_path = f"./output/{args.model}/")
            optimizer = config.get('training_params', dict()).get('optimizer', 'adam')
            model.compile(optimizer = optimizer)
            assert config['training_params']['reload'] > 0, "Please specify the number of the optimal epoch checkpoint in config.py"
            engine.load_model(model, config['training_params']['reload'], f"./DeepCSKeras/output/{model.__class__.__name__}/models/")
            vocab  = data_loader.load_pickle(data_path + config['data_params']['vocab_desc'])
            full_code_reprs  = data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], -1)
            full_codebase    = data_loader.load_codebase(  data_path + config['data_params']['use_codebase'], -1)
            _full_code_reprs = data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], _codebase_chunksize)
            _full_codebase   = data_loader.load_codebase(  data_path + config['data_params']['use_codebase'], _codebase_chunksize)
        else:
            eval_dict = data_loader.load_pickle(data_path + 'eval_filter.pkl')
            queries   = list(eval_dict.keys())
            global_cnt   = Counter()
            result_path  = data_path + 'eval_results.txt'
            if os.path.exists(result_path): os.remove(result_path)
            result_file  = io.open(result_path, "a", encoding='utf8', errors='replace')
        index     = indexer.load_index()
        n_results = 10
        porter    = PorterStemmer()
        max_filtered = max(500, 50 * n_results + 250)
        min_filtered = max(500, 25 * n_results + 250)
        
        
        for query in queries:
            if args.mode  == "eval":
                engine._code_reprs = _full_code_reprs
                engine._codebase   = _full_codebase
                query_DeepCS = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
                deepCS_codes, deepCS_sims = deepCS_main.search_and_print_results(engine, model, vocab, query_DeepCS, n_results, data_path, config['data_params'], True)
            else:
                query_lines  = list(eval_dict[query].keys())
                query_scores = list(eval_dict[query].values())
            ##### Process user query ######
            query_proc = re.sub(pattern1, ' ', query) # replace all non-alphabetic characters except '[' by ' '
            query_proc = re.sub(pattern2, ' ', query_proc.strip()) # remove consecutive spaces
            query_proc = query_proc.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('what is ', '').replace('?', '').replace(' numeric', ' numeric int double decimal').strip()
            query_list = list(set(query_proc.split(' ')) - stopwords)
            for i in range(0, len(query_list)):
                query_list[i] = porter.stem(query_list[i])
            query_list = [indexer.replace_synonyms(w) for w in query_list]
            query_list = list(set(query_list))
            print(f"Query without stopwords and possibly with replaced synonyms as well as added word stems: {query_list}")
            query_cnt, cnt = Counter(), Counter()
            for word in query_list:
                if word in index: # for each word of the processed query that the index contains: ...
                    cnt.update(index[word])
            result_line_numbers, values = zip(*cnt.most_common(max_filtered))
            try:
                last_threshold_index = 1 + max(idx for idx, val in enumerate(list(values)) if val >= tf_idf_threshold)
            except ValueError:
                last_threshold_index = -1
            result_line_numbers = list(result_line_numbers)
            if last_threshold_index >= min_filtered:
                result_line_numbers = result_line_numbers[:last_threshold_index]
            else:
                result_line_numbers = result_line_numbers[:min_filtered]
            result_line_numbers.sort()
            
            if args.mode == "eval":
                chunk_size = math.ceil(len(result_line_numbers) / max(10, n_results / 10))
                vector_lines = full_code_reprs[result_line_numbers]
                engine._code_reprs = [vector_lines[i:i + chunk_size] for i in range(0, len(result_line_numbers), chunk_size)]
                codebase_lines = [full_codebase[line] for line in result_line_numbers]
                engine._codebase = [codebase_lines[i:i + chunk_size] for i in range(0, len(result_line_numbers), chunk_size)]
                codes, sims = deepCS_main.search_and_print_results(engine, model, vocab, query_DeepCS, n_results, data_path, config['data_params'], True)
                mean_sims.append(   mean(deepCS_sims))
                mean_sims_pf.append(mean(       sims))
                amount_diff.append(len(list(set(codes) & set(deepCS_codes))))
                e += 1
                seperator = f"########################## {e} #################################\n"
                metrics = "\n\nFRank:   | P@1:   | P@5:   | P@10: \n\n"
                #out_file.write(   seperator + '\n\n'.join(map(str, list(zip(deepCS_codes, deepCS_sims)))) + metrics)
                out_file_pf.write(seperator + '\n\n'.join(map(str, list(zip(       codes,        sims)))) + metrics)
            else:
                for s, line in enumerate(query_lines):
                    score = query_scores[s]
                    if line in result_line_numbers:
                        query_cnt["found_{}".format(score)] += 1
                    query_cnt["total_{}".format(score)] += 1
                
                global_cnt.update(query_cnt)
                e += 1
                result_file.write(f"{e}&{query}&{query_cnt['found_3']} / {query_cnt['total_3']}&{query_cnt['found_2']} / {query_cnt['total_2']}&{query_cnt['found_1']} / {query_cnt['total_1']}\\\\\n")
        if args.mode == "eval":
            e = 0
            result_file = fileinput.FileInput(data_path + 'eval_difference_results.txt', inplace=1)
            for line in result_file:
                line = re.sub(r'(&[\d,]+&[\d,]+&)', f"&{format(round(mean_sims[e], 4), '.4f')}&{format(round(mean_sims_pf[e], 4), '.4f')}&", line)
                line = re.sub(r'(&\d+\\\\$)', f"&{10 - amount_diff[e]}\\\\\ ", line)
                print(line.strip())
                e += 1
            #out_file.write(   f"Mean sims: {format(round(mean(mean_sims), 4), '.4f')}")
            out_file_pf.write(f"Mean sims: {format(round(mean(mean_sims_pf), 4), '.4f')}")
            #out_file.close()
            out_file_pf.close()
        else:
            result_file.write(f"&Insgesamt&{global_cnt['found_3']} / {global_cnt['total_3']}&{global_cnt['found_2']} / {global_cnt['total_2']}&{global_cnt['found_1']} / {global_cnt['total_1']}\\\\\n")
        result_file.close()
    
    elif args.mode == 'search':
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
        
        if vecs_in_mem: 
            full_code_reprs = data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], -1)
            #full_code_reprs = np.array(data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], -1))
        if code_in_mem: 
            #full_codebase   = np.array(data_loader.load_codebase(  data_path + config['data_params']['use_codebase'], -1))
            full_codebase   = data_loader.load_codebase(  data_path + config['data_params']['use_codebase'], -1)
        
        if index_type == "word_indices":
            methname_vocab  = data_loader.load_pickle(data_path + config['data_params']['vocab_methname'])
            token_vocab     = data_loader.load_pickle(data_path + config['data_params']['vocab_tokens'])
            methnames, tokens = indexer.load_index()
        elif index_in_mem:
            index = indexer.load_index()
        
        while True:
            #tmp = []
            ##### Get user input ######
            try:
                query        =     input('Input query: ')
                if query    == 'q': break
                n_results    = int(input('How many results? '))
                if n_results < 1: raise ValueError('Number of results has to be at least 1!')
            except Exception:
                print("Exception while parsing your input: ")
                traceback.print_exc()
                break
            start        = time.time()
            start_proc   = time.process_time()
            max_filtered = max(500, 50 * n_results + 250)
            #max_filtered = max(1000, 75 * n_results)
            min_filtered = max(500, 25 * n_results + 250)
            ##### Process user query ######
            query_proc = re.sub(pattern1, ' ', query) # replace all non-alphabetic characters except '[' by ' '
            query_proc = re.sub(pattern2, ' ', query_proc.strip()) # remove consecutive spaces and single caracters
            query_proc = query_proc.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('what is ', '').replace('?', '').replace(' numeric', ' numeric int double decimal').strip()
            query      = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
            query_list = list(set(query_proc.split(' ')) - stopwords)
            #len_query_without_stems = len(query_list)
            """for word in query_list:
                word_stem = porter.stem(word)
                if word != word_stem and word_stem not in stopwords:
                    tmp.append(word_stem) # include stems of query words"""
            for i in range(0, len(query_list)):
                query_list[i] = porter.stem(query_list[i])
            #query_list.extend(tmp)
            query_list = [indexer.replace_synonyms(w) for w in query_list]
            query_list = list(set(query_list))
            #print('Time to prepare query:  {:5.3f}s'.format(time.time()-start))
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
                if index_in_mem:
                    #cnt = None
                    """for word in query_list:
                        if word in index: # for each word of the processed query that the index contains: ...
                            #cnt += Counter(dict(index[word].most_common(max_filtered))) # sum tf-idf values for each identical line and merge counters in general 
                            #cnt += Counter(dict(itertools.islice(index[word].items(), max_filtered))) # sum tf-idf values for each identical line and merge counters in general 
                            if cnt:
                                cnt.update(index[word])
                            else:
                                cnt = index[word].copy()"""
                    counters = [index[word] for word in query_list if word in index]
                    cnt = sum(counters[1:], counters[0].copy())
                else:
                    #counters = data_loader.load_index_counters(index_type, query_list, data_path + 'sqlite.db') # TODO: compare
                    counters = data_loader.load_index_counters(index_type, query_list, data_path)
                    """cnt = counters[0]
                    for i in range(1, len(counters)):
                        cnt.update(counters[i]) # sum tf-idf values for each identical line and merge counters in general """
                    cnt = sum(counters[1:], counters[0])
                print('Time to sum the tf-idf counters:  {:5.3f}s'.format(time.time()-start))
                ##################################################################################################################
                result_line_numbers, values = zip(*cnt.most_common(max_filtered))
                #result_line_numbers, values = zip(*itertools.islice(sorted(cnt.items(), key=lambda x: (-x[1], x[0])), max_filtered))
                #cnt = None
                print('Time to sort and slice:  {:5.3f}s'.format(time.time()-start))
                try:
                    last_threshold_index = 1 + max(idx for idx, val in enumerate(list(values)) if val >= tf_idf_threshold)
                except ValueError:
                    last_threshold_index = -1
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
            
            #chunk_size = math.ceil(len(result_line_numbers) / max(10, n_results / 10))
            chunk_size = math.ceil(len(result_line_numbers) / n_threads)
            if vecs_in_mem:
                vector_lines = full_code_reprs[result_line_numbers]
                engine._code_reprs = [vector_lines[i:i + chunk_size] for i in range(0, len(result_line_numbers), chunk_size)]
                #engine._code_reprs = [vector_lines]
            else:
                engine._code_reprs = data_loader.load_code_reprs_lines(data_path + config['data_params']['use_codevecs'], result_line_numbers, chunk_size)
            
            if code_in_mem:
                codebase_lines = [full_codebase[line] for line in result_line_numbers]
                engine._codebase = [codebase_lines[i:i + chunk_size] for i in range(0, len(result_line_numbers), chunk_size)]
                #engine._codebase = [codebase_lines]
            else:
                engine._codebase = data_loader.load_codebase_lines(data_path + 'sqlite.db', result_line_numbers, chunk_size) # database
            #result_line_numbers = None
            print('DeepCS start time: {:5.3f}s  <<<<<<<<<<<<<'.format(time.time() - start))
            deepCS_main.search_and_print_results(engine, model, vocab, query, n_results, data_path, config['data_params'])
            #if not vecs_in_mem: engine._code_reprs = None
            #if not code_in_mem: engine._codebase   = None
            print('Total time:  {:5.3f}s  <<<<<<<<<<<<<'.format(time.time() - start))
            print('System time: {:5.3f}s'.format(time.process_time() - start_proc))
