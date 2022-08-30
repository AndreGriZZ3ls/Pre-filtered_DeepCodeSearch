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
import argparse

from index_creator import IndexCreator
import index_creator

from DeepCSKeras import data_loader, configs, models
from DeepCSKeras import main as deepCS_main
from DeepCSKeras.utils import convert, revert


def parse_args():
    parser = argparse.ArgumentParser("Generate Index or perform pre-filtered deep code search")
    parser.add_argument("--data_path",  type=str, default='./DeepCSKeras/data/', help="working directory")
    parser.add_argument("--model",      type=str, default="JointEmbeddingModel", help="DeepCS model name")
    parser.add_argument("--index_type", type=str, default="word_indices",   help="type of index to be created or used")
    parser.add_argument("--dataset",    type=str, default="github",         help="unfiltered dataset name")
    parser.add_argument("--filtered_dataset", type=str, default="filtered", help="name of filtered dataset")
    parser.add_argument("--mode", choices=["create_index","search"], default='search', help="The mode to run:"
                        " The `create_index` mode constructs an index of specified type on the desired dataset; "
                        " The `search` mode filters the dataset according to given query and index before utilizing "
                        " DeepCS with a trained model to search pre-selected for the K most relevant code snippets.")
    return parser.parse_args()

if __name__ == '__main__':
    args           = parse_args()
    config         = getattr(configs, 'config_' + args.model)()
    data_path      = args.data_path + args.dataset + '/'
    index_type     = args.index_type
    methname_vocab = data_loader.load_pickle(data_path + config['data_params']['vocab_methname'])
    token_vocab    = data_loader.load_pickle(data_path + config['data_params']['vocab_tokens'])
    indexer        = IndexCreator(args, config)
    stopwords      = set("a,about,after,also,an,and,another,are,around,as,at,be,because,been,before,being,between,both,but,by,came,can,come,could,did,do,does,each,every,get,got,had,has,have,he,her,here,him,himself,his,how,into,it,its,just,like,make,many,me,might,more,most,much,must,my,never,no,now,of,on,only,other,our,out,over,re,said,same,see,should,since,so,some,still,such,take,than,that,the,their,them,then,there,these,they,this,those,through,to,too,under,up,use,very,want,was,way,we,well,were,what,when,where,which,who,will,with,would,you,your".split(','))

    _codebase_chunksize = 2000000

    if args.mode == 'create_index':
        indexer.create_index()

    elif args.mode == 'search':
        #data_loader.load_code_reprs_lines(data_path + config['data_params']['use_codevecs'], [])
        engine = deepCS_main.SearchEngine(args, config)

        ##### Define model ######
        model = getattr(models, args.model)(config) # initialize the model
        model.build()
        model.summary(export_path = f"./output/{args.model}/")
        
        optimizer = config.get('training_params', dict()).get('optimizer', 'adam')
        model.compile(optimizer = optimizer)
        
        assert config['training_params']['reload'] > 0, "Please specify the number of the optimal epoch checkpoint in config.py"
        engine.load_model(model, config['training_params']['reload'], f"./DeepCSKeras/output/{model.__class__.__name__}/models/")
        full_code_reprs    = data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], _codebase_chunksize)
        engine._code_reprs = full_code_reprs
        vocab = data_loader.load_pickle(data_path+config['data_params']['vocab_desc'])
        
        methnames, tokens = indexer.load_index()
        while True:
            try:
                query     =     input('Input Query: ')
                n_results = int(input('How many results? '))
            except Exception:
                print("Exception while parsing your input:")
                traceback.print_exc()
                break
            engine._code_reprs = full_code_reprs
            print("Processing. Please wait.")
            query = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
            #print(stopwords)
            #print(query)
            query_list = list(set(query.split(' ')) - stopwords)
            #print(query)
            if index_type == "word_indices":
                query_index_for_methnames = set([methname_vocab.get(w, 0) for w in query_list]) # convert user input to word indices
                query_index_for_tokens    = set([token_vocab.get(   w, 0) for w in query_list])
                #print("query_index_for_methnames:")
                #print(query_index_for_methnames)
                #print("query_index_for_tokens:")
                #print(query_index_for_tokens)
                result_line_numbers = set()
                min_common = len(query_list) * 2 / 3 + len(query_list) % 3
                for i in range(0, len(methnames)):
                    if len(query_index_for_methnames & set(methnames[i])) >= min_common:
                    #if not query_index_for_methnames.isdisjoint(methnames[i]):
                        #print(methnames[i])
                        result_line_numbers.add(i)
                for i in range(0, len(tokens)):
                    if len(query_index_for_tokens & set(tokens[i])) >= min_common:
                    #if not query_index_for_tokens.isdisjoint(tokens[i]):
                        #print(tokens[i])
                        result_line_numbers.add(i)
                print(f"Number of pre-filtered possible results: {len(result_line_numbers)}")
                #print(result_line_numbers)
                result_line_numbers = list(result_line_numbers)
                #engine._code_reprs  = data_loader.load_code_reprs_lines(data_path + config['data_params']['use_codevecs'], result_line_numbers)
                engine._code_reprs = engine.repr_code(model, result_line_numbers)
                engine._codebase   = data_loader.load_codebase_lines(  data_path + config['data_params']['use_codebase'], result_line_numbers)
                deepCS_main.search_and_print_results(engine, model, vocab, query, n_results)
                
