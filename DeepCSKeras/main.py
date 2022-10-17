import os
os.environ['NUMEXPR_MAX_THREADS'] = '128'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
import sys
import random
import traceback
from tensorflow.keras.optimizers import RMSprop, Adam
from scipy.stats import rankdata
import math
import time
#import glob
import shutil
import numpy as np
from tqdm import tqdm
import argparse
random.seed(42)
import threading
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

try:
    from utils import normalize, pad, convert, revert
    import models, data_loader, configs
except Exception:
    from .utils import normalize, pad, convert, revert
    from . import models, data_loader, configs

class SearchEngine:
    def __init__(self, args, conf = None):
        self.data_path    = args.data_path + args.dataset + '/' 
        self.train_params = conf.get('training_params', dict())
        self.data_params  = conf.get('data_params',     dict())
        self.model_params = conf.get('model_params',    dict())
        
        self._eval_sets   = None
        
        self._code_reprs  = None
        self._codebase    = None
        self._codebase_chunksize = 2000000

    ##### Model Loading / saving #####
    def save_model(self, model, epoch):
        model_path = f"./output/{model.__class__.__name__}/models/"
        os.makedirs(model_path, exist_ok = True)
        model.save(model_path + f"epo{epoch}_code.h5", model_path + f"epo{epoch}_desc.h5", overwrite = True)
        
    def load_model(self, model, epoch, model_path = None):
        if model_path == None: model_path = f"./output/{model.__class__.__name__}/models/"
        assert os.path.exists( model_path + f"epo{epoch}_code.h5"), f"Weights at epoch {epoch} not found"
        assert os.path.exists( model_path + f"epo{epoch}_desc.h5"), f"Weights at epoch {epoch} not found"
        model.load(model_path + f"epo{epoch}_code.h5", model_path + f"epo{epoch}_desc.h5")


    ##### Training #####
    def train(self, model):
        if self.train_params['reload'] > 0:
            self.load_model(model, self.train_params['reload'])
        valid_every = self.train_params.get('valid_every',      None)
        save_every  = self.train_params.get('save_every',       None)
        batch_size  = self.train_params.get('batch_size',       128)
        nb_epoch    = self.train_params.get('nb_epoch',         10)
        split       = self.train_params.get('validation_split', 0)
        
        val_loss = {'loss': 1., 'epoch': 0}
        chunk_size = self.train_params.get('chunk_size', 100000)
        
        for i in range(self.train_params['reload'] + 1, nb_epoch):
            print('Epoch %d :: \n' % i, end='')  
            
            logger.debug('loading data chunk..')
            offset = (i - 1) * self.train_params.get('chunk_size', 100000)
            
            names  = data_loader.load_hdf5(self.data_path + self.data_params['train_methname'], offset, chunk_size)
            apis   = data_loader.load_hdf5(self.data_path + self.data_params['train_apiseq'],   offset, chunk_size)
            tokens = data_loader.load_hdf5(self.data_path + self.data_params['train_tokens'],   offset, chunk_size)
            descs  = data_loader.load_hdf5(self.data_path + self.data_params['train_desc'],     offset, chunk_size)
            
            logger.debug('padding data..')
            methnames  = pad(names,  self.data_params['methname_len'])
            apiseqs    = pad(apis,   self.data_params['apiseq_len'])
            tokens     = pad(tokens, self.data_params['tokens_len'])
            good_descs = pad(descs,  self.data_params['desc_len'])
            bad_descs  = [desc for desc in descs]
            random.shuffle(bad_descs)
            bad_descs  = pad(bad_descs, self.data_params['desc_len'])

            hist = model.fit([methnames, apiseqs, tokens, good_descs, bad_descs], epochs = 1, batch_size = batch_size, validation_split = split)

            if hist.history['val_loss'][0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))
            
            if save_every is not None and i % save_every == 0:
                self.save_model(model, i)

            if valid_every is not None and i % valid_every == 0:                
                acc, mrr, map, ndcg = self.valid(model, 1000, 1)             

    ##### Evaluation in the develop set #####
    def valid(self, model, poolsize, K):
        """
        validate in a code pool. 
        param: poolsize - size of the code pool, if -1, load the whole test set
        """
        def ACC(real, predict):
            sum = 0.0
            for val in real:
                try: index = predict.index(val)
                except ValueError: index = -1
                if index != -1: sum = sum + 1  
            return sum / float(len(real))
        
        def MAP(real, predict):
            sum = 0.0
            for id, val in enumerate(real):
                try: index = predict.index(val)
                except ValueError: index = -1
                if index != -1: sum = sum + (id + 1) / float(index + 1)
            return sum / float(len(real))
        
        def MRR(real, predict):
            sum = 0.0
            for val in real:
                try: index = predict.index(val)
                except ValueError: index = -1
                if index != -1: sum = sum + 1.0 / float(index + 1)
            return sum / float(len(real))
        
        def NDCG(real, predict):
            dcg  = 0.0
            idcg = IDCG(len(real))
            for i, predictItem in enumerate(predict):
                if predictItem in real:
                    itemRelevance = 1
                    rank = i + 1
                    dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
            return dcg / float(idcg)
        
        def IDCG(n):
            idcg = 0
            itemRelevance = 1
            for i in range(n):
                idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
            return idcg

        # load valid dataset:
        if self._eval_sets is None:
            methnames = data_loader.load_hdf5(self.data_path + self.data_params['valid_methname'], 0, poolsize)
            apiseqs   = data_loader.load_hdf5(self.data_path + self.data_params['valid_apiseq'],   0, poolsize)
            tokens    = data_loader.load_hdf5(self.data_path + self.data_params['valid_tokens'],   0, poolsize)
            descs     = data_loader.load_hdf5(self.data_path + self.data_params['valid_desc'],     0, poolsize) 
            self._eval_sets = {'methnames':methnames, 'apiseqs':apiseqs, 'tokens':tokens, 'descs':descs}
            
        accs, mrrs, maps, ndcgs = [], [], [], []
        data_len = len(self._eval_sets['descs'])
        for i in tqdm(range(data_len)):
            desc      = self._eval_sets['descs'][i] # good desc
            descs     = pad([desc] * data_len,            self.data_params['desc_len'])
            methnames = pad(self._eval_sets['methnames'], self.data_params['methname_len'])
            apiseqs   = pad(self._eval_sets['apiseqs'],   self.data_params['apiseq_len'])
            tokens    = pad(self._eval_sets['tokens'],    self.data_params['tokens_len'])
            n_results = K          
            sims    = model.predict([methnames, apiseqs, tokens, descs], batch_size = data_len).flatten()
            negsims = np.negative(sims)
            predict = np.argpartition(negsims, kth = n_results - 1)
            predict = predict[:n_results]   
            predict = [int(k) for k in predict]
            real    = [i]
            accs.append( ACC( real, predict))
            mrrs.append( MRR( real, predict))
            maps.append( MAP( real, predict))
            ndcgs.append(NDCG(real, predict))  
        acc, mrr, map_, ndcg = np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)
        logger.info(f'ACC={acc}, MRR={mrr}, MAP={map_}, nDCG={ndcg}')        
        return acc, mrr, map_, ndcg
    
    
    ##### Compute Representation #####
    def repr_code(self, model, lines = None):
        if lines == None: 
            logger.info('Loading the use data ..')
            methnames = data_loader.load_hdf5(self.data_path + self.data_params['use_methname'], 0, -1)
            apiseqs   = data_loader.load_hdf5(self.data_path + self.data_params['use_apiseq'],   0, -1)
            tokens    = data_loader.load_hdf5(self.data_path + self.data_params['use_tokens'],   0, -1) 
        else:
            methnames = data_loader.load_hdf5_lines(self.data_path + self.data_params['use_methname'], lines)
            apiseqs   = data_loader.load_hdf5_lines(self.data_path + self.data_params['use_apiseq'],   lines)
            tokens    = data_loader.load_hdf5_lines(self.data_path + self.data_params['use_tokens'],   lines)
        methnames = pad(methnames, self.data_params['methname_len'])
        apiseqs   = pad(apiseqs,   self.data_params['apiseq_len'])
        tokens    = pad(tokens,    self.data_params['tokens_len'])
        
        if lines == None: logger.info('Representing code ..')
        vecs = model.repr_code([methnames, apiseqs, tokens], batch_size = 10000)
        vecs = vecs.astype(np.float64) #vecs.astype(np.float)
        vecs = normalize(vecs)
        return vecs
            
    
    def search(self, model, vocab, query, n_results = 10):
        desc = [convert(vocab, query)] # convert desc sentence to word indices
        padded_desc = pad(desc, self.data_params['desc_len'])
        desc_repr   = model.repr_desc([padded_desc])
        desc_repr   = desc_repr.astype(np.float64)
        desc_repr   = normalize(desc_repr).T # [dim x 1]
        codes, sims = [], []
        threads     = []
        ################ Reload code each time (to simulate usage of database):
        if not self._code_reprs:
            self._code_reprs = data_loader.load_code_reprs(self.data_path + self.data_params['use_codevecs'], self._codebase_chunksize)
        ################
        for i, code_reprs_chunk in enumerate(self._code_reprs):
            t = threading.Thread(target = self.search_thread, args = (codes, sims, desc_repr, code_reprs_chunk, i, n_results))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:# wait until all sub-threads finish
            t.join()
        ################
        #del self._code_reprs
        #gc.collect()
        #self._code_reprs = None
        ################
        return codes, sims
                
    def search_thread(self, codes, sims, desc_repr, code_reprs, i, n_results):        
    #1. compute similarity
        chunk_sims = np.dot(code_reprs, desc_repr) # [pool_size x 1] 
        ################
        del code_reprs
        gc.collect()
        ################
        chunk_sims = np.squeeze(chunk_sims, axis = 1)
    #2. choose top results
        negsims = np.negative(chunk_sims)
        maxinds = np.argpartition(negsims, kth = n_results - 1)
        maxinds = maxinds[:n_results]  
        chunk_sims  = chunk_sims[maxinds]
        if self._codebase:
            chunk_codes = [self._codebase[i][k] for k in maxinds]
            codes.extend(chunk_codes)
        else:
            ################ added ################
            offset = i * self._codebase_chunksize
            for ind in range(0, len(maxinds)):
                maxinds[ind] = maxinds[ind] + offset
                #print(ind)
            codes.extend(maxinds)
            #######################################
            """chunk_codes = data_loader.load_codebase_lines(self.data_path + self.data_params['use_codebase'], maxinds, self._codebase_chunksize, i)
            codes.extend(chunk_codes)"""
        sims.extend( chunk_sims)
        
    def postproc(self, codes_sims):
        codes_, sims_ = zip(*codes_sims)
        #codes = [code for code in codes_]
        #sims  = [sim  for sim  in sims_ ]
        codes = list(codes_)
        sims  = list(sims_ )
        final_codes = []
        final_sims  = []    
        for i in range(len(codes_sims)):
            is_dup = False
            for j in range(i):
                if codes[i][:80] == codes[j][:80] and abs(sims[i] - sims[j]) < 0.01:
                    is_dup = True
            if not is_dup:
                final_codes.append(codes[i])
                final_sims.append(  sims[i])
        return zip(final_codes, final_sims)

    
def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument("--data_path", type=str, default='./data/',              help="working directory")
    parser.add_argument("--model",     type=str, default="JointEmbeddingModel",  help="model name")
    parser.add_argument("--dataset",   type=str, default="codesearchnet",        help="dataset name")
    parser.add_argument("--mode", choices=["train","eval","repr_code","search"], default='search',
                        help="The mode to run. The `train` mode trains a model;"
                        " the `eval` mode evaluat models in a test set "
                        " The `repr_code/repr_desc` mode computes vectors"
                        " for a code snippet or a natural language description with a trained model.")
    parser.add_argument("--verbose",         action="store_true", default=True, help="Be verbose")
    parser.add_argument("--no_manual_input", action="store_true", default=False, # added
                        help="If active the program will process one input via the arguments 'n_results' "
                        " as well as 'query', print the result and terminate; "
                        " Otherwise the program will process user inputs from the console repeatedly "
                        " until terminated by the user.")
    parser.add_argument("--n_results", type=int, default=None, help="number of results to return") # added
    parser.add_argument("--query",     type=str, default=None, help="user query to search for")    # added
    return parser.parse_args()

# moved into a function:
def search_and_print_results(engine, model, vocab, query, n_results, data_path, data_params):
    codes, sims = engine.search(model, vocab, query, n_results)
    ################ added ################
    if not engine._codebase:
        #codes = data_loader.load_codebase_lines(data_path + data_params['use_codebase'], codes, -1)
        codes, sims = zip(*sorted(zip(codes, sims), key = lambda x:x[0]))
        codes = list(codes)
        sims  = list(sims )
        codes = data_loader.load_codebase_lines(data_path + 'sqlite.db', codes, -1) # database
    #######################################
    zipped  = zip(codes, sims)
    zipped  = sorted(zipped, reverse = True, key = lambda x:x[1])
    zipped  = engine.postproc(zipped)
    zipped  = list(zipped)[:n_results]
    results = '\n\n'.join(map(str, zipped)) # combine the result into a returning string
    print(results)
    ################ added ################
    #del codes, sims, zipped, results, engine._codebase
    #gc.collect()
    #######################################
#

if __name__ == '__main__':
    args   = parse_args()
    config = getattr(configs, 'config_' + args.model)()
    engine = SearchEngine(args, config)

    ##### Define model ######
    logger.info('Build Model')
    model = getattr(models, args.model)(config) # initialize the model
    model.build()
    model.summary(export_path = f"./output/{args.model}/")
    
    optimizer = config.get('training_params', dict()).get('optimizer', 'adam')
    model.compile(optimizer = optimizer)  

    data_path = args.data_path + args.dataset + '/'
    
    if args.mode == 'train':  
        engine.train(model)
        
    elif args.mode == 'eval': # evaluate for a specific epoch:
        assert config['training_params']['reload'] > 0, "Please specify the number of epoch of the optimal checkpoint in config.py"
        engine.load_model(model, config['training_params']['reload'])
        engine.valid(model, -1, 10)
        
    elif args.mode == 'repr_code':
        assert config['training_params']['reload'] > 0, "Please specify the number of epoch of the optimal checkpoint in config.py"
        engine.load_model(model, config['training_params']['reload'])
        vecs = engine.repr_code(model)
        data_loader.save_code_reprs(vecs, data_path + config['data_params']['use_codevecs'])
        
    elif args.mode == 'search':
        try:
            shutil.rmtree('__pycache__')
            print('Info: Cleared DeepCSKeras cache.')
        except FileNotFoundError:
            print('Info: DeepCSKeras cache is not present --> nothing to be cleared.')
            pass
        except:
            print("Exception while trying to clear cache directory '__pycache__'! \n Warning: Cache not cleared. --> Time measurements will be distorted!")
            traceback.print_exc()
            pass
            
        # search code based on a desc:
        assert config['training_params']['reload'] > 0, "Please specify the number of epoch of the optimal checkpoint in config.py"
        engine.load_model(model, config['training_params']['reload'])
        #engine._code_reprs = data_loader.load_code_reprs(data_path + config['data_params']['use_codevecs'], engine._codebase_chunksize)
        #engine._codebase   = data_loader.load_codebase(  data_path + config['data_params']['use_codebase'], engine._codebase_chunksize)
        vocab = data_loader.load_pickle(data_path + config['data_params']['vocab_desc'])
        while True:
            if args.no_manual_input: # added:
                query     = args.query
                n_results = args.n_results
                assert query     is not None, "A query input via the '--query' argument is required for 'no_manual_input' mode."
                assert n_results is not None, "Please specify the number of results via the '--n_results' argument for 'no_manual_input' mode."
                start      = time.time()
                start_proc = time.process_time()
            else: #
                try:
                    query     =     input('Input Query: ')
                    n_results = int(input('How many results? '))
                except Exception:
                    print("Exception while parsing your input: ")
                    traceback.print_exc()
                    break
                start      = time.time()
                start_proc = time.process_time()
                query   = query.lower().replace('how to ', '').replace('how do i ', '').replace('how can i ', '').replace('?', '').strip()
            search_and_print_results(engine, model, vocab, query, n_results, data_path, config['data_params'])
            print('Total time:  {:5.3f}s'.format(time.time()-start))
            print('System time: {:5.3f}s'.format(time.process_time()-start_proc))
            if args.no_manual_input:  # added:
                break
