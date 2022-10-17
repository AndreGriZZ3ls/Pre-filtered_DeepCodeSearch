import os
import io
import math
import time
import tables
import pickle
import logging
import operator
import numpy as np
from tables import *
from tqdm import tqdm
#from unqlite import UnQLite # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
from collections import Counter
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

class IndexMetaData(IsDescription):
    word = StringCol(18)
    len  = UInt32Col()
    pos  = UInt32Col()

def load_pickle(path):
    return pickle.load(open(path, 'rb')) 

# added:
def save_pickle(path, index):
    pickle.dump(index, open(path, 'wb'), pickle.HIGHEST_PROTOCOL) #
    
def load_index_counters(name, word_list, index_path):
    """db = UnQLite(filename = './DeepCSKeras/data/database.udb', open_database = True)
    collec = db.collection(name)
    if not collec.exists():
        raise Exception(f"ERROR: The collection for index type '{name}' does not exist in the database! You have to create this index type first.")
    counters = []
    data     = collec.filter(lambda word: word['word'] in word_list)
    for d in data:
        counters.append(Counter(dict(zip(d[1], d[2]))))
    db.close()
    print(f"Index successfully loaded from '{name}' collection in database.")
    return counters"""
    index_file = index_path + name + '.h5'
    assert os.path.exists(index_file), f"Index file {index_file} not found!"
    h5f      = tables.open_file(index_file, mode = "r")
    meta     = h5f.root.meta
    keys     = h5f.root.keys
    vals     = h5f.root.vals
    counters = []
    for word in word_list:
        #word = word.encode()
        cond = f'word == b"{word}"'
        for row in meta.where(cond):
            l = row['len']
            p = row['pos']
            k = keys[p:p + l]
            v = vals[p:p + l]
            counters.append(Counter(dict(zip(k, v))))
    h5f.close()
    print(f"Successfully loaded tf-idf value counters from index '{index_file}'.")
    return counters
    
    
def save_index(name, index, index_path):
    """db = UnQLite(filename = './DeepCSKeras/data/database.udb', open_database = True)
    collec = db.collection(name)
    collec.drop()
    collec.create()
    for item in index.items():
        cnt  = item[1]
        keys = list(cnt.keys())
        vals = list(cnt.values())
        collec.store({'word': item[0], 'keys': keys, 'vals': vals})
    db.close()
    print(f"Index successfully saved to '{name}' collection in database.")"""
    index_file = index_path + name + '.h5'
    if os.path.exists(index_file):
        os.remove(index_file)
    atom_k  = tables.Atom.from_dtype(np.dtype(np.int32, (0,)))
    atom_v  = tables.Atom.from_dtype(np.dtype(np.float64, (0,)))
    filters = tables.Filters(complib = 'blosc', complevel = 5)
    h5f     = tables.open_file(index_file, mode = "w", title = name)
    table   = h5f.create_table("/", 'meta', IndexMetaData, "index meta data")
    meta    = table.row
    keys    = h5f.create_earray(h5f.root, 'keys', atom_k, (0,), "key of the counter elements", filters)
    vals    = h5f.create_earray(h5f.root, 'vals', atom_v, (0,), "values of the counter elements", filters)
    pos     = 0
    for item in index.items():
        k = np.array(list(item[1].keys()), dtype = np.int32)
        v = np.array(list(item[1].values()), dtype = np.float64)
        k.flatten()
        v.flatten()
        l = k.shape[0]
        meta['word'] = item[0]
        meta['len']  = l
        meta['pos']  = pos
        meta.append()
        pos += l
        keys.append(k)
        vals.append(v)
    table.flush()
    table.cols.word.create_csindex(filters)
    table.flush()
    h5f.close()

##### Data Set #####
#def load_codebase(path, chunk_size, chunk_number = -1):
def load_codebase(path, chunk_size):
    """load codebase
    codefile: h5 file that stores raw code
    """
    logger.info('Loading codebase (chunk size = {}) ...'.format(chunk_size))
    codebase = []
    #if chunk_number > -1:
    #    offset = chunk_size * chunk_number
    #    return io.open(path, encoding='utf8', errors='replace').readlines()[offset:offset + chunk_size]
    codes = io.open(path, "r", encoding='utf8', errors='replace').readlines()
    if chunk_size < 0: return codes
    else:
        for i in tqdm(range(0, len(codes), chunk_size)):
            codebase.append(codes[i:i + chunk_size])            
    return codebase

# added:
def get_lines_generator(iterable, lines):
    results  = [None] * len(lines)
    line_set = set(lines)
    #print(len(line_set))
    for i, line in enumerate(iterable):
        if i in line_set:
            results[lines.index(i)] = line
    return results#

# added:
def load_codebase_lines(path, lines, chunk_size, chunk_number = -1): 
    """load some codebase lines
    codefile: h5 file that stores raw code
    """
    logger.info(f'Loading {len(lines)} pre-filtered codebase lines ...')
    codes = io.open(path, "r", encoding='utf8',errors='replace')
    #codes = io.open(path, encoding='utf8',errors='replace').readlines()
    #f = operator.itemgetter(*lines)
    #codebase_lines = list(f(codes))
    codebase       = []
    start = time.time()
    if chunk_number > 0:
        offset = chunk_number * chunk_size
        for line in lines:
            line += offset
    codebase_lines = get_lines_generator(codes, lines)
    print('Total load_codebase_lines time:  {:5.3f}s  <<<<<<<<<<<<<'.format(time.time()-start))
    #codebase_lines = codes[lines]
    if chunk_number > -1 or chunk_size < 0: return codebase_lines
    for i in range(0, len(lines), chunk_size):
        codebase.append(codebase_lines[i:i + chunk_size])
    return codebase #
    
#def convert_codebase(path, target):
#    codes = io.open(path, encoding='utf8', errors='replace').readlines()

### Results Data ###
def load_code_reprs(path, chunk_size):
    logger.info(f'Loading code vectors (chunk size = {chunk_size}) ...')          
    """reads vectors (2D numpy array) from a hdf5 file"""
    codereprs = []
    #if chunk_size < 0: return np.array(tables.open_file(path).root.vecs)
    h5f  = tables.open_file(path, 'r')
    vecs = h5f.root.vecs
    if chunk_size < 0: return np.array(vecs)
    #for i in tqdm(range(0, len(vecs), chunk_size)):
    #    codereprs.append(vecs[i:i + chunk_size])
    codereprs = [vecs[i:i + chunk_size] for i in tqdm(range(0, len(vecs), chunk_size))]
    h5f.close()
    return codereprs

# added:
def load_code_reprs_lines(path, lines, chunk_size): 
    logger.info(f'Loading {len(lines)} pre-filtered code vectors ...')          
    """reads some of the vectors (2D numpy array) from a hdf5 file"""
    h5f  = tables.open_file(path)
    vecs = h5f.root.vecs
    #f    = operator.itemgetter(*lines)
    codereprs    = []
    #vector_lines = list(get_lines_generator(vecs, lines))
    #print(f'vecs.shape: {vecs.shape}')
    vector_lines = vecs[lines, ...]
    #vector_lines = list(f(vecs))
    for i in range(0, len(lines), chunk_size):
        codereprs.append(vector_lines[i:i + chunk_size])
    h5f.close()
    return codereprs #

def save_code_reprs(vecs, path):
    npvecs  = np.array(vecs)
    fvec    = tables.open_file(path, 'w')
    atom    = tables.Atom.from_dtype(npvecs.dtype)
    filters = tables.Filters(complib = 'blosc', complevel = 5)
    ds      = fvec.create_carray(fvec.root, 'vecs', atom, npvecs.shape, filters=filters)
    ds[:]   = npvecs
    fvec.close()

def load_hdf5(vecfile, start_offset, chunk_size):
    """reads training sentences(list of int array) from a hdf5 file"""  
    table    = tables.open_file(vecfile)
    data     = table.get_node('/phrases')[:].astype(np.int32)
    index    = table.get_node('/indices')[:]
    data_len = index.shape[0]
    if chunk_size == -1: # if chunk_size is set to -1, then, load all data
        chunk_size = data_len
    start_offset = start_offset % data_len    
    logger.debug("{} entries".format(data_len))
    logger.debug("starting from offset {} to {}".format(start_offset, start_offset + chunk_size))
    sents = []
    for offset in tqdm(range(start_offset, start_offset + chunk_size)):
        offset   = offset % data_len
        len, pos = index[offset]['length'], index[offset]['pos']
        sents.append(data[pos:pos + len])
    table.close()
    #print(f">>>>>>>>>>>> type(sents[0]): {type(sents[0])} | type(sents[0][0]): {type(sents[0][0])}")
    return sents 
    
# added:
def load_hdf5_lines(vecfile, lines):
    """reads specified lines of training sentences(list of int array) from a hdf5 file"""  
    table    = tables.open_file(vecfile)
    data     = table.get_node('/phrases')[:].astype(np.int32)
    index    = table.get_node('/indices')[:]
    data_len = index.shape[0]
    sents    = []
    for line in tqdm(lines):
        len, pos = index[line]['length'], index[line]['pos']
        sents.append(data[pos:pos + len])
    table.close()
    return sents #

######## database setup #########
    
def data_to_db(data_path, conf):
    #dataparts = ["apiseq", "methname", "rawcode", "tokens"]
    dataparts = ["apiseq"]
    #dataparts = ["methname", "rawcode", "tokens"]
    for part in dataparts:
        """db = UnQLite(filename = './DeepCSKeras/data/database.udb', open_database = True)
        collec = db.collection(part)
        if part == "rawcode":
            data = list(load_codebase( data_path + conf['data_params']['use_codebase'], -1))
            for i, line in tqdm(enumerate(data)):
                collec.store({'r': line.strip()})
        else:
            char = part[0]
            data = load_hdf5(data_path + conf['data_params'][f'use_{part}'], 0, -1)
            for i, line in tqdm(enumerate(data)):
                collec.store({char: line.tolist()})
        db.close()"""
        # test:
        db = UnQLite(filename = './DeepCSKeras/data/database.udb', open_database = True)
        collec = db.collection(part)
        #print(collec.fetch(177)[0])
        #print(collec.fetch(16000000)[0])
        #print(collec.fetch(collec.last_record_id())[0])
        print(collec.last_record_id())
        start = time.time()
        data = []
        for row in collec.iterator(): # TODO: Load in chunks to fix segmantation fault
            data.append(row)
        print('load time:  {:5.3f}s  <<<<<<<<<<<<<'.format(time.time()-start))
        print(data[collec.last_record_id()][0])
        #data_arrays = [pickle.loads(d[0].decode(errors='replace')) for d in data]
        if part != "rawcode": data_arrays = [np.fromiter(d[0], dtype = np.int32) for d in data]
        print('to arrays time:  {:5.3f}s  <<<<<<<<<<<<<'.format(time.time()-start))
        if part != "rawcode": print(f"len(data_arrays): {len(data_arrays)} | type(data_arrays[0]): {type(data_arrays[0])} | type(data_arrays[0][0]): {type(data_arrays[0][0])}")
        if part != "rawcode": print(data_arrays[collec.last_record_id()])
        db.close()
