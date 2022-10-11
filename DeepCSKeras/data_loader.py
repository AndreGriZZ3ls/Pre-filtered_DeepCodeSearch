import io
import math
import time
import pickle
import tables
import logging
import operator
import numpy as np
from tqdm import tqdm
from unqlite import UnQLite
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

######## database setup #########
def eval_to_db(data_path, conf):
    """dataparts = ["apiseq", "methname", "rawcode", "tokens"]
    db = UnQLite(filename = './DeepCSKeras/data/database.udb', open_database = True)
    for part in dataparts:
        source = io.open("./DeepCSKeras/data/codesearchnet/eval.{}.txt".format(part), "r", encoding='utf8', errors='replace')
        lines  = source.readlines()
        collec = db.collection(part)
        collec.create()
        if part == "rawcode":
            for i, line in enumerate(lines):
                collec.store({str(i): line.strip()})
        else:
            vocab = load_pickle(data_path + conf['data_params'][f'vocab_{part}'])
            start = time.time()
            for i, line in enumerate(lines):
                data = [vocab.get(w, 0) for w in line.strip().lower().split(' ')]
                #collec.store({str(i): data})
                data_arr = np.array(data, dtype = np.int)
                collec.store({str(i): data_arr.tolist()})
                #collec.store({str(i): pickle.dumps(data_arr, pickle.HIGHEST_PROTOCOL)})
            print('store time:  {:5.3f}s  <<<<<<<<<<<<<'.format(time.time()-start))
            #print(pickle.dumps(data_arr, pickle.HIGHEST_PROTOCOL))
            print(data_arr.tolist())
        source.close()
    db.close()"""
       
    # test:
    db = UnQLite(filename = './DeepCSKeras/data/database.udb', open_database = True)
    for part in dataparts:
        collec = db.collection(part)
        print(collec.last_record_id())
        #print(collec.fetch(99)[0])
        start = time.time()
        data = collec.all()
        print(data[collec.last_record_id()])
        print(data[collec.last_record_id()][0])
        #data_arrays = [pickle.loads(d[0].decode(errors='replace')) for d in data]
        data_arrays = [np.fromiter(d[0], dtype = np.int) for d in data]
        print('store time:  {:5.3f}s  <<<<<<<<<<<<<'.format(time.time()-start))
        print(f"len(data_arrays): {len(data_arrays)} | type(data_arrays): {type(data_arrays)} | type(data_arrays)[0]: {type(data_arrays)[0]}")
        print(data_arrays[176])
    db.close()
    
def data_to_db(data_path, conf):
    dataparts = ["apiseq", "methname", "rawcode", "tokens"]
    for part in dataparts:
        db = UnQLite(filename = './DeepCSKeras/data/database.udb', open_database = True)
        collec = db.collection(part)
        if part == "rawcode":
            data = list(load_codebase( data_path + conf['data_params']['use_codebase'], -1))
            for i, line in tqdm(enumerate(data)):
                collec.store({str(i + 177): line.strip()})
        else:
            data = load_hdf5(data_path + conf['data_params'][f'use_{part}'], 0, -1)
            for i, line in tqdm(enumerate(data)):
                collec.store({str(i + 177): line.tolist()})
        db.close()
        # test:
        db = UnQLite(filename = './DeepCSKeras/data/database.udb', open_database = True)
        #print(collec.fetch(177)[0])
        #print(collec.fetch(16000000)[0])
        #print(collec.fetch(collec.last_record_id())[0])
        print(collec.last_record_id())
        start = time.time()
        data = collec.all()
        db.close()

def load_pickle(path):
    return pickle.load(open(path, 'rb')) 

# added:
def save_pickle(path, index):
    pickle.dump(index, open(path, 'wb'), pickle.HIGHEST_PROTOCOL) #
    
def load_index_counters(path, word_list):
    h5f   = tables.open_file(path)
    index = h5f.root.index
    
def save_index(path, index):
    h5f     = tables.open_file(path, 'w')
    table   = h5file.create_table("/", 'readout', Particle, "Readout example")
    atom    = tables.Atom.from_dtype(npvecs.dtype) # TODO
    filters = tables.Filters(complib = 'blosc', complevel = 5)
    ds      = h5f.create_carray(h5f.root, 'index', atom, npvecs.shape, filters=filters) # TODO
    ds[:]   = npvecs  # TODO
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
    codes = io.open(path, encoding='utf8', errors='replace').readlines()
    if chunk_size < 0: return codes
    else:
        for i in tqdm(range(0, len(codes), chunk_size)):
            codebase.append(codes[i:i + chunk_size])            
    return codebase

# added:
def get_lines_generator(iterable, lines):
    return (line for i, line in enumerate(iterable) if i in lines) #

# added:
def load_codebase_lines(path, lines, chunk_size, chunk_number = -1): 
    """load some codebase lines
    codefile: h5 file that stores raw code
    """
    logger.info(f'Loading {len(lines)} pre-filtered codebase lines ...')
    codes = io.open(path, encoding='utf8',errors='replace')
    #codes = io.open(path, encoding='utf8',errors='replace').readlines()
    #f = operator.itemgetter(*lines)
    #codebase_lines = list(f(codes))
    codebase       = []
    start = time.time()
    if chunk_number > 0:
        offset = chunk_number * chunk_size
        for line in lines:
            line += offset
    codebase_lines = list(get_lines_generator(codes, set(lines)))
    print('Total load_codebase_lines time:  {:5.3f}s  <<<<<<<<<<<<<'.format(time.time()-start))
    #codebase_lines = codes[lines]
    if chunk_number > -1: return codebase_lines # TODO: fix (under this condition include chunk_number to correct lines)
    for i in range(0, len(lines), chunk_size):
        codebase.append(codebase_lines[i:i + chunk_size])
    return codebase #
    
def convert_codebase(path, target):
    codes = io.open(path, encoding='utf8', errors='replace').readlines()

### Results Data ###
def load_code_reprs(path, chunk_size):
    logger.info(f'Loading code vectors (chunk size = {chunk_size}) ...')          
    """reads vectors (2D numpy array) from a hdf5 file"""
    codereprs = []
    #if chunk_size < 0: return np.array(tables.open_file(path).root.vecs)
    h5f  = tables.open_file(path, 'r')
    vecs = h5f.root.vecs
    if chunk_size < 0: return np.array(vecs)
    for i in tqdm(range(0, len(vecs), chunk_size)):
        codereprs.append(vecs[i:i + chunk_size])
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
    data     = table.get_node('/phrases')[:].astype(np.int)
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
    print(f">>>>>>>>>>>> type(sents[0]): {type(sents[0])} | type(sents[0][0]): {type(sents[0][0])}")
    return sents 
    
# added:
def load_hdf5_lines(vecfile, lines):
    """reads specified lines of training sentences(list of int array) from a hdf5 file"""  
    table    = tables.open_file(vecfile)
    data     = table.get_node('/phrases')[:].astype(np.int)
    index    = table.get_node('/indices')[:]
    data_len = index.shape[0]
    sents    = []
    for line in tqdm(lines):
        len, pos = index[line]['length'], index[line]['pos']
        sents.append(data[pos:pos + len])
    table.close()
    return sents #
