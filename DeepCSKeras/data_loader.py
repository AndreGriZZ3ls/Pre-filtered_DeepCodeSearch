import pickle
import codecs
import tables
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))    

##### Data Set #####
def load_codebase(path, chunk_size):
    """load codebase
    codefile: h5 file that stores raw code
    """
    logger.info('Loading codebase (chunk size={})..'.format(chunk_size))
    codebase=[]
    #codes=codecs.open(self.path+self.data_params['use_codebase']).readlines()
    codes=codecs.open(path, encoding='utf8',errors='replace').readlines()
        #use codecs to read in case of encoding problem
    for i in tqdm(range(0,len(codes), chunk_size)):
        codebase.append(codes[i:i+chunk_size])            
    return codebase

# added:
def load_codebase_lines(path, lines): 
    """load some codebase lines
    codefile: h5 file that stores raw code
    """
    logger.info('Loading pre-filtered codebase liens ...')
    codebase=[]
    #codes=codecs.open(self.path+self.data_params['use_codebase']).readlines()
    codes=codecs.open(path, encoding='utf8',errors='replace').readlines()
        #use codecs to read in case of encoding problem
    for line in tqdm(lines):
        codebase.append(codes[line])            
    return codebase #

### Results Data ###
def load_code_reprs(path, chunk_size):
    logger.debug(f'Loading code vectors (chunk size={chunk_size})..')          
    """reads vectors (2D numpy array) from a hdf5 file"""
    codereprs=[]
    h5f = tables.open_file(path)
    vecs = h5f.root.vecs
    for i in range(0, len(vecs), chunk_size):
        codereprs.append(vecs[i: i+ chunk_size])
    h5f.close()
    return codereprs

# added:
def load_code_reprs_lines(path, lines): 
    logger.debug(f'Loading specific code vectors (those listed in lines)..')          
    """reads some of the vectors (2D numpy array) from a hdf5 file"""
    codereprs=[[]]
    h5f = tables.open_file(path)
    vecs = h5f.root.vecs
    for line in tqdm(lines):
        codereprs[0].append(vecs[line])
    h5f.close()
    return codereprs #

def save_code_reprs(vecs, path):
    npvecs=np.array(vecs)
    fvec = tables.open_file(path, 'w')
    atom = tables.Atom.from_dtype(npvecs.dtype)
    filters = tables.Filters(complib = 'blosc', complevel=5)
    ds = fvec.create_carray(fvec.root, 'vecs', atom, npvecs.shape,filters=filters)
    ds[:] = npvecs
    fvec.close()

def load_hdf5(vecfile, start_offset, chunk_size):
    """reads training sentences(list of int array) from a hdf5 file"""  
    table = tables.open_file(vecfile)
    data = table.get_node('/phrases')[:].astype(np.int)
    index = table.get_node('/indices')[:]
    data_len = index.shape[0]
    if chunk_size == -1:#if chunk_size is set to -1, then, load all data
        chunk_size = data_len
    start_offset = start_offset % data_len    
    logger.debug("{} entries".format(data_len))
    logger.debug("starting from offset {} to {}".format(start_offset, start_offset + chunk_size))
    sents = []
    for offset in tqdm(range(start_offset, start_offset + chunk_size)):
        offset = offset % data_len
        len, pos = index[offset]['length'], index[offset]['pos']
        sents.append(data[pos:pos + len])
    table.close()
    return sents 
    
def load_hdf5_lines(vecfile, lines):
    """reads specified lines of training sentences(list of int array) from a hdf5 file"""  
    table = tables.open_file(vecfile)
    data = table.get_node('/phrases')[:].astype(np.int)
    index = table.get_node('/indices')[:]
    data_len = index.shape[0]
    sents = []
    for line in lines:
        len, pos = index[line]['length'], index[line]['pos']
        sents.append(data[pos:pos + len])
    table.close()
    return sents 
