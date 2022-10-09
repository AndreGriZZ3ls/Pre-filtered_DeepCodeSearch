import os
import io
import re
import sys
import math
import time
import shutil
import codecs
import argparse
import operator
import traceback
import fileinput
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.stem import PorterStemmer
from DeepCodeSearchBT.DeepCSKeras import data_loader

#################################################################################################
camelcase_file = "./DeepCodeSearchBT/DeepCSKeras/data/codesearchnet/allData/eval.methname.all.txt"
deduplic_file  = "./DeepCodeSearchBT/DeepCSKeras/data/codesearchnet/eval.tokens.txt"

source_file1   = "./DeepCodeSearchBT/DeepCSKeras/data/codesearchnet/allData/eval_filter.txt"
source_file2   = "./DeepCodeSearchBT/DeepCSKeras/data/codesearchnet/allData/eval.URLs.txt"
target_file1   = "./DeepCodeSearchBT/DeepCSKeras/data/codesearchnet/eval_filter.pkl"

source_file3   = "./DeepCodeSearchBT/DeepCSKeras/data/codesearchnet/allData/eval.{}.all.txt"
target_file3   = "./DeepCodeSearchBT/DeepCSKeras/data/codesearchnet/eval.{}.txt"
#################################################################################################

def replace_camelcase():
    #f     = io.open(camelcase_file, "rb", encoding='utf8', errors='replace')
    #lines = f.readlines()
    f = fileinput.FileInput(camelcase_file, inplace=1)
    for line in f:
        line = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', line)
        print(line.strip().lower())
    f.close()

def deduplicate_preserving_order():
    f = fileinput.FileInput(deduplic_file, inplace=1)
    for line in f:
        line   = line.strip().split()
        result = sorted(set(line), key = line.index)
        print(" ".join(result))
    f.close()

def find_line_numbers_and_build_eval_dict():
    i = 0
    positions  = []
    eval_dict  = dict()
    eval_file  = io.open(source_file1, "r", encoding='utf8', errors='replace')
    eval_lines = eval_file.readlines()
    data_file  = io.open(source_file2, "r", encoding='utf8', errors='replace')
    data_lines = list(data_file.readlines())
    #print(len(data_lines))
    #print(data_lines[0:10])
    for line in eval_lines:
        eval_list = line.split(",")
        query     =     eval_list[0]
        url       =     eval_list[1] + '\n'
        rating    = int(eval_list[2])
        try:
            positions.append(data_lines.index(url))
            if query in eval_dict:
                eval_dict[query][i] = rating
            else:
                sub_dict = dict()
                sub_dict[i] = rating
                eval_dict[query] = sub_dict
            i += 1
        except ValueError:
            print(f"URL '{url}' not found! Skipping.")
            pass
    print(positions)
    data_loader.save_pickle(target_file1, eval_dict)
    eval_file.close()
    data_file.close()
    print(eval_dict)

"""def copy_data():
    positions = [2542, 430612, 432091, 102856, 402542, 244285, 44924, 210828, 154337, 401612, 447498, 415112, 134952, 97746, 439252, 190099, 226348, 129085, 294296, 313603, 95381, 195761, 422612, 45573, 308671, 285036, 276124, 193241, 32797, 272599, 388536, 202689, 243968, 450641, 437179, 244707, 25779, 434684, 92471, 349908, 328489, 244084, 22081, 432026, 123801, 434671, 396731, 192317, 385261, 448421, 243666, 425188, 435251, 364462, 373237, 170710, 44680, 404826, 49402, 399595, 34103, 451532, 31790, 243299, 134953, 197699, 163296, 218952, 277441, 15359, 420320, 75727, 284914, 272598, 201745, 94569, 244165, 81966, 160113, 304351, 290253, 94242, 327162, 312563, 272595, 276124, 31727, 400787, 272597, 249938, 333265, 343133, 163295, 335617, 220620, 387569, 198187, 450637, 411566, 161285, 44923, 278556, 383275, 205768, 328531, 243662, 335105, 94571, 345889, 312564, 186427, 94512, 344453, 316300, 125392, 188255, 438420, 348535, 49763, 449380, 347646, 45391, 193191, 360867, 327163, 276141, 199742, 396273, 451529, 94933, 161296, 275924, 290010, 336911, 241195, 49589, 242905, 188604, 401, 2543, 125392, 236038, 97639, 280876, 335249, 94330, 243760, 242034, 311792, 195770, 49539, 422591, 132074, 352849, 241678, 45367, 20555, 12362, 103364, 4336, 420013, 161045, 438420, 263250, 94570, 312640, 241533, 169612, 353645, 420014, 192932, 149075, 242048, 889, 241532, 170630, 252516]
    dataparts = ["apiseq", "methname", "rawcode", "tokens"]
    for part in dataparts:
        source = io.open(source_file3.format(part), "r", encoding='utf8', errors='replace')
        target = io.open(target_file3.format(part), "w", encoding='utf8', errors='replace')
        lines = source.readlines()
        for pos in positions:
            target.write(lines[pos])
        source.close()
        target.close()"""
        

if __name__ == '__main__':
    #replace_camelcase()
    #find_line_numbers_and_build_eval_dict()
    #copy_data()
    #deduplicate_preserving_order()
    