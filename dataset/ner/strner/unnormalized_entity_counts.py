from concurrent.futures import process
import math
import sys
from glob import glob
from multiprocess import Pool
import json
from multiprocessing.dummy import Pool as ThreadPool
#import matplotlib.pyplot as plt
import numpy as np
import traceback
from collections import OrderedDict
import pickle
import argparse
import gc
from tqdm import tqdm
import math

RELARGS_DELIMITER_BERT = '[SEP]'


class F1scoreMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.exact_match_acc = 0
        self.correct_exact_match = 0.0
        self.number_of_instance = 0.0

    def update(self, gold, result):
        self.number_of_instance += 1
        
        self.tp += len(result.intersection(gold))
        self.fp += len(result.difference(gold))
        self.fn += len(gold.difference(result))
        if self.tp > 0 or self.fp > 0:
            self.precision = self.tp / (self.tp + self.fp)
        if self.tp > 0 or self.fn > 0:
            self.recall = self.tp / (self.tp + self.fn)
        if self.precision > 0 or self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        
        self.exact_match_acc = self.correct_exact_match / self.number_of_instance




def count_file(data_file):
    try:
        return _getcounts(data_file)
    except Exception as e:
        print(traceback.format_exc())
        print('Failed ', data_file)
        return 0

def _getcounts(data_file):
    
    try:
        data = json.load(open(data_file, 'r'))
    except json.decoder.JSONDecodeError as e:
        print('Failed loading json file: ', data_file)
        raise e
   
    entity_counts = {}
    def add_to_dict(e):
        if e in entity_counts.keys():
            entity_counts[e] += 1
        else:
            entity_counts[e] = 1
    
    for conversation in [data]:
        
        turns = len(conversation) // 2

        for i in range(turns):            
            
            user = conversation[2*i]
            system = conversation[2*i + 1]
            if 'entities_in_utterance' in user.keys() or 'entities' in user.keys():
                if 'entities_in_utterance' in user.keys():
                    user_gold = user['entities_in_utterance']
                    for e in user_gold:
                        add_to_dict(e)
                else:
                    user_gold = set(user['entities'])
                    for e in user_gold:
                        add_to_dict(e)
                
            #   print(user['utterance'], data_file)


            if 'entities_in_utterance' in system.keys():
                system_gold = set(system['entities_in_utterance'])
                for e in system_gold:
                    add_to_dict(e)
            else:
                print('entities_in_utterance not present', data_file)

    return entity_counts


def main(args):
    global_entity_counts= {}
    def add_to_global_dict(e, c):
        if e in global_entity_counts.keys():
            global_entity_counts[e] += c
        else:
            global_entity_counts[e] = c
    
    if (args.dataset != ''):
        split_path = args.data_path + f'/{args.dataset}/*'
        split_files = glob(split_path + '/*.json')
        if args.debug:
            split_files = split_files[:500]
        print('Loading files from ', split_path, len(split_files))

        corpora = {f'{args.dataset}': split_files}
    else:
        # do all
        train_path = args.data_path + '/train/*'
        val_path = args.data_path + '/valid/*'
        test_path = args.data_path + '/test/*'
        train_files = glob(train_path + '/*.json')
        print('Train files ', train_path, len(train_files))
        valid_files = glob(val_path + '/*.json')
        print('Valid files ', val_path, len(valid_files))
        test_files = glob(test_path + '/*.json')
        print('Test files ', test_path, len(test_files))
        corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}

    for corpus_type in corpora.keys():
        #a_lst = [(f, args, csqa, corpus_type) for f in corpora[corpus_type]]
        filelist = corpora[corpus_type]
        pool = Pool(args.n_cpus)
        for entity_count in tqdm(pool.imap_unordered(count_file, filelist), total=len(filelist)):
            for e, c in entity_count.items():
                add_to_global_dict(e, c)


        pool.close()
        pool.join()
        
    json.dump(global_entity_counts, open('entity_count.json', 'w', encoding='utf8'), indent=2, ensure_ascii=False)
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # python unnormalized_entity_counts.py -data_path /home/s1959796/csqparsing/dataversion_aug27_2022/CSQA_v9_skg.v6_compar_spqres9_subkg2_tyTop_nelctx_cleaned/
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", required=True)
    parser.add_argument('-dataset', default='')
    parser.add_argument('-n_cpus', default=10, type=int)
    parser.add_argument('-debug', nargs='?',const=False,default=False)
    args = parser.parse_args()
    main(args)


