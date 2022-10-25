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



def lineariseAsTriples(subgraph):
    label_triples = []
    wikidata_entity_triples = []
    entity_label_dict = {}
    entity_type_dict = {}
    all_entity_set = set()
    all_entity_relation_set = set()
    all_rel_set = set()
    all_type_set = set()
    edge_set = set()
    all_ent_rel_types_text = []

def preprocess_file(data_file):
    try:
        return _preprocess_file(data_file)
    except Exception as e:
        print(traceback.format_exc())
        print('Failed ', data_file)
        return 0


def _preprocess_file(data_file):
    
    try:
        data = json.load(open(data_file, 'r'))
    except json.decoder.JSONDecodeError as e:
        print('Failed loading json file: ', data_file)
        raise e
   
    user_tp = []
    user_fp = []
    user_fn = []
    sys_tp = []
    sys_fp = []
    sys_fn = []
    for conversation in [data]:
        
        turns = len(conversation) // 2

        for i in range(turns):            
            
            user = conversation[2*i]
            system = conversation[2*i + 1]
            if 'entities_in_utterance' in user.keys() or 'entities' in user.keys():
                user_tags = set(user['es_links'])
                if 'entities_in_utterance' in user.keys():
                    user_gold = set(user['entities_in_utterance'])
                else:
                    user_gold = set(user['entities'])
                utp = len(user_tags.intersection(user_gold))
                ufp = len(user_tags.difference(user_gold))
                ufn = len(user_gold.difference(user_tags))
                user_tp.append(utp)
                user_fp.append(ufp)
                user_fn.append(ufn)
            #else:
            #   print(user['utterance'], data_file)


            if 'entities_in_utterance' in system.keys():
                system_tags = set(system['es_links'])
                system_gold = set(system['entities_in_utterance'])
                utp = len(system_tags.intersection(system_gold))
                ufp = len(system_tags.difference(system_gold))
                ufn = len(system_gold.difference(system_tags))
                sys_tp.append(utp)
                sys_fp.append(ufp)
                sys_fn.append(ufn)
            else:
                print('entities_in_utterance not present', data_file)

    return user_tp, user_fp, user_fn, sys_tp, sys_fp, sys_fn


def main(args):
    if (args.dataset != ''):
        split_path = args.data_path + f'/{args.dataset}/*'
        split_files = glob(split_path + '/*.tagged')
        if args.debug:
            split_files = split_files[:500]
        print('Loading files from ', split_path, len(split_files))

        corpora = {f'{args.dataset}': split_files}
    else:
        # do all
        train_path = args.data_path + '/train/*'
        val_path = args.data_path + '/valid/*'
        test_path = args.data_path + '/test/*'
        train_files = glob(train_path + '/*.tagged')
        print('Train files ', train_path, len(train_files))
        valid_files = glob(val_path + '/*.tagged')
        print('Valid files ', val_path, len(valid_files))
        test_files = glob(test_path + '/*.tagged')
        print('Test files ', test_path, len(test_files))
        corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}

    for corpus_type in corpora.keys():
        #a_lst = [(f, args, csqa, corpus_type) for f in corpora[corpus_type]]
        filelist = corpora[corpus_type]
        pool = Pool(args.n_cpus)
        user_tp = []
        user_fp = []
        user_fn = []
        sys_tp = []
        sys_fp = []
        sys_fn = []
        for processed_input in tqdm(pool.imap_unordered(preprocess_file, filelist), total=len(filelist)):
            p = processed_input
            user_tp.extend(p[0])
            user_fp.extend(p[1])
            user_fn.extend(p[2])
            sys_tp.extend(p[3])
            sys_fp.extend(p[4])
            sys_fn.extend(p[5])

        pool.close()
        pool.join()
        utp = sum(user_tp)
        ufp = sum(user_fp)
        ufn = sum(user_fn)
        stp = sum(sys_tp)
        sfp = sum(sys_fp)
        sfn = sum(sys_fn)
        uprec = utp / (utp + ufp)
        urecall = utp / (utp + ufn)
        sprec = stp / (stp + sfp)
        srecall = stp / (stp + sfn)
        uf1 = (2 * uprec * urecall) / (uprec + urecall)
        sf1 = (2 * sprec * srecall) / (sprec + srecall)
        print('User F1: {f1} precision {p} recall {r}  '.format(f1=uf1,p=uprec,r=urecall))
        print('Sys F1: {f1} precision {p} recall {r}  '.format(f1=sf1,p=sprec,r=srecall))
        

           


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # python preprocess.py -data_path /home/s1959796/csqparsing/dataset/data/version_splits -save_path /home/s1959796/csqparsing/processed_data -dataset train -debug 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", required=True)
    parser.add_argument('-dataset', default='')
    parser.add_argument('-n_cpus', default=10, type=int)
    parser.add_argument('-debug', nargs='?',const=False,default=False)
    args = parser.parse_args()
    main(args)

