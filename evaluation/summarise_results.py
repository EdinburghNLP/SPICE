import sys
import json
from glob import glob
import os
import argparse

# add arguments to parser
parser = argparse.ArgumentParser(description='Summarise evaluation results')
parser.add_argument('--file_path', default='', help='path that contains the .json files to summarise')

args = parser.parse_args()

# load constants module
from constants import *


files = glob(os.path.join(args.file_path, '*.json'))

metric = {INME_CTX: {EM: 0, 'cnt': 0},
          LARGE_CTX: {EM: 0, 'cnt': 0},
          ELLIPSIS: {EM: 0, 'cnt': 0},
          MULTI_ENTITY: {EM: 0, 'cnt': 0}}

for f in files:
    if 'eval_summary.json' in f:
        continue #skip the summary file in case it was already generated
    with open(f) as json_file:
        print(f)
        data = json.load(json_file)
        for key in data.keys():
            if MULTI_ENTITY == key and EM in data[key].keys():
                metric[MULTI_ENTITY][EM] += data[key][EM]
                metric[MULTI_ENTITY]['cnt'] += 1
            if ELLIPSIS == key and EM in data[key].keys():
                metric[ELLIPSIS][EM] += data[key][EM]
                metric[ELLIPSIS]['cnt'] += 1
            if INME_CTX == key and EM in data[key].keys():
                metric[INME_CTX][EM] += data[key][EM]
                metric[INME_CTX]['cnt'] += 1
            if LARGE_CTX == key and EM in data[key].keys():
                metric[LARGE_CTX][EM] += data[key][EM]
                metric[LARGE_CTX]['cnt'] += 1

res = {}
res[MULTI_ENTITY] = {}
res[MULTI_ENTITY][EM] = metric[MULTI_ENTITY][EM] / metric[MULTI_ENTITY]['cnt'] if metric[MULTI_ENTITY]['cnt'] > 0 else 0
res[ELLIPSIS] = {}
res[ELLIPSIS][EM] = metric[ELLIPSIS][EM] / metric[ELLIPSIS]['cnt'] if metric[ELLIPSIS]['cnt'] > 0 else 0
res[INME_CTX] = {}
res[INME_CTX][EM] = metric[INME_CTX][EM] / metric[INME_CTX]['cnt'] if metric[INME_CTX]['cnt'] > 0 else 0
res[LARGE_CTX] = {}
res[LARGE_CTX] [EM] = metric[LARGE_CTX][EM] / metric[LARGE_CTX]['cnt'] if metric[LARGE_CTX]['cnt'] > 0 else 0

# write .json file with details about the results
with open(os.path.join(args.file_path, 'eval_summary.json'), 'w') as fp:
    json.dump(res, fp)

