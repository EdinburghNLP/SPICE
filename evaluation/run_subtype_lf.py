import os
import time
import json
import argparse
import traceback
import sys
from glob import glob
from pathlib import Path
from prettytable import PrettyTable 
import sys
import random

from executor import ActionExecutor
from meters import AccuracyMeter, F1scoreMeter
ROOT_PATH = Path(os.path.dirname(__file__)).parent

SERVER = ''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# add arguments to parser
parser = argparse.ArgumentParser(description='Execute queries')
parser.add_argument('--file_path', default='', help='json file(s) with model predictions')
parser.add_argument('--read_all', type=str2bool, nargs='?', const=True, default=False, help='list a set of files with'
                                                            'file name in --file_path is specified with * . Load all.')
parser.add_argument('--question_type', default='Clarification', help='question type to evaluate')
parser.add_argument('--max_results', default=1000, help='json file with actions')
parser.add_argument('--context_dist_file', default='', help='annotations with context distance for '
                                                            'coreference and ellipsis (indirect and incomplete types)')
parser.add_argument('--em_only', type=str2bool, nargs='?', const=True, default=False, help='only compute exact match')
parser.add_argument('--server_link', default=SERVER, help='Sparql server link. E.g., '
                                                          'http://localhost:9999/blazegraph/namespace/wd/sparq')
parser.add_argument('--out_eval_file', default='', help='output json file(s) with the evaluation results')

args = parser.parse_args()

# load constants module
from constants import *

if not args.server_link:
    print("To execute a Sparql server link is needed. E.g., http://localhost:9999/blazegraph/namespace/wd/sparq")
    exit()

# load data
if args.read_all:
    data_paths = glob(args.file_path)
else:
    data_paths = [f'{args.file_path}']
data = []
print(f'*\tEvaluating: \n{data_paths}')
for f in data_paths:
    with open(f) as json_file:
        data.extend(json.load(json_file))

# load action executor
action_executor = ActionExecutor(args.server_link)

# define question type meters
question_types_meters_1 = {
    'Clarification': F1scoreMeter(),
    'Comparative Reasoning (All)': F1scoreMeter(),
    'Logical Reasoning (All)': F1scoreMeter(),
    'Quantitative Reasoning (All)': F1scoreMeter(),
    'Simple Question (Coreferenced)': F1scoreMeter(),
    'Simple Question (Direct)': F1scoreMeter(),
    'Simple Question (Ellipsis)': F1scoreMeter(),
    # -------------------------------------------
    'Verification (Boolean) (All)': AccuracyMeter(),
    'Quantitative Reasoning (Count) (All)': AccuracyMeter(),
    'Comparative Reasoning (Count) (All)': AccuracyMeter()
}

def get_meter(qtype):
    if qtype == 'Clarification': return F1scoreMeter()
    elif qtype == 'Comparative Reasoning (All)': return F1scoreMeter()
    elif qtype == 'Logical Reasoning (All)': return F1scoreMeter()
    elif qtype == 'Quantitative Reasoning (All)': return F1scoreMeter()
    elif qtype == 'Simple Question (Coreferenced)': return F1scoreMeter()
    elif qtype == 'Simple Question (Direct)': return F1scoreMeter()
    elif qtype == 'Simple Question (Ellipsis)': return F1scoreMeter()
    elif qtype == 'Verification (Boolean) (All)': return AccuracyMeter()
    elif qtype == 'Quantitative Reasoning (Count) (All)': return AccuracyMeter()
    elif qtype == 'Comparative Reasoning (Count) (All)': return AccuracyMeter()
    else:
        raise NotImplementedError(qtype)

# If also group results/statistics per context distance (in addition to question (sub-)type).
ctx_distance = {}
if args.context_dist_file != '':
    ctx_distance_file = open(args.context_dist_file)
    for l in ctx_distance_file.readlines():
        l = l.strip()
        ctx_distance[l.split('\t')[0]] = l.split('\t')[1]

context_questions = ['Comparative|More/Less|Mult. entity type|Indirect', 'Comparative|More/Less|Single entity type|Indirect',
                     'Simple Question|Single Entity|Indirect', 'Verification|2 entities, one direct and one indirect, object is indirect',
                     'Verification|2 entities, one direct and one indirect, subject is indirect',
                     'Verification|3 entities, 2 direct, 2(direct) are query entities, subject is indirect',
                     'Verification|one entity, multiple entities (as object) referred indirectly',
                     'Quantitative|Count|Logical operators|Indirect',
                     'Quantitative|Count|Single entity type|Indirect',
                     'Comparative|Count over More/Less|Mult. entity type|Indirect',
                     'Comparative|Count over More/Less|Single entity type|Indirect']

ELLIPSIS_QSUBTYPE = ['Logical|Difference|Single_Relation|Incomplete', 'Logical|Intersection|Single_Relation|Incomplete',
                     'Logical|Union|Single_Relation|Incomplete', 'Comparative|More/Less|Mult. entity type|Incomplete',
                     'Comparative|More/Less|Single entity type|Incomplete',
                     'Incomplete|object parent is changed, subject and predicate remain same',
                     'Incomplete count-based ques', 'Comparative|Count over More/Less|Mult. entity type|Incomplete',
                     'Comparative|Count over More/Less|Single entity type|Incomplete']

MULTY_ENTITY_QSUBTYPE = ['Logical|Difference|Multiple_Relation', 'Logical|Intersection|Multiple_Relation',
                         'Logical|Union|Multiple_Relation',
                         'Quantitative|Atleast/ Atmost/ Approx. the same/Equal|Mult. entity type',
                         'Quantitative|Min/Max|Mult. entity type', 'Comparative|More/Less|Mult. entity type',
                         'Comparative|More/Less|Mult. entity type|Incomplete',
                         'Comparative|More/Less|Mult. entity type|Indirect', 'Simple Question|Mult. Entity|Indirect',
                         'Simple Question|Mult. Entity',
                         'Verification|one entity, multiple entities (as object) referred indirectly',
                         'Quantitative|Count over Atleast/ Atmost/ Approx. the same/Equal|Mult. entity type',
                         'Quantitative|Count|Mult. entity type', 'Comparative|Count over More/Less|Mult. entity type',
                         'Comparative|Count over More/Less|Mult. entity type|Incomplete',
                         'Comparative|Count over More/Less|Mult. entity type|Indirect']

question_types_meters = {}
count_no_answer = 0
count_total = 0
tic = time.perf_counter()
for i, d in enumerate(data):
    if d['question_type'] != args.question_type and args.question_type not in d['question_type']:
        continue

    print(d)

    description = d['description']
    meter_key = args.question_type + '_' + description
    
    if args.question_type not in question_types_meters.keys():
        question_types_meters[args.question_type] = get_meter(args.question_type)
    
    if meter_key not in question_types_meters.keys():
        question_types_meters[meter_key] = get_meter(args.question_type)

    # key to group per context distance
    if 'turnID' in d.keys():
        measure = 'accuracy' if args.question_type in \
            ['Verification (Boolean) (All)', 'Quantitative Reasoning (Count) (All)', 'Comparative Reasoning (Count) (All)'] \
            else 'fscore'

        if d["turnID"] in ctx_distance.keys():
            meter_key_context = f'{measure}_{ctx_distance[d["turnID"]]}'
        else:
            if description in context_questions:
                print('Not present', d["turnID"], d['question'])
            meter_key_context = f'{measure}_none'
        if meter_key_context not in question_types_meters.keys():
            question_types_meters[meter_key_context] = AccuracyMeter() if measure == 'accuracy' else F1scoreMeter()

    count_total += 1

    # take gold sparql
    if 'sparql_delex' in d.keys() and d['sparql_delex'] is not None:
        gold_sparql = d['sparql_delex']
    else:
        gold_sparql = None

    # take predicted sparql
    if d['actions'] is not None:
        pred_sparql = d['actions']

        if args.em_only:
            result, result_type = set(), None
        else:
            all_actions = pred_sparql.split()
            if 'entity' not in all_actions : # 'entity' not in all_actions is some Lasagne remaining condition?
                try:
                    result, result_type = action_executor(pred_sparql, None, d['question_type'], sparql=True)
                except Exception as ex:
                    count_no_answer += 1
                    result, result_type = None, None
                    print(ex)
                    print(traceback.format_exc())
                    print(d['question'])
                    print(f'Pred: {pred_sparql}')
                    print(f'Gold: {gold_sparql}')
            else:
                pred_sparql = None
                count_no_answer += 1
                result, result_type = None, None
    else:
        count_no_answer += 1
        result = None

    '''
    if result is None:
        result = set([])
        count_no_answer += 1
    '''
    try:
        # boolean
        if d['question_type'] == 'Verification (Boolean) (All)':
            answer = True if d['answer'] == 'YES' else False
            question_types_meters[args.question_type].update(answer, result, gold_sparql, pred_sparql)
            question_types_meters[meter_key].update(answer, result, gold_sparql, pred_sparql)
            question_types_meters[meter_key_context].update(answer, result, gold_sparql, pred_sparql)
        else:
            # numeric
            if d['question_type'] in ['Quantitative Reasoning (Count) (All)', 'Comparative Reasoning (Count) (All)']:
                if d['answer'].isnumeric():
                    if result is not None:
                        if len(result) == 1 and result[0].isnumeric():
                            r = int(result[0])
                        else:
                            r = len(result)

                        question_types_meters[d['question_type']].update(int(d['answer']), r, gold_sparql, pred_sparql)
                        question_types_meters[meter_key].update(int(d['answer']), r, gold_sparql, pred_sparql)
                        question_types_meters[meter_key_context].update(int(d['answer']), r, gold_sparql, pred_sparql)
                    else:
                        question_types_meters[d['question_type']].update(int(d['answer']), None, gold_sparql, pred_sparql)
                        question_types_meters[meter_key].update(int(d['answer']), None, gold_sparql, pred_sparql)
                        question_types_meters[meter_key_context].update(int(d['answer']), None, gold_sparql, pred_sparql)
                else:
                    print(result_type)
                    if result is not None:
                        if len(result) == 1 and result[0].isnumeric():
                            r = int(result[0])
                        else:
                            r = len(result)
                        # we need to change this too if this is still the case with some
                        question_types_meters[d['question_type']].update(len(d['results']), r, gold_sparql, pred_sparql)
                        question_types_meters[meter_key].update(len(d['results']), r, gold_sparql, pred_sparql)
                        question_types_meters[meter_key_context].update(len(d['results']), r, gold_sparql, pred_sparql)
                    else:
                        question_types_meters[d['question_type']].update(len(d['results']), None, gold_sparql, pred_sparql)
                        question_types_meters[meter_key].update(len(d['results']), None, gold_sparql, pred_sparql)
                        question_types_meters[meter_key_context].update(len(d['results']), None, gold_sparql, pred_sparql)
            else:
                # set
                if result is None or isinstance(result, (int)):
                    result = set([])
                result = set(result)
                if result != set(d['results']) and len(result) > args.max_results:
                    new_result = result.intersection(set(d['results']))
                    for res in result:
                        if res not in result: new_result.add(res)
                        if len(new_result) == args.max_results: break
                    result = new_result.copy()

                gold = set(d['results']) if not args.em_only else set()
                if type(d['question_type']) == list:
                    question_types_meters[args.question_type].update(gold, result, gold_sparql, pred_sparql)
                else:
                    question_types_meters[d['question_type']].update(gold, result, gold_sparql, pred_sparql)
                question_types_meters[meter_key].update(gold, result, gold_sparql, pred_sparql)
                question_types_meters[meter_key_context].update(gold, result, gold_sparql, pred_sparql)
    except Exception as ex:
        print(d['question'])
        print(d['actions'])
        raise ValueError(ex)

    toc = time.perf_counter()

# print results
res = {}
acc_coref_1, acc_coref_1_inst = 0.0, 0.0
acc_coref_gt1, acc_coref_gt1_cnt, acc_coref_gt1_inst = 0.0, 0.0, 0.0
f1_coref_1, f1_coref_1_inst = 0.0, 0.0
f1_coref_gt1, f1_coref_gt1_cnt, f1_coref_gt1_inst = 0.0, 0.0, 0.0
em_ellipsis, em_ellipsis_cnt, em_ellipsis_inst = 0.0, 0.0, 0.0
em_multi_entity, em_multi_entity_cnt, em_multi_entity_inst = 0.0, 0.0, 0.0
print(args.question_type)
print(f'NA actions: {count_no_answer}')
print(f'Total samples: {count_total}')
if args.question_type in ['Verification (Boolean) (All)', 'Quantitative Reasoning (Count) (All)', 'Comparative Reasoning (Count) (All)']:
    myTable = PrettyTable(["Type", "#Instances", "Accuracy", "EM"])
    for key in question_types_meters.keys():
        print(key)
        print(f'Number of Instances: {question_types_meters[key].number_of_instance}')
        print(f'Accuracy: {question_types_meters[key].accuracy}')
        myTable.add_row([key, question_types_meters[key].number_of_instance, question_types_meters[key].accuracy, question_types_meters[key].exact_match_acc]) 

        res[key] = {INSTANCES: question_types_meters[key].number_of_instance,
                    ACCURACY: question_types_meters[key].accuracy,
                    EM: question_types_meters[key].exact_match_acc}

        if 'accuracy_' in key:
            parts = key.split('accuracy_')[1]
            if parts == '1':
                acc_coref_1 = question_types_meters[key].exact_match_acc
                acc_coref_1_inst = question_types_meters[key].number_of_instance
            elif parts != 'none':
                acc_coref_gt1 += question_types_meters[key].exact_match_acc
                acc_coref_gt1_inst += question_types_meters[key].number_of_instance
                acc_coref_gt1_cnt += 1

        keyStr = key.split(args.question_type + '_')[1] if len(key.split(args.question_type + '_')) > 1 else None
        if keyStr in ELLIPSIS_QSUBTYPE:
            em_ellipsis += question_types_meters[key].exact_match_acc
            em_ellipsis_cnt +=1
            em_ellipsis_inst += question_types_meters[key].number_of_instance
        if keyStr in MULTY_ENTITY_QSUBTYPE:
            em_multi_entity += question_types_meters[key].exact_match_acc
            em_multi_entity_cnt +=1
            em_multi_entity_inst += question_types_meters[key].number_of_instance

else:
    myTable = PrettyTable(["Type", "#Instances", "Precision", "Recall", "F1 Score", "EM", "Macro F1 Score", "Missmatch"])
    for key in question_types_meters.keys(): 
        myTable.add_row([key, question_types_meters[key].number_of_instance, \
                         question_types_meters[key].precision, \
                         question_types_meters[key].recall, \
                         question_types_meters[key].f1_score, \
                         question_types_meters[key].exact_match_acc, \
                         question_types_meters[key].acc_f1_macro / question_types_meters[key].number_of_instance, \
                         question_types_meters[key].missmatch])
        print(key)
        print(f'Number of Instances: {question_types_meters[key].number_of_instance}')
        print(f'Precision: {question_types_meters[key].precision}')
        print(f'Precision-2: {question_types_meters[key].acc_prec_macro / question_types_meters[key].number_of_instance}')
        print(f'Recall: {question_types_meters[key].recall}')
        print(f'F1-score: {question_types_meters[key].f1_score}')
        print(f'F1-score-2: {question_types_meters[key].acc_f1_macro / question_types_meters[key].number_of_instance}')


        res[key] = {INSTANCES : question_types_meters[key].number_of_instance,
                    PRECISION: question_types_meters[key].precision,
                    RECALL: question_types_meters[key].recall,
                    F1SCORE: question_types_meters[key].f1_score,
                    MACRO_F1SCORE: question_types_meters[key].acc_f1_macro / question_types_meters[key].number_of_instance,
                    EM: question_types_meters[key].exact_match_acc}

        if 'fscore_' in key:
            parts = key.split('fscore_')[1]
            if parts == '1':
                f1_coref_1 = question_types_meters[key].exact_match_acc
                f1_coref_1_inst = question_types_meters[key].number_of_instance
            elif parts != 'none':
                f1_coref_gt1 += question_types_meters[key].exact_match_acc
                f1_coref_gt1_cnt +=1
                f1_coref_gt1_inst += question_types_meters[key].number_of_instance

        keyStr = key.split(args.question_type + '_')[1] if len(key.split(args.question_type + '_')) > 1 else None
        if keyStr in ELLIPSIS_QSUBTYPE:
            em_ellipsis += question_types_meters[key].exact_match_acc
            em_ellipsis_cnt +=1
            em_ellipsis_inst += question_types_meters[key].number_of_instance
        if keyStr in MULTY_ENTITY_QSUBTYPE:
            em_multi_entity += question_types_meters[key].exact_match_acc
            em_multi_entity_cnt +=1
            em_multi_entity_inst += question_types_meters[key].number_of_instance

print(myTable)

if args.context_dist_file != '':
    if args.question_type in \
            ['Verification (Boolean) (All)', 'Quantitative Reasoning (Count) (All)',
             'Comparative Reasoning (Count) (All)']:
        print(f'{INME_CTX}: {acc_coref_1}')
        print(f'{LARGE_CTX}: {acc_coref_gt1/acc_coref_gt1_cnt if acc_coref_gt1_cnt > 0 else 0}')
        res[INME_CTX] = {INSTANCES: acc_coref_1_inst, EM: acc_coref_1}
        res[LARGE_CTX] = {INSTANCES: acc_coref_gt1_inst, EM: acc_coref_gt1/acc_coref_gt1_cnt if acc_coref_gt1_cnt > 0 else 0}
    else:
        print(f'{INME_CTX}: {f1_coref_1}')
        print(f'{LARGE_CTX}: {f1_coref_gt1 / f1_coref_gt1_cnt if f1_coref_gt1_cnt > 0 else 0}')
        res[INME_CTX] = {INSTANCES: f1_coref_1_inst, EM: f1_coref_1}
        res[LARGE_CTX] = {INSTANCES: f1_coref_gt1_inst, EM: f1_coref_gt1 / f1_coref_gt1_cnt if f1_coref_gt1_cnt > 0 else 0 }

if em_ellipsis_cnt > 0:
    res[ELLIPSIS] = {INSTANCES: em_ellipsis_inst, EM: em_ellipsis / em_ellipsis_cnt if em_ellipsis_cnt > 0 else 0}

if em_multi_entity_cnt > 0:
    res[MULTI_ENTITY] = {INSTANCES: em_multi_entity_inst, EM: em_multi_entity / em_multi_entity_cnt if em_multi_entity_cnt > 0 else 0}

# write .json file with details about the results
with open(args.out_eval_file, 'w') as fp:
    json.dump(res, fp)
