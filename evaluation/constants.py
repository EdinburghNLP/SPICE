import os
import torch
import re
from pathlib import Path

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# model name
MODEL_NAME = 'BertSP'

# fields
INPUT = 'input'
STR_LOGICAL_FORM = 'str_logical_form'
LOGICAL_FORM = 'logical_form'
GRAPH = 'graph'
NEL = 'nel'
SEGMENT = 'segment'
START = 'start'
END = 'end'
DYNBIN = 'dynbin'
ENTITY_MAP = 'entity_map'
GLOBAL_SYNTAX = 'global_syntax'

# json annotations fields, used in precompute_local_subgraphs.py
ES_LINKS = 'es_links'
SPANS = 'tagged_words'
ALLEN_SPANS = 'allen_tagged_words'
ALLEN_TAGS = 'allennlp_tags'
ALLEN_ES_LINKS = 'allen_es_links'
STR_ES_LINKS = 'str_es_links'
STR_SPANS = 'str_tagged_words'

# helper tokens
BOS_TOKEN = '[BOS]'
BOS_TOKEN_BERT = '[unused6]'
EOS_TOKEN = '[EOS]'
EOS_TOKEN_BERT = '[unused7]'
CTX_TOKEN = '[CTX]'
CTX_TOKEN_BERT = '[unused2]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
NA_TOKEN = 'NA'
NA_TOKEN_BERT = '[unused3]'
OBJ_TOKEN = '[OBJ]'
SUBJ_TOKEN = '[SUBJ]'
PRED_TOKEN = '[PRED]'
TYPE_TOKEN = '[TYPE]'
KGELEM_DELIMITER = ';'
KGELEM_DELIMITER_BERT = '[unused4]'
TMP_DELIMITER = '||||'
RELARGS_DELIMITER = '[->]'
RELARGS_DELIMITER_BERT = '[unused5]'

# KB graph
SUBJECT = 'subject'
OBJECT = 'object'
TYPE = 'type'
TYPE_SUBJOBJ = 'typesubjobj'

# ner tag
B = 'B'
I = 'I'
O = 'O'

# question types
TOTAL = 'total'
OVERALL = 'Overall'
CLARIFICATION = 'Clarification'
COMPARATIVE = 'Comparative Reasoning (All)'
LOGICAL = 'Logical Reasoning (All)'
QUANTITATIVE = 'Quantitative Reasoning (All)'
SIMPLE_COREFERENCED = 'Simple Question (Coreferenced)'
SIMPLE_DIRECT = 'Simple Question (Direct)'
SIMPLE_ELLIPSIS = 'Simple Question (Ellipsis)'
VERIFICATION = 'Verification (Boolean) (All)'
QUANTITATIVE_COUNT = 'Quantitative Reasoning (Count) (All)'
COMPARATIVE_COUNT = 'Comparative Reasoning (Count) (All)'

# action related
ENTITY = 'entity'
RELATION = 'relation'
TYPE = 'type'
VALUE = 'value'
ACTION = 'action'

# other
UTTERANCE = 'utterance'
QUESTION_TYPE = 'question_type'
DESCRIPTION = 'description'
IS_CORRECT = 'is_correct'
QUESTION = 'question'
ANSWER = 'answer'
ACTIONS = 'actions'
GOLD_ACTIONS = 'sparql_delex'
RESULTS = 'results'
PREV_RESULTS = 'prev_results'
CONTEXT_QUESTION = 'context_question'
CONTEXT_ENTITIES = 'context_entities'
BERT_BASE_UNCASED = 'bert-base-uncased'
TURN_ID = 'turnID'
USER = 'USER'
SYSTEM = 'SYSTEM'

# ENTITY and TYPE annotations options, defined in preprocess.py
TGOLD = 'gold'
TLINKED = 'linked'
TNONE = 'none'
NEGOLD = 'gold'
NELGNEL = 'lgnel'
NEALLENNEL = 'allennel'
NESTRNEL = 'strnel'

# max limits, truncations in inputs used in data_builder.py
MAX_TYPE_RESTRICTIONS = 5
MAX_LINKED_TYPES = 3 # graph from type linking, we know in average there are 2.3 gold types
MAX_INPUTSEQ_LEN = 508

# Eval script output json keys
INSTANCES = 'instances'
ACCURACY = 'accuracy'
PRECISION = 'precision'
RECALL = 'recall'
F1SCORE = 'f1score'
MACRO_F1SCORE = 'macro-f1score'
EM = 'em'
INME_CTX = 'Ctx=-1'
LARGE_CTX = 'Ctx<-1'
ELLIPSIS = 'ellipsis'
MULTI_ENTITY = 'multi_entity'

QTYPE_DICT = {
'Comparative Reasoning (All)': 0,
'Logical Reasoning (All)': 1,
'Quantitative Reasoning (All)': 2,
'Simple Question (Coreferenced)': 3,
'Simple Question (Direct)': 4,
'Simple Question (Ellipsis)': 5,
'Verification (Boolean) (All)': 6,
'Quantitative Reasoning (Count) (All)': 7,
'Comparative Reasoning (Count) (All)': 8,
'Clarification': 9
}

INV_QTYPE_DICT = {}
for k, v in QTYPE_DICT.items():
    INV_QTYPE_DICT[v] = k


def get_value(question):
    if 'min' in question.split():
        value = '0'
    elif 'max' in question.split():
        value = '0'
    elif 'exactly' in question.split():
        value = re.search(r'\d+', question.split('exactly')[1])
        if value:
            value = value.group()
    elif 'approximately' in question.split():
        value = re.search(r'\d+', question.split('approximately')[1])
        if value:
            value = value.group()
    elif 'around' in question.split():
        value = re.search(r'\d+', question.split('around')[1])
        if value:
            value = value.group()
    elif 'atmost' in question.split():
        value = re.search(r'\d+', question.split('atmost')[1])
        if value:
            value = value.group()
    elif 'atleast' in question.split():
        value = re.search(r'\d+', question.split('atleast')[1])
        if value:
            value = value.group()
    else:
        print(f'Could not extract value from question: {question}')
        value = '0'

    return value