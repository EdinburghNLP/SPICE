import json
from glob import glob
import os
import sys
import requests
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool

from transformers import BertTokenizer

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet


from SparqlServer import SparqlServer
from SparqlResults import SparqlResults

# KB graph
SUBJECT = 'subject'
OBJECT = 'object'
TYPE = 'type'

ROOT_PATH_JSON_KG = ''
ROOT_PATH = ''
DST_ROOT_PATH = ''

# add arguments to parser
parser = argparse.ArgumentParser(description='Pre-compute types sub-graphs')
parser.add_argument('--partition', default='train', choices=['train', 'valid', 'test'], type=str, help='Partition to preprocess.')
parser.add_argument('--read_folder', default=ROOT_PATH, help='Folder to read conversations.')
parser.add_argument('--write_folder', default=DST_ROOT_PATH, help='Folder to write annotated conversations.')
parser.add_argument('--refine', default=False, action='store_true', help='Refine existing type_subgraph field. DEPRECATED')
parser.add_argument('--json_kg_folder', default=ROOT_PATH_JSON_KG, help='Folder that contains KG in .json format. used for faster annotation')

args = parser.parse_args()

# set tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize
lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words('english'))

# we'll use this for efficiency as contains pre-computed type-relations.
print('Loading KG .json files')
id_relation = json.loads(open(f'{args.json_kg_folder}/knowledge_graph/filtered_property_wikidata4.json').read())
id_entity = json.loads(open(f'{args.json_kg_folder}/knowledge_graph/items_wikidata_n.json').read())
TYPE_TRIPLES = json.loads(open(f'{args.json_kg_folder}/knowledge_graph/wikidata_type_dict.json').read())
REV_TYPE_TRIPLES = json.loads(open(f'{args.json_kg_folder}/knowledge_graph/wikidata_rev_type_dict.json').read())
print('DONE')

def loadTypeIDLabelDict():
    typeIDLabelDict = {}
    for k in TYPE_TRIPLES.keys():
        typeIDLabelDict[k] = id_entity[k]
    for k in REV_TYPE_TRIPLES.keys():
        typeIDLabelDict[k] = id_entity[k]
    return typeIDLabelDict

TYPE_ID_LABEL = loadTypeIDLabelDict()

def getTypesGraph(gold_types):
    """ Retrieve relations where the types in gold_types are either domain are range of."""
    tgraph = {}
    for t in gold_types:
        # just take the type and associated relations
        try:
            tg = [rel for rel in TYPE_TRIPLES[t].keys()]
        except KeyError:
            tg = []

        try:
            tg.extend([rel for rel in REV_TYPE_TRIPLES[t].keys()])
        except KeyError:
            pass

        tgraph[t] = list(set(tg))
    return tgraph

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = []

    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

def getLinkTypesSubgraph(utterance, existing_type_set=None):

    types = []
    ori_utt = utterance
    utterance = lemmatize_sentence(utterance).split()
    if existing_type_set:
        id_label_set = {}
        for t in existing_type_set:
            id_label_set[t] = TYPE_ID_LABEL[t]
    else:
        id_label_set = TYPE_ID_LABEL

    for k, v in id_label_set.items():
        type = lemmatize_sentence(v).split()
        type_non_stop = [w for w in type if w not in stops.union({'number'})] #number is a wd type but freq =count operator in utterances
        inter = (set(utterance) & set(type_non_stop))
        if type_non_stop and len(inter) == len(set(type_non_stop)):
            types.append((k, len(set(type_non_stop))))

    # we know that type 'people'/'Q2472587' is never used but 'common name'/'Q502895'
    foo = [x for x in types if x!= ('Q2472587', 1)]
    if len(foo) == len(types)-1:
        foo.append(('Q502895', 2))
        types = foo

    # we know that type 'occupation'/'Q528892' is never used but 'occupation'/'Q12737077'
    types = [x for x in types if x!= ('Q528892', 1)]

    # remove types that are covered by longer labels ['work of art', 'art', 'work'] ==> 'art' and 'work'
    types = sorted(types, key=lambda x: x[1], reverse=True)
    if types:
        keep = [(TYPE_ID_LABEL[types[0][0]], types[0][0])]
        for t, _ in types[1:]:
            present = False
            for l, _ in keep:
                incl = set(TYPE_ID_LABEL[t].split()).intersection(set(l.split()))
                if len(incl) == len(set(TYPE_ID_LABEL[t].split())):
                    present = True
                    break
            if not present:
                keep.append((TYPE_ID_LABEL[t], t))
    types = [t for t,_ in types if t in [k for _, k in keep]]
    types_graph = getTypesGraph(types)

    return types_graph

def getLabelJson(r):
    return id_relation[r]

splits=[args.partition]

# new splits directory
if not os.path.isdir(DST_ROOT_PATH):
    os.mkdir(DST_ROOT_PATH)
    for sp in splits:
        os.mkdir(os.path.join(DST_ROOT_PATH, sp))
    print(f'Directory "{DST_ROOT_PATH}" created')

def annotate_conversation(f):
    print(f)
    with open(f) as json_file:
        try:
            fileName = f.split('/')[-1]
            dirName = f.split('/')[-2]
            # load conversation
            conversation = json.load(json_file)
            new_conversation = []
            is_clarification = False
            turns = len(conversation) // 2
            for i in range(turns):

                if is_clarification:
                    is_clarification = False
                    continue

                user = conversation[2 * i]
                system = conversation[2 * i + 1]

                if user['question-type'] == 'Clarification':
                    new_conversation.append(user)
                    new_conversation.append(system)

                    # get next context
                    is_clarification = True
                    next_user = conversation[2 * (i + 1)]
                    next_system = conversation[2 * (i + 1) + 1]

                    if args.refine and 'type_subgraph' in next_system.keys():
                        next_system['type_subgraph'] = getLinkTypesSubgraph(user['utterance'],
                                                                            next_system['type_subgraph'])
                    else:
                        next_system['type_subgraph'] = getLinkTypesSubgraph(user['utterance'])

                    new_conversation.append(next_user)
                    new_conversation.append(next_system)
                else:
                    if args.refine and 'type_subgraph' in system.keys():
                        system['type_subgraph'] = getLinkTypesSubgraph(user['utterance'],
                                                                       system['type_subgraph'])
                    else:
                        system['type_subgraph'] = getLinkTypesSubgraph(user['utterance'])

                    new_conversation.append(user)
                    new_conversation.append(system)

            # write conversation
            assert len(conversation) == len(new_conversation)

            if not os.path.isdir(os.path.join(DST_ROOT_PATH, sp, dirName)):
                os.mkdir(os.path.join(DST_ROOT_PATH, sp, dirName))
            with open(f'{DST_ROOT_PATH}/{sp}/{dirName}/{fileName}', 'w') as formatted_json_file:
                json.dump(new_conversation, formatted_json_file, ensure_ascii=False, indent=4)

        except json.decoder.JSONDecodeError:
            print('Fail', f)

for sp in splits:
    # read data
    files = glob(f'{ROOT_PATH}/{sp}/*' + '/*.json')

    print(f'Remain to do {len(files)}')
    with Pool(20) as pool:
        res = pool.map(annotate_conversation, files) 
