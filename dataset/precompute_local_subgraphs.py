import json
from glob import glob
import os
import sys
import requests
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime

from transformers import BertTokenizer

from SparqlServer import SparqlServer
from SparqlResults import SparqlResults

# constants
SUBJECT = 'subject'
OBJECT = 'object'
TYPE = 'type'
ES_LINKS = 'es_links'
SPANS = 'tagged_words'
ALLEN_SPANS = 'allen_tagged_words'
ALLEN_TAGS = 'allennlp_tags'
ALLEN_ES_LINKS = 'allen_es_links'
STR_ES_LINKS = 'str_es_links'
STR_SPANS = 'str_tagged_words'

ROOT_PATH_JSON_KG = ''
ROOT_PATH = ''
DST_ROOT_PATH = ''

ROOT_PATH_COMBI_ALLEN = '' # from where to take off-the-shelf NELs
ROOT_PATH_COMBI_STR = '' # from where to take string based NELs


# add arguments to parser
parser = argparse.ArgumentParser(description='Pre-compute entity neighbourhood sub-graphs')
parser.add_argument('--partition', default='train', choices=['train', 'valid', 'test'], type=str, help='partition to preprocess')
parser.add_argument('--read_folder', default=ROOT_PATH, help='Folder to read conversations.')
parser.add_argument('--write_folder', default=DST_ROOT_PATH, help='Folder to write the annotated conversations.')
parser.add_argument('--json_kg_folder', default=ROOT_PATH_JSON_KG, help='Folder that contains KG in .json format. used for faster annotation')
parser.add_argument('--allennlpNER_folder', default=ROOT_PATH_COMBI_ALLEN, help='Folder from where to read conversations '
                                                                                'annotated with AllenNLP NER +NEL. Note that'
                                                                                'these conversations are expected to have .tagged extension.')
parser.add_argument('--strNEL_folder', default=ROOT_PATH_COMBI_STR, help='Folder from where to read conversations '
                                                                'annotated with String-Match NER. Note that these converstaions'
                                                                'are expected to have .strtaggedwithoutproperty extension.')
parser.add_argument('--part', default='none', choices=['first', 'second', 'third', 'fourth', 'none'], type=str,
                                                                help='split conversation processing by file ID in groups.')
parser.add_argument('--task', default='expansion', choices=['expansion', 'vocab'], help='either build vocab or expansion graphs')
parser.add_argument('--nel_entities', default=False, action='store_true', help='build expansion graphs for NEL.'
                                                                'We assume we have already done expansion graph on GOLD entities,'
                                                                'so NEL expansion graph will just complete with the entities introduced'
                                                                'by NEL.')

args = parser.parse_args()
print(args)

vocab_file = os.path.join(args.write_folder, 'expansion_vocab.json')

# set tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize

id_relation = json.loads(open(f'{args.json_kg_folder}/knowledge_graph/filtered_property_wikidata4.json').read())
id_entity = json.loads(open(f'{args.json_kg_folder}/knowledge_graph/items_wikidata_n.json').read())

SERVER = SparqlServer.instance()

cache = {}

def getSubgraph(utteranceEntities, local_subgraph):

    def associatedRelations(result, rel, typ):
        relations = {}
        for x, y in zip(SparqlResults.getEntitySetFromBindings(result)[rel],
                SparqlResults.getEntitySetFromBindings(result)[typ]):
            if not (x in IDs and y in IDs):
                continue
            if x not in relations.keys():
                relations[x] = {'label': " ".join(GRAPH_vocab[x]), 'type_restriction': []}
            if not (y, " ".join(GRAPH_vocab[y])) in relations[x]['type_restriction']:
                    relations[x]['type_restriction'].append((y, " ".join(GRAPH_vocab[y])))
        return relations

    entityGraph = {}

    for ent in utteranceEntities:
        if ent in local_subgraph.keys():
            print(f'****** Entity in GOLD set: {ent}')
            continue
        print(f'%%%%%% New NEL entity: {ent}')

        entityGraph[ent] = {}
        #query_entsubj_str = 'SELECT ?r ?t WHERE { wd:' + ent + ' ?r ?o . ?o wdt:P31 ?t . }'
        #query_entobj_str = 'SELECT ?r ?t WHERE { ?s ?r wd:' + ent + ' . ?s wdt:P31 ?t . }'
        query_entsubjobj_str = 'SELECT ?ro ?to ?rs ?ts WHERE { { wd:' + ent + ' ?ro ?o . ?o wdt:P31 ?to .} ' \
                                                    'UNION {?s ?rs wd:' + ent + ' . ?s wdt:P31 ?ts .}}'
        query_ent_type = 'SELECT ?t WHERE { wd:' + ent + ' wdt:P31 ?t . }'
        ## execute
        try:
            then = datetime.now()

            entityGraph[ent]['label'] = " ".join(bert_tokenizer(id_entity[ent]))

            try:
                result = cache[ent]
                entityGraph[ent][SUBJECT] = result[SUBJECT]
                entityGraph[ent][OBJECT] = result[OBJECT]
            except KeyError:
                result = SERVER.query(query_entsubjobj_str)
                entityGraph[ent][SUBJECT] = associatedRelations(result, 'rs', 'ts')
                entityGraph[ent][OBJECT] = associatedRelations(result, 'ro', 'to')

            now = datetime.now()
            duration = now - then
            if duration.total_seconds() > 1:
                cache[ent] = {SUBJECT: entityGraph[ent][SUBJECT], OBJECT: entityGraph[ent][OBJECT]}
                print(ent, 'cached!')

            result = SERVER.query(query_ent_type)
            entityGraph[ent][TYPE] = \
                list(set([(x, " ".join(GRAPH_vocab[x])) for x in
                          SparqlResults.getEntitySetFromBindings(result)['t'] if x in IDs]))

        except requests.exceptions.Timeout:
            print('FAIL on ', ent)
            continue

    return entityGraph

def graphSize(g):
    ret = 0
    for e in g.keys():
        ret += len(g[e][SUBJECT].keys()) + \
                    sum([len(g[e][SUBJECT][k]['type_restriction']) for k in g[e][SUBJECT].keys()]) + \
                    len(g[e][OBJECT].keys()) + \
                    sum([len(g[e][OBJECT][k]['type_restriction']) for k in g[e][OBJECT].keys()]) + \
                    len(g[e][TYPE])
    return  ret

def getLabel(e):
    query_str = 'SELECT ?l WHERE { wd:' + e + ' rdfs:label ?l . }'
    try:
        result = SERVER.query(query_str)
        result = SparqlResults.getEntitySetFromBindings(result)['l']
        result = result[0] if len(result) > 0 else e

    except requests.exceptions.Timeout:
        print('FAIL on ', e)
        result = e

    return result

def getLabelJson(r):
    return id_relation[r]

def buildVocab(splits):
    # Lasagne loads the graph to be used accessed by to model from the underlying knowledge data (wikidata_type_dict.json)
    # but only those elements (types and relations) that appear in conversations in train/test/valid. That is when
    # processing splits they build a 'graph vocabulary' then when loading the KG keep only those in the graph vocabulary.
    # We do the same here.
    GRAPH_vocab = {}
    for sp in splits:
        print("*\t Creating vocab for ", sp)
        # read data
        files = glob(f'{ROOT_PATH}/{sp}/*' + '/*.json')
        pbar = tqdm(total=len(files))
        for e, f in enumerate(files):
            #if e>10:
            #    break
            with open(f) as json_file:
                try:
                    # load conversation
                    conversation = json.load(json_file)
                    for turn in conversation:
                        if turn['speaker'] == 'USER':
                            if 'relations' in turn.keys():
                                for r in turn['relations']:
                                    GRAPH_vocab[r] = bert_tokenizer(getLabelJson(r).lower())
                            if 'type_list' in turn.keys():
                                for t in turn['type_list']:
                                    GRAPH_vocab[t] = bert_tokenizer(getLabel(t).lower())

                except json.decoder.JSONDecodeError:
                    continue
            if (e % 200) == 0:
                pbar.update(200)
    return GRAPH_vocab

def take_nels(nel_field):
    ret = []
    if len(nel_field) > 0:
        if isinstance(nel_field[0], list):
            ret = [x[0] for x in nel_field if len(x) > 0] # TODO: take the top one, see if we want to choose other top-k
        else:
            ret = nel_field
    return ret

splits=[args.partition]

# new splits directory
if not os.path.isdir(DST_ROOT_PATH):
    os.mkdir(DST_ROOT_PATH)
    for sp in splits:
        os.mkdir(os.path.join(DST_ROOT_PATH, sp))
    print(f'Directory "{DST_ROOT_PATH}" created')

if args.task == 'vocab':
    print('*\t Build vocab...')
    GRAPH_vocab = buildVocab(['valid', 'train', 'test'])
    with open(vocab_file, 'w') as fvoc:
        json.dump(GRAPH_vocab, fvoc, ensure_ascii=False)
    print('*\t Save vocab...', len(GRAPH_vocab))
    exit()
else:
    with open(vocab_file) as fvoc:
        GRAPH_vocab = json.load(fvoc)
        print('*\t Load vocab...', len(GRAPH_vocab))

def copy_allen_annotation(user, system, conversation_nel, i):
    if ES_LINKS in conversation_nel[2 * i].keys():
        user[ALLEN_ES_LINKS] = conversation_nel[2 * i][ES_LINKS]
        user[ALLEN_SPANS] = conversation_nel[2 * i][SPANS]
        user[ALLEN_TAGS] = conversation_nel[2 * i][ALLEN_TAGS]
    if ES_LINKS in conversation[2 * i + 1].keys():
        system[ALLEN_ES_LINKS] = conversation_nel[2 * i + 1][ES_LINKS]
        system[ALLEN_SPANS] = conversation_nel[2 * i + 1][SPANS]
        system[ALLEN_TAGS] = conversation_nel[2 * i + 1][ALLEN_TAGS]

    return user, system

def copy_str_annotation(user, system, conversation_nel, i):
    if ES_LINKS in conversation_nel[2 * i].keys():
        user[STR_ES_LINKS] = conversation_nel[2 * i][ES_LINKS]
        user[STR_SPANS] = conversation_nel[2 * i][SPANS]
    if ES_LINKS in conversation[2 * i + 1].keys():
        system[STR_ES_LINKS] = conversation_nel[2 * i + 1][ES_LINKS]
        system[STR_SPANS] = conversation_nel[2 * i + 1][SPANS]

    return user, system


log_missing = open('log_miss.txt', 'w')
IDs = GRAPH_vocab.keys()
subGraphSizes = {f'{args.partition}': []}
for sp in splits:
    # read data
    files = glob(f'{ROOT_PATH}/{sp}/*' + '/*.json')

    pbar = tqdm(total=len(files))

    for e, f in enumerate(files):

        if args.part != 'none':
            # run by sets of folders
            folder = f.rsplit('/', 1)[0].rsplit('/', 1)[-1]
            p = int(folder.split('QA_')[1][0])
            if (args.part == 'first' and p not in range(0, 2)) or \
                    (args.part == 'second' and p not in range(2, 5)) or \
                    (args.part == 'third' and p not in range(5, 7)) or \
                    (args.part == 'fourth' and p not in range(7, 10)):
                continue

        if args.allennlpNER_folder:
            f_nel_allen = f'{args.allennlpNER_folder}/{sp}{f.split(sp)[1]}.tagged'
            json_file_nel_allen = open(f_nel_allen)
        if args.strNEL_folder:
            f_nel_str  = f'{args.strNEL_folder}/{sp}{f.split(sp)[1]}.strtaggedwithoutproperty'
            json_file_nel_str = open(f_nel_str)

        with open(f) as json_file:
                try:
                    fileName = f.split('/')[-1]
                    dirName = f.split('/')[-2]
                    # load conversation
                    conversation = json.load(json_file)
                    new_conversation = []
                    conversation_nel_allen = json.load(json_file_nel_allen) if args.allennlpNER_folder else None
                    conversation_nel_str = json.load(json_file_nel_str) if args.strNEL_folder else None
                    assert len(conversation) == len(conversation_nel_allen)

                    prev_user_conv = None
                    prev_system_conv = None
                    is_clarification = False
                    is_history_ner_spurious = False
                    turns = len(conversation) // 2
                    for i in range(turns):

                        if is_clarification:
                            is_clarification = False
                            continue

                        user = conversation[2 * i]
                        system = conversation[2 * i + 1]

                        # copy nel annotations
                        if conversation_nel_allen:
                            user, system = copy_allen_annotation(user, system, conversation_nel_allen, i)
                        if conversation_nel_str:
                            user, system = copy_str_annotation(user, system, conversation_nel_str, i)

                        if user['question-type'] == 'Clarification':
                            new_conversation.append(user)
                            new_conversation.append(system)

                            # get next context
                            is_clarification = True
                            next_user = conversation[2 * (i + 1)]
                            next_system = conversation[2 * (i + 1) + 1]

                            # copy nel annotations
                            if conversation_nel_allen:
                                next_user, next_system = copy_allen_annotation(next_user, next_system,
                                                                           conversation_nel_allen, i + 1)
                            if conversation_nel_str:
                                next_user, next_system = copy_str_annotation(next_user, next_system,
                                                                           conversation_nel_str, i + 1)

                            # collect entities, here we are taking gold annotations
                            utteranceEntities = []
                            utteranceLinkedEntities = []
                            if i > 0:
                                # context Gold
                                if 'entities_in_utterance' in prev_user_conv.keys() \
                                        and 'entities_in_utterance' in prev_system_conv:
                                    utteranceEntities.extend(prev_user_conv['entities_in_utterance'])
                                    utteranceEntities.extend(prev_system_conv['entities_in_utterance'])
                                else:
                                    if 'entities_in_utterance' in prev_user_conv.keys():
                                        utteranceEntities.extend(prev_user_conv['entities_in_utterance'])
                                    # when previous is Clarification the name of the field is different!!!!
                                    elif 'entities' in prev_user_conv.keys():
                                        utteranceEntities.extend(prev_user_conv['entities'])
                                    if 'entities_in_utterance' in prev_system_conv.keys():
                                        utteranceEntities.extend(prev_system_conv['entities_in_utterance'])

                                    # debug
                                    print('what is missing? C', 'entities_in_utterance' in prev_user_conv.keys(),
                                          'entities_in_utterance' in prev_system_conv,
                                          'entities_in_utterance' in prev_system_conv.keys(),
                                          '\n', f, '\n', user['utterance'])

                                # context nel
                                if ALLEN_ES_LINKS in prev_user_conv.keys():
                                    utteranceLinkedEntities.extend(take_nels(prev_user_conv[ALLEN_ES_LINKS]))
                                if ALLEN_ES_LINKS in prev_system_conv.keys():
                                    utteranceLinkedEntities.extend(take_nels(prev_system_conv[ALLEN_ES_LINKS]))
                                if STR_ES_LINKS in prev_user_conv.keys():
                                    utteranceLinkedEntities.extend(take_nels(prev_user_conv[STR_ES_LINKS]))
                                if STR_ES_LINKS in prev_system_conv.keys():
                                    utteranceLinkedEntities.extend(take_nels(prev_system_conv[STR_ES_LINKS]))

                            # user context
                            if 'entities_in_utterance' in user.keys():
                                utteranceEntities.extend(user['entities_in_utterance'])
                            if ALLEN_ES_LINKS in user.keys():
                                utteranceLinkedEntities.extend(take_nels(user[ALLEN_ES_LINKS]))
                            if STR_ES_LINKS in user.keys():
                                utteranceLinkedEntities.extend(take_nels(user[STR_ES_LINKS]))

                            # system context
                            if 'entities_in_utterance' in system.keys():
                                utteranceEntities.extend(system['entities_in_utterance'])
                            if ALLEN_ES_LINKS in  system.keys():
                                utteranceLinkedEntities.extend(take_nels(system[ALLEN_ES_LINKS]))
                            if STR_ES_LINKS in  system.keys():
                                utteranceLinkedEntities.extend(take_nels(system[STR_ES_LINKS]))

                            # next user context
                            if 'entities_in_utterance' in next_user.keys():
                                utteranceEntities.extend(next_user['entities_in_utterance'])
                            if ALLEN_ES_LINKS in  next_user.keys():
                                utteranceLinkedEntities.extend(take_nels(next_user[ALLEN_ES_LINKS]))
                            if STR_ES_LINKS in  next_user.keys():
                                utteranceLinkedEntities.extend(take_nels(next_user[STR_ES_LINKS]))

                            # EXTRACT KB SUBGRAPH
                            if hasattr(args, 'nel_entities'):
                                # add additional sub_graph for linked entities, if *local_subgraph* exists add extras from
                                # NEL only.
                                next_system['local_subgraph_nel'] = getSubgraph(set(utteranceLinkedEntities),
                                                                       next_system['local_subgraph'] if  'local_subgraph'
                                                                        in next_system.keys() else None)
                            else:
                                next_system['local_subgraph'] = getSubgraph(set(utteranceEntities))

                            if 'local_subgraph' in next_system.keys():
                                subGraphSizes[sp].append(graphSize(next_system['local_subgraph']))

                            # track context history
                            prev_user_conv = next_user.copy()
                            prev_system_conv = next_system.copy()

                            new_conversation.append(next_user)
                            new_conversation.append(next_system)
                        else:

                            # collect entities, here we are taking gold annotations
                            utteranceEntities = []
                            utteranceLinkedEntities = []
                            if i > 0:
                                # context
                                if 'entities_in_utterance' in prev_user_conv.keys()\
                                        and 'entities_in_utterance' in prev_system_conv:
                                    utteranceEntities.extend(prev_user_conv['entities_in_utterance'])
                                    utteranceEntities.extend(prev_system_conv['entities_in_utterance'])
                                else:
                                    if 'entities_in_utterance' in prev_user_conv.keys():
                                        utteranceEntities.extend(prev_user_conv['entities_in_utterance'])
                                    # when previous is Clarification the name of the field is different!!!!
                                    elif 'entities' in prev_user_conv.keys():
                                        utteranceEntities.extend(prev_user_conv['entities'])
                                    if 'entities_in_utterance' in prev_system_conv.keys():
                                        utteranceEntities.extend(prev_system_conv['entities_in_utterance'])

                                    print('WHAT gold ents?', 'entities_in_utterance' in prev_user_conv.keys(),
                                                    'entities_in_utterance' in prev_system_conv,
                                                    'entities_in_utterance' in prev_system_conv.keys(),
                                          '\n', f, '\n', user['utterance'])

                                # context nel
                                if ALLEN_ES_LINKS in prev_user_conv.keys():
                                    utteranceLinkedEntities.extend(take_nels(prev_user_conv[ALLEN_ES_LINKS]))
                                if STR_ES_LINKS in prev_user_conv.keys():
                                    utteranceLinkedEntities.extend(take_nels(prev_user_conv[STR_ES_LINKS]))
                                if ALLEN_ES_LINKS in prev_system_conv.keys():
                                    utteranceLinkedEntities.extend(take_nels(prev_system_conv[ALLEN_ES_LINKS]))
                                if STR_ES_LINKS in prev_system_conv.keys():
                                    utteranceLinkedEntities.extend(take_nels(prev_system_conv[STR_ES_LINKS]))

                            # user context
                            if 'entities_in_utterance' in user.keys():
                                utteranceEntities.extend(user['entities_in_utterance'])
                            if ALLEN_ES_LINKS in user.keys():
                                utteranceLinkedEntities.extend(take_nels(user[ALLEN_ES_LINKS]))
                            if STR_ES_LINKS in user.keys():
                                utteranceLinkedEntities.extend(take_nels(user[STR_ES_LINKS]))

                            # EXTRACT KB SUBGRAPH
                            if hasattr(args, 'nel_entities') and args.nel_entities:
                                # add additional sub_graph for linked entities, if *local_subgraph* exists add extras from
                                # NEL only.
                                system['local_subgraph_nel'] = getSubgraph(set(utteranceLinkedEntities),
                                                                       system['local_subgraph'] if  'local_subgraph'
                                                                        in system.keys() else None)
                            else:
                                system['local_subgraph'] = getSubgraph(set(utteranceEntities))

                            if 'local_subgraph' in system.keys():
                                subGraphSizes[sp].append(graphSize(system['local_subgraph']))

                            # track context history
                            prev_user_conv = user.copy()
                            prev_system_conv = system.copy()

                            new_conversation.append(user)
                            new_conversation.append(system)

                    # write conversation
                    assert len(conversation) == len(new_conversation)

                    if not os.path.isdir(os.path.join(DST_ROOT_PATH, sp, dirName)):
                        os.mkdir(os.path.join(DST_ROOT_PATH, sp, dirName))
                    with open(f'{DST_ROOT_PATH}/{sp}/{dirName}/{fileName}', 'w') as formatted_json_file:
                        json.dump(new_conversation, formatted_json_file, ensure_ascii=False, indent=4)

                except json.decoder.JSONDecodeError:
                    print('Fail',f)
                    continue
        if (e % 50) == 0:
            pbar.update(50)

    sp_graph_lens = np.array(subGraphSizes[sp])
    print("Finished formatting", sp, np.mean(sp_graph_lens), np.min(sp_graph_lens), np.max(sp_graph_lens))
