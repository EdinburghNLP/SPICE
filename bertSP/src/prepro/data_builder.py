import gc
import json
from glob import glob
from collections import Counter
from tqdm import tqdm
import time
from multiprocess import Pool
import itertools
from os.path import join as pjoin

from transformers import BertTokenizer
from torchtext.data import Field, Example, Dataset, NestedField, RawField
from torchtext.vocab import Vocab

# import constants
from others.constants import *
from others.logging import logger

class SPICEDataset:
    def __init__(self, args):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        gobalVocF = open(args.tgt_dict)
        d = {}
        for l in gobalVocF.readlines():
            d[l.strip()] = 1
        self.gobal_syntax_vocabulary = Vocab(Counter(d), specials=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, 'num', '0']) #TODO: add 0 to voc construct
        print("*\t Global syntax constants vocabulary size:", len(self.gobal_syntax_vocabulary))

        self.nbunks = 0

        if hasattr(args, 'types'):
            self.types = args.types
        if hasattr(args, 'nentities'):
            self.nentities = args.nentities

    def load_type_triples(self, args):
        self.type_triples = json.loads(open(args.kb_graph + '/wikidata_type_dict.json').read())
        self.rev_type_triples = json.loads(open(args.kb_graph + '/wikidata_rev_type_dict.json').read())
        self.id_relation = json.loads(open(args.kb_graph +  '/filtered_property_wikidata4.json').read())
        self.id_entity = json.loads(open(args.kb_graph + '/items_wikidata_n.json').read())

    def bert_binarise(self, subtokens):
        return self.bert_tokenizer.convert_tokens_to_ids(subtokens)

    def lineariseGraph(self, graph, nel, f, utt, types, subgraph_nel):

        def getTypesGraph(gold_types):
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

        def extendIdsMap(types_graph, ids_map):

            for t_head in types_graph.keys():
                if t_head not in ids_map.keys():
                    ids_map[t_head] = " ".join(self.bert_tokenizer.tokenize(ID_ENTITY[t_head]))

                for r in types_graph[t_head]:
                    if r not  in ids_map.keys():
                        ids_map[r] = " ".join(self.bert_tokenizer.tokenize(ID_RELATION[r]))

            return ids_map

        def groupTypes(relSubset, direction, ids_map):
            # entity is object
            typeRestrictions = [] # group type restrictions for each relation/property
            for k in relSubset.keys():
                tmp = []
                for i, l in relSubset[k]['type_restriction']:
                    tmp.append(i)

                if tmp:
                    if len(tmp) > MAX_TYPE_RESTRICTIONS: # chunk as could be a quite big set in some cases, take first randomly
                        tmp = tmp[:MAX_TYPE_RESTRICTIONS]
                    typeRestrictions.append(k + f' {RELARGS_DELIMITER} ' + \
                                                             f' {RELARGS_DELIMITER} '.join(tmp) \
                                        if direction == SUBJECT else \
                                            f' {RELARGS_DELIMITER} '.join(tmp) + \
                                             f' {RELARGS_DELIMITER} ' + k)
                else:
                    # no type restriction for this relation
                    # never have this case!!!!!!
                    typeRestrictions.append(k)

            return typeRestrictions

        def getGraphVocab(graph):
            ids_map = {}
            for ent in graph.keys():
                entLabel = graph[ent]['label']
                ids_map[ent] =  entLabel

                #types
                for i, l in graph[ent][TYPE]:
                    ids_map[i] = l

                #relations' type restrictions
                for direction in [SUBJECT, OBJECT]:
                    for k in graph[ent][direction].keys():
                        for i, l in graph[ent][direction][k]['type_restriction']:
                            ids_map[i] = l

            # relations
            for ent in graph.keys():
                for direction in [SUBJECT, OBJECT]:
                    for k in graph[ent][direction].keys():
                        l = graph[ent][direction][k]['label']
                        ids_map[k] = l

            return ids_map

        # accommodate the entity graph to be used according whether we use gold/lasagne-gold-nel/external-nel
        if self.nentities in [NELGNEL, NEALLENNEL, NESTRNEL]:
            d = {}
            for ent in graph.keys():
                if ent not in nel:
                    continue
                else:
                    d[ent] = graph[ent]
            graph = d
        if self.nentities in [NEALLENNEL, NESTRNEL]:
            for ent in nel:
                # 'subgraph_nel' contains additional entities introduced by external off-the-shelf NEL methods
                # the 'nel' set contains only the entities for the chosen nel method (allenNLP or Str)
                # whereas the subgraph_nel collects all nel entities from all methods
                if ent in subgraph_nel.keys(): # could already present in gold too
                    graph[ent] = subgraph_nel[ent]

        ids_map = getGraphVocab(graph)

        types_graph = None
        if types:
            if self.types == TGOLD:
                types_graph = getTypesGraph(types)
            else: # types are linked
                types_graph = types
                # graph from type linking, we know in average there are 2.3 gold types
                if len(types.keys()) > MAX_LINKED_TYPES:
                    types_graph = {}
                    for k in list(types.keys())[:MAX_LINKED_TYPES]:
                        types_graph[k] = types[k]

            ids_map = extendIdsMap(types_graph, ids_map)

        entitySequences = {}

        # if adding types (gold or linked)
        if types_graph:
            for type in types_graph.keys():

                seqElements = [type]

                if len(types_graph[type]) > 0:
                    seqElements.extend(types_graph[type])

                entitySequences[type] = seqElements

        # add lines from entity graph
        for ent in graph.keys():
            seqElements = [ent]

            # add types
            if len(graph[ent][TYPE]) > 0:
                seqElements.extend([i for i, l in graph[ent][TYPE]])

            # entity is subject or object
            for direction in [SUBJECT, OBJECT]:
                if len(graph[ent][direction].keys()) > 0:
                    seqElements.extend(groupTypes(graph[ent][direction], direction, ids_map))

            entitySequences[ent] = seqElements

        return entitySequences, ids_map

    def vocInputchunk(self):
        str_syn_voc = f' {TMP_DELIMITER} ' .join([' '.join(self.bert_tokenizer.tokenize(k.lower()))
                                                  for k,v in self.gobal_syntax_vocabulary.stoi.items()
                                                     if not (k == BOS_TOKEN or k == PAD_TOKEN or k == EOS_TOKEN)])

        # they where the first 3 elements in the voc, so should work, they still be aligned with ids 0, 1, and 2
        str_syn_voc = f' {TMP_DELIMITER} '.join([PAD_TOKEN, BOS_TOKEN_BERT, EOS_TOKEN_BERT]) \
                      + f' {TMP_DELIMITER} ' + str_syn_voc
        vocIds = [v for k,v in self.gobal_syntax_vocabulary.stoi.items()]

        end = []
        start = []
        start.append(0)
        for i, token in enumerate(str_syn_voc.split()):
            if (str(token) == TMP_DELIMITER):
                end.append(i - 1)
                start.append(i + 1)

        if len(start)==1 and len(str_syn_voc.split()) == start[0]: # case where there is no KB elements
            start = []
        else:
            end.append(len(str_syn_voc.split())-1) #-1 for las [SEP] token

        str_syn_voc = self.bert_binarise(str_syn_voc.split(' '))

        return str_syn_voc, vocIds, start, end

    def generateInputChunks(self, context, subgraph, nel, f, utt, types, subgraph_nel):
        """Create input chunks combine conversation context with KG sub-graph (context related neighbourhood)"""
        # TODO: Each chunk goes for each anchor entity, then truncate it at 512 if needed. Improve on this

        entity_sequences, entity_map = self.lineariseGraph(subgraph, nel, f, utt, types, subgraph_nel)

        #derive instance dynamic dict
        dyDict = list(entity_map.keys())

        textual = f'{CLS_TOKEN} ' + ' '.join(context).lower()
        textualLen = len(textual.split()) + 1

        # seems some turns (sp. ellided) may have zero entities
        if len(entity_sequences.keys()) == 0:
            return [textual + ' ' + SEP_TOKEN], [[0]*textualLen], [[-1]], [[-1]], [[-1]], entity_map

        textKG = [' '.join(([textual] + \
                            [SEP_TOKEN] + \
                            (f' {TMP_DELIMITER} '.join([ f' {RELARGS_DELIMITER} '.join([entity_map[e.strip()]
                                                                                    for e in reltype.split(RELARGS_DELIMITER)])
                                                    for reltype in entity_sequences[ent]
                                                    ])).lower().split() + \
                            [SEP_TOKEN]))  \
                  for ent in entity_sequences.keys()]

        # if instead of chunk to MAX_INPUTSEQ_LEN add another entity chunk for excedent, then no need of this... TODO
        tmp = []
        for s in textKG:
            s = s.split(' ')[:MAX_INPUTSEQ_LEN]
            if str(s[-1]) == TMP_DELIMITER or str(s[-1]) == RELARGS_DELIMITER:
                s = s[:MAX_INPUTSEQ_LEN - 1]
            if str(s[-1]) != SEP_TOKEN:
                s.append(SEP_TOKEN)
            tmp.append(' '.join(s))
        textKG = tmp

        textKG = textKG[:10] # take maxi 10 chunks for each entity

        # there are few questions where the question itself is longer than BERT limit 512.
        # textKG elems, basically only the question, will be truncated above. Then truncate the
        # the segments to have the truncated question len. Ex of this case is 'valid#QA_773#QA_125#3'
        if textualLen > len(textKG[0].split(' ')):
            textualLen = len(textKG[0].split(' '))

        # context part has segment A, KB part will have segment B
        textKGSegments = [[0]*textualLen + [1]*(len(x.split(' '))-textualLen) for x in textKG]
        # TODO: in  a couple of cases textKGSegments has different sizes that textKG, how can? will need to debug here

        assert len(textKG) == len(textKGSegments), \
            f'Fail to match sizes of textKG={len(textKG)} and textKGSegments={len(textKGSegments)}'
        for i in range(len(textKG)):
            if not (len(textKG[i].split(' ')) == len(textKGSegments[i])):
                print(f'src={len(textKG[i].split(" "))} segs={len(textKGSegments[i])}')

        # register positions for begin-end token positions of KB elements
        starts = []
        ends = []
        chunked_entity_sequences = {}
        for tokenized_source, ent in zip(textKG, entity_sequences.keys()):
            end = []
            start = []
            sepTokenCount = 0
            flag = False
            idSeqs = []
            for e in entity_sequences[ent]:
                idSeqs.extend(e.split(f' {RELARGS_DELIMITER} '))
            chunked_entity_sequences[ent] = []
            idSeqsCnt = 0
            for i, token in enumerate(tokenized_source.split()):
                if flag and (str(token) == TMP_DELIMITER or str(token) == RELARGS_DELIMITER):
                    end.append(i - 1)
                    start.append(i + 1)
                    chunked_entity_sequences[ent].append(idSeqs[idSeqsCnt])
                    idSeqsCnt +=1
                if str(token) == SEP_TOKEN:
                    sepTokenCount +=1
                    if sepTokenCount == 1:
                        #reached the first [SEP] so KG part starts, the second & last [SEP] is at the end of the sequence
                        if not flag:
                            start.append(i + 1)
                        flag = True
            if len(start) == 1 and len(tokenized_source.split()) == start[0]: # case where there is no KB elements
                start = []
            else:
                end.append(len(tokenized_source.split())-1) #-1 for las [SEP] token
                chunked_entity_sequences[ent].append(idSeqs[idSeqsCnt])
            if len(start)==0:
                print('textKG, building chunks, start is 0', textKG)

            if len(start)>0 and len(end)>0:
                assert start[-1] <= end[-1], f'tokenized_source: {tokenized_source} \n{start} \n{end}'

            starts.append(start)
            ends.append(end)

        # register vocabulary id for elements
        maxIDsAllowed = len(self.gobal_syntax_vocabulary) + len(dyDict)
        kgVocIds = []
        for ent in chunked_entity_sequences.keys():
            entVocIds = []
            for item in chunked_entity_sequences[ent]:
                for kgel in item.split(f' {RELARGS_DELIMITER} '):
                    entVocIds.append(dyDict.index(kgel) + len(self.gobal_syntax_vocabulary))
                    assert dyDict.index(kgel) + len(self.gobal_syntax_vocabulary) < maxIDsAllowed
            kgVocIds.append(entVocIds)

        textKG = [s.replace(RELARGS_DELIMITER, RELARGS_DELIMITER_BERT)
                      .replace(TMP_DELIMITER, KGELEM_DELIMITER_BERT) for s in textKG]

        assert  len(starts) == len(ends) == len(kgVocIds)

        return textKG, textKGSegments, starts, ends, kgVocIds, entity_map

    def format_delex(self, turn_sp):
        # reformat as Lasagna expects the logical forms (lists of lists with token types)
        lasagneFormat = []
        sparqToks = turn_sp['sparql'].replace('.', ' . ').replace('wd:', 'wd: ') \
            .replace('wdt:', 'wdt: ').replace('{', ' { ').replace('}', ' } ') \
            .replace('(', ' ( ').replace(')', ' ) ') \
            .split()
        sparqdlToks = turn_sp['sparql_delex'].replace('.', ' . ') \
            .replace('{', ' { ').replace('}', ' } ') \
            .replace('(', ' ( ').replace(')', ' ) ') \
            .split()
        assert len(sparqToks) == len(sparqdlToks), f
        'Different len in tokens sparqToks={len(sparqToks)} vs sparqdlToks={len(sparqdlToks)}'
        for sparq, sparqdl in zip(sparqToks, sparqdlToks):
            if sparqdl == ENTITY or sparqdl == RELATION or sparqdl == TYPE or sparqdl == VALUE:
                lasagneFormat.append([sparqdl, sparq])
            else:
                lasagneFormat.append([ACTION, sparqdl])

        return lasagneFormat

    def spurious_interaction(self, user, system):
        if self.nentities == NELGNEL or self.nentities == NEGOLD: #keep gold also as we used in trained gold
            # Use lasagne ner/nel gold based annotations
            return 'is_ner_spurious' not in user.keys() or user['is_ner_spurious'] or 'is_ner_spurious' not in system.keys() or system['is_ner_spurious']
        else:
            return not (('type_subgraph' in system.keys() and system['type_subgraph'].keys()) \
                    or (self.take_nels(user['es_links']))) # there are linked types or linked entities
                    # type_subgraph corresponds to those types in the user question that are linked.
                    # for entities linked we check in user question, as entities in system utterances are known.

    def spurious_interaction_clarification(self, user, system, next_user, next_system):
        if self.nentities == NELGNEL or self.nentities == NEGOLD: #keep gold also as we used in trained gold
            # Use lasagne ner/nel gold based annotations
            return 'is_ner_spurious' not in user.keys() or 'is_ner_spurious' not in system.keys() or \
                'is_ner_spurious' not in next_user.keys() or 'is_ner_spurious' not in next_system.keys() or \
                user['is_ner_spurious'] or system['is_ner_spurious'] or next_user['is_ner_spurious'] or next_system['is_ner_spurious']
        else:
            ents = self.take_nels(user['es_links']) + self.take_nels(system['es_links']) + self.take_nels(next_user['es_links'])
            return not (('type_subgraph' in next_system.keys() and next_system['type_subgraph'].keys()) \
                    or ents)

    def take_nels(self, nel_field):
        ret = []
        if len(nel_field) > 0:
            if isinstance(nel_field[0], list):
                ret = [x[0] for x in nel_field if
                       len(x) > 0]  # TODO: take the top one, see if we want to choose other top-k
            else:
                ret = nel_field
        return ret

    def addContext(self, turn, do_num=False):
        '''Adds context: textual and Named Entity annotations. NER annotations could
        be Lasagne ones or external NER tool (es_links).'''
        ret_input, ret_nel = [], []
        if self.nentities == NELGNEL or self.nentities == NEGOLD: #keep gold also as we used in trained gold

            if 'context' in turn:
                for context in turn['context']:
                    if do_num:
                        ret_input.append(' '.join(self.bert_tokenizer.tokenize('num')) if context[1] == 'num' else context[1])
                    else:
                        ret_input.append(context[1])
                    if (context[-1] in [B]):  # only enter when B otherwise it repeats
                        ret_nel.append(context[-3])
            else:
                ret_input = self.bert_tokenizer.tokenize(turn['utterance'])
            if turn['speaker'] == SYSTEM:
                 ret_nel = turn['entities_in_utterance']
            if self.nentities == NEGOLD:
                ret_nel = [] # if we are in gold setting this will not be necessary.
        else:
            if 'context' in turn:
                for context in turn['context']:
                    if do_num:
                        ret_input.append(' '.join(self.bert_tokenizer.tokenize('num')) if context[1] == 'num' else context[1])
                    else:
                        ret_input.append(context[1])
            else:
                ret_input = self.bert_tokenizer.tokenize(turn['utterance'])
            if turn['speaker'] == USER:
                if self.nentities == NEALLENNEL: # AllenNLP -based NEL annotations
                    ret_nel = self.take_nels(turn[ALLEN_ES_LINKS])
                elif self.nentities == NESTRNEL: # Str -based NEL annotations
                    ret_nel = self.take_nels(turn[STR_ES_LINKS])
            else: # is SYSTEM
                ret_nel = turn['entities_in_utterance'] # as previous works take previous gold answer (systems output answers so thye know the entity they output -> no need to do NEL)

        return ret_input, ret_nel

    def _prepare_data(self, data, infiles=False, isTest=False):
        '''
        When infiles = True, we are actually passing the json file paths, this is more
        memory efficient rather the loading whole data into memory and then process it.
        '''
        input_data = []
        f = None
        fID = None
        for conversation in data:
            if infiles:
                try:
                    f = conversation
                    conversation = json.load(open(conversation, 'r'))
                    fID = f.split('/')
                    fID = f'{fID[-3]}#{fID[-2]}#{fID[-1].split(".json")[0]}'
                    print(f)

                except json.decoder.JSONDecodeError:
                    print(f'Failed to load conversation: {conversation}')
                    continue

            prev_user_conv = None
            prev_system_conv = None
            is_clarification = False
            turns = len(conversation) // 2
            for i in range(turns):
                input = []
                str_logical_form = []
                logical_form = [self.gobal_syntax_vocabulary.stoi[BOS_TOKEN]]
                nel = []
                context_types = [] if self.types == TGOLD else ({} if self.types == TLINKED else None)

                if is_clarification:
                    is_clarification = False
                    continue

                user = conversation[2*i]
                system = conversation[2*i + 1]

                if user['question-type'] == 'Clarification':
                    # get next context
                    is_clarification = True
                    next_user = conversation[2*(i+1)]
                    next_system = conversation[2*(i+1) + 1]

                    # skip if no gold action (or spurious)
                    if GOLD_ACTIONS not in next_system.keys(): # or next_system['sparql_spurious']: ==> remove this condition, we have already cleaned the dataset
                        prev_user_conv = next_user.copy()
                        prev_system_conv = next_system.copy()
                        continue

                    if i == 0:
                        input.extend([NA_TOKEN_BERT, CTX_TOKEN_BERT, NA_TOKEN_BERT, CTX_TOKEN_BERT])
                    else:
                        # add prev context user
                        ret_in, ret_nel = self.addContext(prev_user_conv)
                        input.extend(ret_in)
                        nel.extend(ret_nel)

                        # sep token
                        input.append(CTX_TOKEN_BERT)

                        # add prev context answer
                        ret_in, ret_nel = self.addContext(prev_system_conv, do_num=True)
                        input.extend(ret_in)
                        nel.extend(ret_nel)

                        # sep token
                        input.append(CTX_TOKEN_BERT)

                        ## types from context
                        if self.types == TGOLD:
                            context_types = prev_user_conv['type_list'] if 'type_list' in prev_user_conv.keys() else []
                        elif self.types == TLINKED and 'type_subgraph' in prev_system_conv.keys():
                            # these are automatically linked types
                            context_types = prev_system_conv['type_subgraph']

                    # user context
                    ret_in, ret_nel = self.addContext(user)
                    input.extend(ret_in)
                    nel.extend(ret_nel)

                    # system context
                    ret_in, ret_nel = self.addContext(system)
                    input.extend(ret_in)
                    nel.extend(ret_nel)

                    # next user context
                    ret_in, ret_nel = self.addContext(next_user)
                    input.extend(ret_in)
                    nel.extend(ret_nel)

                    # resolution sub-graph
                    ## types
                    types = None
                    existing_types = True # if we decide to not include types, then are fine
                    if self.types == TGOLD:
                        types = user['type_list'] if 'type_list' in user.keys() else []
                        types = list(set(context_types + types))
                        existing_types = len(types) > 0
                    elif self.types == TLINKED and 'type_subgraph' in next_system.keys():
                        # these are automatically linked types
                        types = next_system['type_subgraph']
                        for k, v in context_types.items(): types[k] = v
                        existing_types = len(types.keys()) > 0

                    ## entities
                    existing_entities = (self.nentities in [NELGNEL, NEALLENNEL, NESTRNEL] and nel) or \
                                        (self.nentities == NEGOLD and 'local_subgraph' in next_system.keys() \
                                                        and len(next_system['local_subgraph'].keys())>0 )

                    # check if there are entities and types in the input for c_t + x_t (spurious)
                    if not isTest and not (existing_types or existing_entities):
                        prev_user_conv = next_user.copy()
                        prev_system_conv = next_system.copy()
                        print('Spurious CLARI', next_user['utterance'])
                        continue

                    # prepare the input
                    if 'local_subgraph' in next_system.keys():
                        input, inputChunks, starts, ends, kgVocIds, entity_map = \
                            self.generateInputChunks( \
                                input, next_system['local_subgraph'], nel, f, user['utterance'], \
                                types, next_system['local_subgraph_nel'] if 'local_subgraph_nel' in next_system.keys() \
                                            else None)

                    # get gold actions
                    gold_actions = self.format_delex(next_system)

                    question_type = [user['question-type'], next_user['question-type']] \
                                    if 'question-type' in next_user else user['question-type']
                    description = user[DESCRIPTION]

                    # redefine gold answer
                    results = next_system['sparql_entities'] if 'sparql_entities' in next_system.keys() \
                                else next_system['all_entities']
                    answer = next_system['utterance']

                    # track context history
                    prev_user_conv = next_user.copy()
                    prev_system_conv = next_system.copy()

                else:
                    # skip if logical form is spurious ==> normally does not apply, we have already cleaned the dataset
                    if GOLD_ACTIONS not in system.keys():
                        prev_user_conv = user.copy()
                        prev_system_conv = system.copy()
                        continue

                    if i == 0:
                        input.extend([NA_TOKEN_BERT, CTX_TOKEN_BERT, NA_TOKEN_BERT, CTX_TOKEN_BERT])
                    else:
                        # add prev context user
                        ret_in, ret_nel = self.addContext(prev_user_conv)
                        input.extend(ret_in)
                        nel.extend(ret_nel)

                        # sep token
                        input.append(CTX_TOKEN_BERT)

                        # add prev context answer
                        ret_in, ret_nel = self.addContext(prev_system_conv, do_num=True)
                        input.extend(ret_in)
                        nel.extend(ret_nel)

                        # sep token
                        input.append(CTX_TOKEN_BERT)

                        ## types from context
                        if self.types == TGOLD:
                            context_types = prev_user_conv['type_list'] if 'type_list' in prev_user_conv.keys() else []
                        elif self.types == TLINKED and 'type_subgraph' in prev_system_conv.keys():
                            # these are automatically linked types
                            context_types = prev_system_conv['type_subgraph']

                    # user context
                    ret_in, ret_nel = self.addContext(user)
                    input.extend(ret_in)
                    nel.extend(ret_nel)

                    # resolution sub-graph
                    ## types
                    types = None
                    existing_types = True # if we decide to not include types, then are fine
                    if self.types == TGOLD:
                        types = user['type_list'] if 'type_list' in user.keys() else []
                        types = list(set(context_types + types))
                        existing_types = len(types) > 0
                    elif self.types == TLINKED and 'type_subgraph' in system.keys():
                        # these are automatically linked types
                        types = system['type_subgraph']
                        for k,v in context_types.items(): types[k] = v
                        existing_types = len(types.keys()) > 0

                    ## entities
                    existing_entities = ((self.nentities == NELGNEL or self.nentities == NEL) and nel) or \
                                        (self.nentities == NEGOLD and 'local_subgraph' in system.keys() \
                                                        and len(system['local_subgraph'].keys())>0 )

                    # check if there are entities and types in the input for c_t + x_t (spurious)
                    if not isTest and not (existing_types or existing_entities):
                        prev_user_conv = user.copy()
                        prev_system_conv = system.copy()
                        print('Spurious', user['utterance'])
                        continue

                    # prepare the input
                    if 'local_subgraph' in system.keys():
                        input, inputChunks, starts, ends, kgVocIds, entity_map = \
                            self.generateInputChunks( \
                                input, system['local_subgraph'], nel, f, user['utterance'], types, \
                                system['local_subgraph_nel'] if 'local_subgraph_nel' in system.keys() else None)
                    # TODO: debug, are cases where there is no 'local_subgraph' field ?

                    # get gold actions
                    gold_actions = self.format_delex(system)

                    question_type = user['question-type']
                    description = user[DESCRIPTION]
                    # redefine gold answer
                    answer = system['utterance']
                    results = system['sparql_entities'] if 'sparql_entities' in system.keys() else system['all_entities']
                    if question_type in ['Quantitative Reasoning (Count) (All)',
                                              'Comparative Reasoning (Count) (All)']:
                        if 'sparql_entities' in system.keys() :
                            answer = f'{len(system["sparql_entities"])}'
                        elif len(set(system["all_entities"])) > 0:
                            # redefine answer value as some gold sets have duplicates
                            # Lasagne annotation tool always takes set(gold) in all comparisons to assess formula adequacy
                            answer = f'{len(set(system["all_entities"]))}'

                    # track context history
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()

                # prepare logical form
                kbVoc = list(entity_map.keys())
                # logical forms can contain KG elements that are not in the input/graph.
                # UNK for now (to continue with simple.)
                globalSyntaxVocLen = len(self.gobal_syntax_vocabulary)
                unks = set()
                for action in gold_actions:
                    if action[0] == ACTION:
                        str_logical_form.append(action[1])
                        logical_form.append(self.gobal_syntax_vocabulary.stoi[action[1]])

                    elif action[0] == RELATION:
                        str_logical_form.append(action[1])
                        logical_form.append(kbVoc.index(action[1]) + globalSyntaxVocLen if action[1] in kbVoc
                                                            else self.gobal_syntax_vocabulary.stoi[UNK_TOKEN])
                        if action[1] not in kbVoc:
                            unks.add(action[1])
                    elif action[0] == TYPE:
                        str_logical_form.append(action[1])
                        logical_form.append(kbVoc.index(action[1]) + globalSyntaxVocLen if action[1] in kbVoc
                                                else self.gobal_syntax_vocabulary.stoi[UNK_TOKEN])
                        if action[1] not in kbVoc:
                            unks.add(action[1])
                    elif action[0] == ENTITY:
                        str_logical_form.append(action[1])
                        logical_form.append(kbVoc.index(action[1]) + globalSyntaxVocLen if action[1] in kbVoc
                                                else self.gobal_syntax_vocabulary.stoi[UNK_TOKEN])
                        if action[1] not in kbVoc:
                            unks.add(action[1])
                    elif action[0] == VALUE:
                        str_logical_form.append(action[0])
                        logical_form.append(self.gobal_syntax_vocabulary.stoi['num'])
                    else:
                        raise Exception(f'Unkown logical form action {action[0]} \n {prev_user_conv} \n {prev_system_conv}')


                logical_form.append(self.gobal_syntax_vocabulary.stoi[EOS_TOKEN])
                if isTest:
                    input_data.append([input, inputChunks, starts, ends, kgVocIds, entity_map,
                                       str_logical_form, logical_form, question_type, description,
                                       user['utterance'], results, answer,
                                       f'{fID}#{i}' if fID else '' ])

                else:
                    input_data.append([input, inputChunks, starts, ends, kgVocIds, entity_map,
                                   str_logical_form, logical_form, question_type, description])


                if len(unks) > 0:
                    self.nbunks += len(unks)
                    logger.info(f'\n{f} \nunkset: {unks} \nlf: {str_logical_form} ' \
                                f'\ndescription: {description} \nquestion: {user["utterance"]}')

        return input_data

    def get_tgt_vocab(self):
        return {GLOBAL_SYNTAX: self.gobal_syntax_vocabulary}

def get_corpora(args):
    def read_split_map(mapfile, data_path, splits):
        ret = {}
        for x in splits:
            ret[x] = []
        mapf = open(mapfile)
        for l in mapf.readlines():
            f, sp = l.strip().split('\t')
            if sp in splits:
                ret[sp].append(f'{data_path}/{f}')
        return ret

    if args.mapsplits:
        print(f'Using mapping file: {args.mapfile}')

    if (args.dataset != ''):
        if args.mapsplits:
            corpora = read_split_map(args.mapfile, args.data_path, [args.dataset])
            print(f'{args.dataset} files ', args.data_path + f'/{args.dataset}/', len(corpora[args.dataset]))
        else:
            split_path = args.data_path + f'/{args.dataset}/*'
            split_files = glob(split_path + '/*.json')
            print('Files ', split_path, len(split_files))
            corpora = {f'{args.dataset}': split_files}
    else:
        # do all
        if args.mapsplits:
            corpora = read_split_map(args.mapfile, args.data_path, ['train', 'valid', 'test'])
        else:
            train_path = args.data_path + '/train/*'
            val_path = args.data_path + '/valid/*'
            test_path = args.data_path + '/test/*'
            train_files = glob(train_path + '/*.json')
            valid_files = glob(val_path + '/*.json')
            test_files = glob(test_path + '/*.json')
            corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
        for x in ['train', 'valid', 'test']:
            print(f'{x} files ', f'{args.data_path}/{x}', len(corpora[x]))

    return corpora

def format_to_lines(args):
    global TYPE_TRIPLES, REV_TYPE_TRIPLES, ID_RELATION, ID_ENTITY
    if args.types == TGOLD:
        TYPE_TRIPLES = json.loads(open(args.kb_graph +  '/wikidata_type_dict.json').read())
        REV_TYPE_TRIPLES = json.loads(open(args.kb_graph +  '/wikidata_rev_type_dict.json').read())
        ID_RELATION = json.loads(open(args.kb_graph +  '/filtered_property_wikidata4.json').read())
        ID_ENTITY = json.loads(open(args.kb_graph +  '/items_wikidata_n.json').read())
        print("*\t Loaded types graph for GOLD TYPES")
    else:
        ID_RELATION = json.loads(open(args.kb_graph +  '/filtered_property_wikidata4.json').read())
        ID_ENTITY = json.loads(open(args.kb_graph +  '/items_wikidata_n.json').read())

    corpora = get_corpora(args)

    spice = SPICEDataset(args)

    print('Dataset created!')

    for corpus_type in corpora.keys():
        a_lst = [(f, args, spice, corpus_type) for f in corpora[corpus_type] ]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        gc.collect()
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.extend(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                logger.info(f'Saving shard: {pt_file}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []
                    gc.collect()

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            logger.info(f'Saving shard: {pt_file}')
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []

    logger.info(f' total UNK in formulas: {spice.nbunks}')

def _format_to_lines(params):
    f, args, spice, corpus_type = params
    data = spice._prepare_data([f], infiles=True, isTest = (corpus_type in ['test', 'valid']))
    return data

def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']

    spice = SPICEDataset(args)

    for corpus_type in datasets:
        a_lst = []
        for json_f in glob(args.raw_path + '.' + corpus_type + '.*.json'):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt')), spice))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()

def _format_to_bert(params):
    corpus_type, json_file, args, save_file, spice = params
    is_test = corpus_type in ['test', 'valid']
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for b_data in jobs:
        if (b_data is None):
            continue
        if is_test:
            input, inputChunks, starts, ends, kgVocIds, entity_map, str_logical_form, logical_form, \
                    qtype, description, question, results, answer, turnID = b_data
        else:
            input, inputChunks, starts, ends, kgVocIds, entity_map, str_logical_form, logical_form, \
                    qtype, description = b_data

        b_input = [spice.bert_binarise(x.strip().split(' ')) for x in input]
        b_data_dict = {INPUT: b_input, SEGMENT: inputChunks,
                       START: starts, END: ends,
                       DYNBIN: kgVocIds, ENTITY_MAP: entity_map,
                       STR_LOGICAL_FORM: str_logical_form,
                       LOGICAL_FORM: logical_form,
                       QUESTION_TYPE: qtype,
                       DESCRIPTION: description}
        if is_test:
            b_data_dict[QUESTION] = question
            b_data_dict[RESULTS] = results
            b_data_dict[ANSWER] = answer
            b_data_dict[TURN_ID] = turnID

        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
