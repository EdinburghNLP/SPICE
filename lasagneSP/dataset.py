import json
from tqdm import tqdm
import time
from glob import glob
from transformers import BertTokenizer
try:
    from torchtext.data import Field, Example, Dataset
except:
    from torchtext.legacy.data import Field, Example, Dataset

#import constants
from myconstants import *

class CSQADataset:
    def __init__(self):
        #self.train_path = str(ROOT_PATH.parent) + args.data_path + '/train/*'
        #self.val_path = str(ROOT_PATH.parent) + args.data_path + '/valid/*'
        #self.test_path = str(ROOT_PATH.parent) + args.data_path + '/test/*'
        self.train_path =  args.data_path + '/train/*'
        self.val_path =  args.data_path + '/valid/*'
        self.test_path =  args.data_path + '/test/*'
        self.load_data_and_fields()


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

    def _prepare_data(self, data, infiles=False):
        '''
        When infiles = True, we are actually passing the json file paths, this is more 
        memory efficient rather the loading whole data into memory and then process it.
        '''
        input_data = []
        helper_data = {QUESTION_TYPE: []}
        for conversation in tqdm(data, total=len(data)):
            if infiles:
                #print('Loading ', conversation)
                f = conversation
                fID = f.split('/')
                fID = f'{fID[-3]}#{fID[-2]}#{fID[-1].split(".json")[0]}'
                conversation = json.load(open(conversation, 'r'))
            assert infiles, 'This should be true for fast loading'
            
            prev_user_conv = None
            prev_system_conv = None
            is_clarification = False
            is_history_ner_spurious = False
            turns = len(conversation) // 2
            for i in range(turns):
                input = []
                logical_form = []
                ner_tag = []
                coref = []
                graph_cls = []

                if is_clarification:
                    is_clarification = False
                    continue

                user = conversation[2*i]
                system = conversation[2*i + 1]
                if 'is_ner_spurious' not in user.keys():
                    user['is_ner_spurious'] = True
                if 'is_ner_spurious' not in system.keys():
                    system['is_ner_spurious'] = True

                if user['question-type'] == 'Clarification':
                    # get next context
                    is_clarification = True
                    next_user = conversation[2*(i+1)]
                    next_system = conversation[2*(i+1) + 1]
                    if 'is_ner_spurious' not in next_user.keys():
                        next_user['is_ner_spurious'] = True
                    if 'is_ner_spurious' not in next_system.keys():
                        next_system['is_ner_spurious'] = True

                    # skip if ner history is spurious
                    if is_history_ner_spurious:
                        is_history_ner_spurious = False
                        if not next_user['is_ner_spurious'] and not next_system['is_ner_spurious']:
                            prev_user_conv = next_user.copy()
                            prev_system_conv = next_system.copy()
                        else:
                            is_history_ner_spurious = True
                        continue

                    # skip if ner is spurious
                    if user['is_ner_spurious'] or system['is_ner_spurious'] or next_user['is_ner_spurious'] or next_system['is_ner_spurious']:
                        is_history_ner_spurious = True
                        continue

                    # skip if no gold action (or spurious)
                    #if 'gold_actions' not in next_system or next_system['is_spurious']:
                    if 'sparql_delex' not in next_system or next_system['sparql_spurious']:
                        prev_user_conv = next_user.copy()
                        prev_system_conv = next_system.copy()
                        continue

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                        ner_tag.extend([O, O, O, O])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                    # user context
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # system context
                    for context in system['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # next user context
                    for context in next_user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # coref entities - prepare coref values
                    #action_entities = [action[1] for action in next_system[GOLD_ACTIONS] if action[0] == ENTITY]
                    action_entities = [action[1] for action in self.format_delex(next_system) if action[0] == ENTITY]
                    for context in reversed(user['context'] + system['context'] + next_user['context']):
                        if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                            coref.append(str(action_entities.index(context[2])))
                        else:
                            coref.append(NA_TOKEN)

                    if i == 0:
                        coref.extend([NA_TOKEN, NA_TOKEN, NA_TOKEN, NA_TOKEN])
                    else:
                        coref.append(NA_TOKEN)
                        for context in reversed(prev_system_conv['context']):
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                        coref.append(NA_TOKEN)
                        for context in reversed(prev_user_conv['context']):
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                    # get gold actions
                    #gold_actions = next_system[GOLD_ACTIONS]
                    gold_actions = self.format_delex(next_system)

                    # track context history
                    prev_user_conv = next_user.copy()
                    prev_system_conv = next_system.copy()
                else:
                    if is_history_ner_spurious: # skip if history is ner spurious
                        is_history_ner_spurious = False
                        if not user['is_ner_spurious'] and not system['is_ner_spurious']:
                            prev_user_conv = user.copy()
                            prev_system_conv = system.copy()
                        else:
                            is_history_ner_spurious = True

                        continue
                    if user['is_ner_spurious'] or system['is_ner_spurious']: # skip if ner is spurious
                        is_history_ner_spurious = True
                        continue

                    #if GOLD_ACTIONS not in system or system['is_spurious']: # skip if logical form is spurious
                    if GOLD_ACTIONS not in system or system['sparql_spurious']: # skip if logical form is spurious
                        prev_user_conv = user.copy()
                        prev_system_conv = system.copy()
                        continue

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                        ner_tag.extend([O, O, O, O])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                    # user context
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # coref entities - prepare coref values
                    #action_entities = [action[1] for action in system[GOLD_ACTIONS] if action[0] == ENTITY]
                    action_entities = [action[1] for action in self.format_delex(system) if action[0] == ENTITY]
                    for context in reversed(user['context']):
                        #if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                        if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                            coref.append(str(action_entities.index(context[2])))
                        else:
                            coref.append(NA_TOKEN)

                    if i == 0:
                        coref.extend([NA_TOKEN, NA_TOKEN, NA_TOKEN, NA_TOKEN])
                    else:
                        coref.append(NA_TOKEN)
                        for context in reversed(prev_system_conv['context']):
                            #if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                        coref.append(NA_TOKEN)
                        for context in reversed(prev_user_conv['context']):
                            #if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                    # get gold actions
                    #gold_actions = system[GOLD_ACTIONS]
                    gold_actions = self.format_delex(system)
                    #print(gold_actions)

                    # track context history
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()

                # prepare logical form
                for action in gold_actions:
                    if action[0] == ACTION:
                        logical_form.append(action[1])
                        graph_cls.append(NA_TOKEN)
                    elif action[0] == RELATION:
                        logical_form.append(RELATION)
                        graph_cls.append(action[1])
                    elif action[0] == TYPE:
                        logical_form.append(TYPE)
                        graph_cls.append(action[1])
                    elif action[0] == ENTITY:
                        logical_form.append(PREV_ANSWER if action[1] == PREV_ANSWER else ENTITY)
                        graph_cls.append(NA_TOKEN)
                    elif action[0] == VALUE:
                        logical_form.append(action[0])
                        graph_cls.append(NA_TOKEN)
                    else:
                        raise Exception(f'Unkown logical form action {action[0]}')

                assert len(input) == len(ner_tag)
                assert len(input) == len(coref)
                assert len(logical_form) == len(graph_cls)

                input_data.append([input, logical_form, ner_tag, list(reversed(coref)), graph_cls, f'{fID}#{i}' if fID else ''])
                helper_data[QUESTION_TYPE].append(user['question-type'])

        return input_data, helper_data

    def get_inference_data(self, inference_partition):
        assert inference_partition == 'test'
        if inference_partition == 'val':
            files = glob(self.val_path + '/*.json')
        elif inference_partition == 'test':
            files = glob(self.test_path + '/*.json')
        else:
            raise ValueError(f'Unknown inference partion {inference_partition}')

        partition = []
        for f in files:
            with open(f) as json_file:
                partition.append([json.load(json_file), f])

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize
        inference_data = []
        fidsprint = open('fids.txt', 'a')

        for (conversation, filename) in partition:
            f = filename
            fID = f.split('/')
            fID = f'{fID[-3]}#{fID[-2]}#{fID[-1].split(".json")[0]}'
            
            
            is_clarification = False
            prev_user_conv = {}
            prev_system_conv = {}
            turns = len(conversation) // 2
            for i in range(turns):
                fidsprint.write(filename + ' ## ' + f'{fID}#{i}' + '\n')
                input = []
                gold_entities = []

                if is_clarification:
                    is_clarification = False
                    continue

                user = conversation[2*i]
                system = conversation[2*i + 1]

                if i > 0 and 'context' not in prev_system_conv:
                    if len(prev_system_conv['entities_in_utterance']) > 0:
                        tok_utterance = tokenizer(prev_system_conv['utterance'].lower())
                        prev_system_conv['context'] = [[i, tok] for i, tok in enumerate(tok_utterance)]
                    elif prev_system_conv['utterance'].isnumeric():
                        prev_system_conv['context'] = [[0, 'num']]
                    elif prev_system_conv['utterance'] == 'YES':
                        prev_system_conv['context'] = [[0, 'yes']]
                    elif prev_system_conv['utterance'] == 'NO':
                        prev_system_conv['context'] = [[0, 'no']]
                    elif prev_system_conv['utterance'] == 'YES and NO respectively':
                        prev_system_conv['context'] = [[0, 'no']]
                    elif prev_system_conv['utterance'] == 'NO and YES respectively':
                        prev_system_conv['context'] = [[0, 'no']]
                    elif prev_system_conv['utterance'][0].isnumeric():
                        prev_system_conv['context'] = [[0, 'num']]

                if user['question-type'] == 'Clarification':
                    # get next context
                    is_clarification = True
                    next_user = conversation[2*(i+1)]
                    next_system = conversation[2*(i+1) + 1]

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                    # user context
                    for context in user['context']:
                        input.append(context[1])

                    # system context
                    for context in system['context']:
                        input.append(context[1])

                    # next user context
                    for context in next_user['context']:
                        input.append(context[1])

                    question_type = [user['question-type'], next_user['question-type']] if 'question-type' in next_user else user['question-type']
                    description = user[DESCRIPTION]
                    results = next_system['sparql_entities'] if 'sparql_entities' in next_system.keys() else next_system['all_entities']
                    #results = next_system['all_entities']
                    #answer = next_system['utterance']
                    if question_type in ['Quantitative Reasoning (Count) (All)',
                                            'Comparative Reasoning (Count) (All)']\
                            and 'sparql_entities' in next_system.keys() :
                        answer = f'{len(next_system["sparql_entities"])}'
                    else:
                        answer = next_system['utterance']
                    #gold_actions = next_system[GOLD_ACTIONS] if GOLD_ACTIONS in next_system else None
                    gold_actions = self.format_delex(next_system) if GOLD_ACTIONS in next_system else None
                    
                    prev_answer = prev_system_conv['all_entities'] if 'all_entities' in prev_system_conv else None
                    context_entities = user['entities_in_utterance'] + system['entities_in_utterance']
                    if 'entities_in_utterance' in next_user: context_entities.extend(next_user['entities_in_utterance'])
                    if 'entities_in_utterance' in prev_user_conv: context_entities.extend(prev_user_conv['entities_in_utterance'])
                    if 'entities_in_utterance' in prev_system_conv: context_entities.extend(prev_system_conv['entities_in_utterance'])

                    # track context history
                    prev_user_conv = next_user.copy()
                    prev_system_conv = next_system.copy()
                else:
                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                    if 'context' not in user:
                        tok_utterance = tokenizer(user['utterance'].lower())
                        user['context'] = [[i, tok] for i, tok in enumerate(tok_utterance)]

                    # user context
                    for context in user['context']:
                        input.append(context[1])

                    question_type = user['question-type']
                    description = user[DESCRIPTION]
                    #results = system['all_entities']
                    #answer = system['utterance']
                    results = system['sparql_entities'] if 'sparql_entities' in system.keys() else system['all_entities']
                    if question_type in ['Quantitative Reasoning (Count) (All)',
                          'Comparative Reasoning (Count) (All)']\
                            and 'sparql_entities' in system.keys() :
                        answer = f'{len(system["sparql_entities"])}'
                    else:
                        answer = system['utterance']
                    #gold_actions = system[GOLD_ACTIONS] if GOLD_ACTIONS in system else None
                    gold_actions = self.format_delex(system) if GOLD_ACTIONS in system else None
                    prev_results = prev_system_conv['all_entities'] if 'all_entities' in prev_system_conv else None
                    context_entities = user['entities_in_utterance'] + system['entities_in_utterance']
                    if 'entities_in_utterance' in prev_user_conv: context_entities.extend(prev_user_conv['entities_in_utterance'])
                    if 'entities_in_utterance' in prev_system_conv: context_entities.extend(prev_system_conv['entities_in_utterance'])

                    # track context history
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()

                if gold_actions == None:
                    # for one sub type of questions the data is kept unannotated, as of now we skip eval on those
                    continue
                inference_data.append({
                    QUESTION_TYPE: question_type,
                    DESCRIPTION: description,
                    QUESTION: user['utterance'],
                    CONTEXT_QUESTION: input,
                    CONTEXT_ENTITIES: context_entities,
                    ANSWER: answer,
                    RESULTS: results,
                    PREV_RESULTS: prev_results,
                    GOLD_ACTIONS: gold_actions,
                    TURN_ID: f'{fID}#{i}'
                })

        fidsprint.close()
        return inference_data

    def _make_torchtext_dataset(self, data, fields):
        examples = [Example.fromlist(i, fields) for i in data]
        return Dataset(examples, fields)

    
    
    def get_corpora(self, args):
        def read_split_map(mapfile, data_path, splits):
            ret = {}
            for x in splits:
                ret[x] = []
            mapf = open(mapfile)
            for l in mapf.readlines():
                f, sp = l.strip().split('\t')
                if sp in splits:
                    ret[sp].append(f'{data_path}/{f}')
                #if sp not in f:
                #    print('switch')
                #else: print('stay')
            return ret


        if args.mapsplits:
            print('Reading mapsplits ', args.mapfile)
            corpora = read_split_map(args.mapfile, args.data_path, ['train', 'valid', 'test'])
        else:
            print('simple read dataset.....\n')
            #train_path = args.data_path + '/train/*'
            #val_path = args.data_path + '/valid/*'
            #test_path = args.data_path + '/test/*'
            train_files = glob(self.train_path + '/*.json')
            valid_files = glob(self.val_path + '/*.json')
            test_files = glob(self.test_path + '/*.json')
            corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
        for x in ['train', 'valid', 'test']:
            print(f'{x} files ', f'{args.data_path}/{x}', len(corpora[x]))

        return corpora
    
    def load_data_and_fields(self):
        infiles = True
        corpora = self.get_corpora(args)
        train_files = corpora['train']
        val_files = corpora['valid']
        test_files = corpora['test']
        train, val, test = [], [], []
        # read data
        print('Train files ', self.train_path, len(train_files))
        assert len(train_files) > 10, 'Something wrong with train path'
        tic = time.perf_counter()
        if not infiles:
            for f in train_files:
                with open(f) as json_file:
                    train.append(json.load(json_file))
        else:
            train = train_files

        print("preparing data...")
        
        train, self.train_helper = self._prepare_data(train, infiles=infiles)
        toc = time.perf_counter()
        print(f'==> Finished loading train {toc - tic:0.2f}s')
        #assert len(val_files) > 10, 'Something wrong with val path'
        tic = time.perf_counter()
        if not infiles:
            for f in val_files:
                with open(f) as json_file:
                    val.append(json.load(json_file))
        else:
            val = val_files
                
        val, self.val_helper = self._prepare_data(val, infiles=infiles)
        toc = time.perf_counter()
        print(f'==> Finished loading Validation {toc - tic:0.2f}s')
        
        assert len(test_files) > 10, 'Something wrong with test path' + str(self.test_path)
        if not infiles:
            for f in test_files:
                with open(f) as json_file:
                    test.append(json.load(json_file))
        else:
            test = test_files

        # prepare data
        #train, self.train_helper = self._prepare_data(train)
        #val, self.val_helper = self._prepare_data(val)
        test, self.test_helper = self._prepare_data(test, infiles=infiles)

        # create fields
        self.input_field = Field(init_token=START_TOKEN,
                                eos_token=CTX_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=UNK_TOKEN,
                                lower=True,
                                batch_first=True)

        self.lf_field = Field(init_token=START_TOKEN,
                                eos_token=END_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=UNK_TOKEN,
                                lower=True,
                                batch_first=True)

        self.ner_field = Field(init_token=O,
                                eos_token=O,
                                pad_token=PAD_TOKEN,
                                unk_token=O,
                                batch_first=True)

        self.coref_field = Field(init_token='0',
                                eos_token='0',
                                pad_token=PAD_TOKEN,
                                unk_token='0',
                                batch_first=True)

        self.graph_field = Field(init_token=NA_TOKEN,
                                eos_token=NA_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=NA_TOKEN,
                                batch_first=True)

        fields_tuple = [(INPUT, self.input_field), (LOGICAL_FORM, self.lf_field),
                        (NER, self.ner_field), (COREF, self.coref_field),
                        (GRAPH, self.graph_field)]

        # create toechtext datasets
        self.train_data = self._make_torchtext_dataset(train, fields_tuple)
        self.val_data = self._make_torchtext_dataset(val, fields_tuple)
        self.test_data = self._make_torchtext_dataset(test, fields_tuple)

        # build vocabularies
        self.input_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0, vectors='glove.840B.300d')
        self.lf_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.ner_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.coref_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.graph_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
    
    
    def load_data_and_fields_simple(self):
        infiles = True
        train, val, test = [], [], []
        # read data
        train_files = glob(self.train_path + '/*.json')
        print('Train files ', self.train_path, len(train_files))
        assert len(train_files) > 10, 'Something wrong with train path'
        tic = time.perf_counter()
        if not infiles:
            for f in train_files:
                with open(f) as json_file:
                    train.append(json.load(json_file))
        else:
            train = train_files

        print("preparing data...")
        
        train, self.train_helper = self._prepare_data(train, infiles=infiles)
        toc = time.perf_counter()
        print(f'==> Finished loading train {toc - tic:0.2f}s')
        val_files = glob(self.val_path + '/*.json')
        #assert len(val_files) > 10, 'Something wrong with val path'
        tic = time.perf_counter()
        if not infiles:
            for f in val_files:
                with open(f) as json_file:
                    val.append(json.load(json_file))
        else:
            val = val_files
                
        val, self.val_helper = self._prepare_data(val, infiles=infiles)
        toc = time.perf_counter()
        print(f'==> Finished loading Validation {toc - tic:0.2f}s')
        
        test_files = glob(self.test_path + '/*.json')
        assert len(test_files) > 10, 'Something wrong with test path' + str(self.test_path)
        if not infiles:
            for f in test_files:
                with open(f) as json_file:
                    test.append(json.load(json_file))
        else:
            test = test_files

        # prepare data
        #train, self.train_helper = self._prepare_data(train)
        #val, self.val_helper = self._prepare_data(val)
        test, self.test_helper = self._prepare_data(test, infiles=infiles)

        # create fields
        self.input_field = Field(init_token=START_TOKEN,
                                eos_token=CTX_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=UNK_TOKEN,
                                lower=True,
                                batch_first=True)

        self.lf_field = Field(init_token=START_TOKEN,
                                eos_token=END_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=UNK_TOKEN,
                                lower=True,
                                batch_first=True)

        self.ner_field = Field(init_token=O,
                                eos_token=O,
                                pad_token=PAD_TOKEN,
                                unk_token=O,
                                batch_first=True)

        self.coref_field = Field(init_token='0',
                                eos_token='0',
                                pad_token=PAD_TOKEN,
                                unk_token='0',
                                batch_first=True)

        self.graph_field = Field(init_token=NA_TOKEN,
                                eos_token=NA_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=NA_TOKEN,
                                batch_first=True)

        fields_tuple = [(INPUT, self.input_field), (LOGICAL_FORM, self.lf_field),
                        (NER, self.ner_field), (COREF, self.coref_field),
                        (GRAPH, self.graph_field)]

        # create toechtext datasets
        self.train_data = self._make_torchtext_dataset(train, fields_tuple)
        self.val_data = self._make_torchtext_dataset(val, fields_tuple)
        self.test_data = self._make_torchtext_dataset(test, fields_tuple)

        # build vocabularies
        self.input_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0, vectors='glove.840B.300d')
        self.lf_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.ner_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.coref_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.graph_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)

    def get_data(self):
        return self.train_data, self.val_data, self.test_data

    def get_data_helper(self):
        return self.train_helper, self.val_helper, self.test_helper

    def get_fields(self):
        return {
            INPUT: self.input_field,
            LOGICAL_FORM: self.lf_field,
            NER: self.ner_field,
            COREF: self.coref_field,
            GRAPH: self.graph_field,
        }

    def get_vocabs(self):
        return {
            INPUT: self.input_field.vocab,
            LOGICAL_FORM: self.lf_field.vocab,
            NER: self.ner_field.vocab,
            COREF: self.coref_field.vocab,
            GRAPH: self.graph_field.vocab,
        }
