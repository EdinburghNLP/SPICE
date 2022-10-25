import json
import traceback
import os
from glob import glob
from multiprocess import Pool
from tqdm import tqdm
import argparse
import pathlib
from os.path import exists
from unidecode import unidecode
from elasticsearch import Elasticsearch
from allennlp.predictors.predictor import Predictor

'''
B - 'beginning'
I - 'inside'
L - 'last'
O - 'outside'
U - 'unit'
'''

elastic_search = Elasticsearch([{'host': 'localhost', 'port': 9200}]) # connect to elastic search server
#allennlp_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
# do not lower case

def allen_simple_token_ner(utterance, predictor_elmo):
    simple_tokens = utterance.lower().split()
    result = predictor_elmo.predict(sentence=utterance)
    new_ner = []
    s_idx = 0
    cache_w = ''
    combining = False
    allennlp_words = result['words']
    allennlp_tags = result['tags']
    tagged_words = []
    
    for t_idx, t in enumerate(result['tags']):
        if t == 'O':
            continue
        if t.startswith('B-'):
            cache_w += result['words'][t_idx]
        if t.startswith('I-'):
            cache_w += ' '+result['words'][t_idx]
        if t.startswith('L-'):
            cache_w += ' ' +result['words'][t_idx]
            tagged_words.append(cache_w)
            cache_w = ''
        if t.startswith('U-'):
            tagged_words.append(result['words'][t_idx])


    return new_ner, allennlp_words, allennlp_tags, tagged_words


def elasticsearch_query(query, res_size=1):
    res = elastic_search.search(index='csqa_wikidata', doc_type='entities', body={'size': res_size, 'query': {'match': {'label': {'query': unidecode(query), 'fuzziness': '1'}}}})
    results = []
    for hit in res['hits']['hits']: results.append(hit['_source']['id'])
    return results


def add_nel_file(data_file,outfile, predictor_elmo):
    if exists(outfile):
        print('File {f} already exist ... '.format(f=outfile))
        return
    print('File {f} will be processed ... '.format(f=outfile))
    try:
        return _add_nel_file(data_file,outfile, predictor_elmo)
    except Exception as e:
        print(traceback.format_exc())
        print('Failed ', data_file)
        return 0


def _add_nel_file(data_file, outfile, predictor_elmo):
    
    conversation_triple_length = []
    conversation_triples = set()
    all_kg_element_set = set()
    input_data = []

    try:
        data = json.load(open(data_file, 'r'))
    except json.decoder.JSONDecodeError as e:
        print('Failed loading json file: ', data_file)
        raise e
    for conversation in [data]:
        is_clarification = False
        prev_user_conv = None
        prev_system_conv = None
        turns = len(conversation) // 2

        for i in range(turns):
            input = []
            logical_form = []
            # If the previous was a clarification question we basically took next 
            # logical form so need to skip
            if is_clarification:
                is_clarification = False
                continue
            user = conversation[2*i]
            system = conversation[2*i + 1]
            utterance = user['utterance']
            _, _, allennlp_tags, tagged_words = allen_simple_token_ner(utterance, predictor_elmo)
            es_links = []
            for w in tagged_words:
                r=elasticsearch_query(query=w, res_size=5)
                es_links.append(r)

            user['tagged_words'] = tagged_words
            user['allennlp_tags'] = allennlp_tags
            user['es_links'] = es_links
            if 'utterance' in system.keys():
                utterance = system['utterance']
                _, _, allennlp_tags, tagged_words = allen_simple_token_ner(utterance, predictor_elmo)
                es_links = []
                for w in tagged_words:
                    r=elasticsearch_query(query=w, res_size=5)
                    es_links.append(r)

                system['tagged_words'] = tagged_words
                system['allennlp_tags'] = allennlp_tags
                system['es_links'] = es_links

    if exists(outfile):
        print('File {f} already exist ... '.format(f=outfile))
        #raise Exception('File exist check input')
    base_folder = os.path.dirname(outfile)
    pathlib.Path(base_folder).mkdir(exist_ok=True, parents=True)
    
    json.dump(data, open(outfile, 'w', encoding='utf8'), indent=2, ensure_ascii=False)


def process_files(args_list, colour=None):
    print('Process files ...')
    predictor_elmo = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
    for ifile, ofile in tqdm(args_list, total=len(args_list), colour=colour):
        add_nel_file(ifile, ofile, predictor_elmo)

def tag_data(args):
    allfiles = []
    for f in open(args.file_path, 'r').readlines():
        allfiles.append(f.rstrip())
    if args.end > 0:
        allfiles = allfiles[args.start:args.end]
    print('Processing from {s} and {e}, will save at {p}'.format(s=args.start,e=args.end,p=args.save_path))
    #args_list = [(inp_file, os.path.join(args.save_path, corpus_type, os.path.basename(inp_file))) for inp_file in allfiles]
    def get_inp_out_file(filename, local_path, out_dir):
        lsys='/disk/scratch/parag/tagged/' # set this path according to the list
        filename = filename.replace('.tagged', '')
        inp_file = filename.replace(lsys, local_path)
        outfile = filename.replace(lsys, out_dir) + '.tagged'
        return (inp_file, outfile)
    args_list = [get_inp_out_file(inp_file, args.data_path, args.save_path) for inp_file in allfiles]
    process_files(args_list)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # python preprocess.py -data_path /home/s1959796/csqparsing/dataset/data/version_splits -save_path /home/s1959796/csqparsing/nel_data -dataset train -debug 1
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", required=False)
    parser.add_argument("-save_path", required=True)
    parser.add_argument("-file_path", required=False)
    
    parser.add_argument('-dataset', default='')
    parser.add_argument('-n_cpus', default=10, type=int)
    parser.add_argument('-start', default=10, type=int)
    parser.add_argument('-end', default=10, type=int)
    parser.add_argument('-debug', nargs='?',const=False,default=False)
    args = parser.parse_args()
    tag_data(args)