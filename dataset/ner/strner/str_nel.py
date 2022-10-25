import json
import traceback
import os, re
from glob import glob
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import argparse
import pathlib
import ahocorasick
from os.path import exists
from unidecode import unidecode
import pickle

basic_stops = ['where', 'did', 'how', 'many', 'where', 'when', 'which']

entity_count = json.load(open('entity_count.json', 'r'))
def disambiguate(listofids):
    counts = []
    for id in listofids:
        c=entity_count.get(id, -1)
        #if c > -1:
        #    print(id)
        counts.append(c)
    max_id = counts.index(max(counts))
    return listofids[max_id]

def preprocess(text):
  text = text.lower()
  text = text.translate(str.maketrans('', '', ",.?"))
  text = ' '.join([t  for t in text.split() if t not in basic_stops])
  return text.lower()

def create_index(data_path, file_list):
    index = ahocorasick.Automaton()
    for filename in file_list:
        fpath = os.path.join(data_path, filename)
        print('Loading json file from ', fpath)
        id_val_dict = json.load(open(fpath, 'r'))
        count = 0
        for val, idlist in tqdm(id_val_dict.items(), total=len(id_val_dict)):
            ## you could disambiguate later based on the counts... but if we are taking the top count then we might just add the top one in our index. This way the whole process is faster.
            disambiguated_id = disambiguate(idlist)
            index.add_word(preprocess(val), (disambiguated_id, val))
            count += 1
        print(f'Added {count} items.')
    
    index.make_automaton()
    return index


def str_nel_long(utterance, automaton):
    tagged_words = []
    elinks = []
    start_end_pos = []
    preptext = preprocess(utterance)
    for end_index, found_value in automaton.iter_long(preptext):
        text_value = found_value[1]
        id = found_value[0]
        end = end_index - 1
        start = end - len(text_value)
        start_end_pos.append((start, end))
        tagged_words.append(text_value)
        elinks.append(id)
    return tagged_words, elinks, start_end_pos


def str_nel(utterance, automaton):
    tagged_words = []
    elinks = []
    start_end_pos = []
    preptext = preprocess(utterance)
    matched_items = [] # start, end, len, value
    for end_index, found_value in automaton.iter(preptext):
      text_value = found_value[1]
      id = found_value[0]
      start_index = end_index - len(text_value) + 1
      if (start_index - 1 < 0 or preptext[start_index - 1] == ' ') and (end_index == len(preptext) - 1 or preptext[end_index+1] == ' '):
        matched_items.append((start_index, end_index, end_index - start_index, found_value))
    
    keep_items  = []
    matched_items = sorted(matched_items, key=lambda x: x[2], reverse=True)
    for m in matched_items:
      start_index, end_index, vlen, v = m
      flag=True
      for k in keep_items:
        if (start_index >= k[0] and start_index <= k[1]) or (end_index >= k[0] and end_index <= k[1]):  # other condition where curr string encapculated m is not needed as we have already sorted and kept bigger string first
          flag=False
          break
      if flag:
        keep_items.append(m)
        tagged_words.append(m[3][1])
        elinks.append(m[3][0])
        #start_end_pos.append((m[0], m[1]))
    
    return tagged_words, elinks


def add_nel_file(params):
    data_file,outfile, automaton = params
    if exists(outfile):
        print('File {f} already exist ... '.format(f=outfile))
        return
    try:
        return _add_nel_file(data_file,outfile, automaton)
    except Exception as e:
        print(traceback.format_exc())
        print('Failed ', data_file)
        return 0


def _add_nel_file(data_file, outfile, automaton):

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
            tagged_words, es_links = str_nel(utterance, automaton)
            user['tagged_words'] = tagged_words
            user['es_links'] = es_links

            if 'utterance' in system.keys():
                utterance = system['utterance']
                tagged_words, es_links = str_nel(utterance, automaton)
                system['tagged_words'] = tagged_words
                system['es_links'] = es_links



    if exists(outfile):
        print('File {f} already exist ... '.format(f=outfile))
        #raise Exception('File exist check input')
    base_folder = os.path.dirname(outfile)
    pathlib.Path(base_folder).mkdir(exist_ok=True, parents=True)
    
    json.dump(data, open(outfile, 'w', encoding='utf8'), indent=2, ensure_ascii=False)
    return outfile


def process_files_parallel(args_list, automaton, colour=None):
    print('Process files ...')
    a_lst = [(ifile, ofile, automaton) for ifile, ofile in args_list]
    pool = ThreadPool(args.n_cpus)
    outfiles = set()
    for processed_filename in tqdm(pool.imap_unordered(add_nel_file, a_lst), total=len(a_lst)):
        if processed_filename in outfiles:
            print('processed_filename', processed_filename)
        outfiles.add(processed_filename)


def tag_str_all_files(args):
    '''
    after some issues in data were fixed some more files could be tagged and added
    '''
    automaton_filename = 'automaton_noproperty.pkl'
    data_path='wikidata_proc_json/wikidata_proc_json_2/'
    #file_list = ['filtered_property_wikidata4.json.redump' , 'items_wikidata_n.json.redump']
    file_list = ['items_wikidata_n.json.redump']
    if not os.path.exists(automaton_filename):
        automaton = create_index(data_path=data_path, file_list=file_list)
        pickle.dump(automaton, open(automaton_filename, 'wb'))
    else:
        automaton = pickle.load(open(automaton_filename, 'rb'))
    allfiles = []
    for f in open(args.file_path, 'r').readlines():
        allfiles.append(f.rstrip())
    if args.end > 0:
        allfiles = allfiles[args.start:args.end]
    print('Processing from {s} and {e}, will save at {p}'.format(s=args.start,e=args.end,p=args.save_path))
    #args_list = [(inp_file, os.path.join(args.save_path, corpus_type, os.path.basename(inp_file))) for inp_file in allfiles]
    def get_inp_out_file(filename, local_path, out_dir):
        lsys=args.data_path
        inp_file = filename
        outfile = filename.replace(lsys, out_dir) + '.strtaggedwithoutproperty'
        return (inp_file, outfile)
    args_list = [get_inp_out_file(inp_file, args.data_path, args.save_path) for inp_file in allfiles]
    #print(args_list)
    process_files_parallel(args_list, automaton)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

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
    tag_str_all_files(args)
