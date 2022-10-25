import ahocorasick
import json
import os, re
from tqdm import tqdm

data_path='wikidata_proc_json/wikidata_proc_json_2/'
file_list = ['filtered_property_wikidata4.json' , 'items_wikidata_n.json']

def preprocess(text):
  return '_{}_'.format(re.sub('[^a-z]', '_', text.lower()))

def create_index(data_path, file_list):
    index = ahocorasick.Automaton()
    for filename in file_list:
        fpath = os.path.join(data_path, filename)
        print('Loading json file from ', fpath)
        id_val_dict = json.load(open(fpath, 'r'))
        count = 0
        for id, val in tqdm(id_val_dict.items(), total=len(id_val_dict)):
            index.add_word(preprocess(val), (id, val))
            count += 1
        print(f'Added {count} items.')
    
    index.make_automaton()
    return index


def find_indices_and_position(text, searcher):
    result = dict()
    preptext = preprocess(text)
    print(preptext)
    for end_index, found_value in searcher.iter_long(preptext):
        print(found_value)
        text_value = found_value[1]
        id = found_value[1]
        print(end_index, text_value)
        end = end_index - 1
        start = end - len(text_value)
        occurrence_text = text[start:end]
        print('occurrence_text ', occurrence_text)
        result[(start, end)] = text_value
    return result


index = create_index(data_path=data_path, file_list=file_list)
utterances = open('example_utterances.txt' , 'r').readlines()

for u in utterances:
    print(find_indices_and_position(u, index))
