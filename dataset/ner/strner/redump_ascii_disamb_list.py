import json, os


data_path='/home/s1959796/spice_dataset_project/spice_project/nel_scripts/wikidata_proc_json/wikidata_proc_json_2/'
file_list = ['filtered_property_wikidata4.json' , 'items_wikidata_n.json']

for filename in file_list:
    fpath = os.path.join(data_path, filename)
    print('Loading json file from ', fpath)
    id_val_dict = json.load(open(fpath, 'r'))
    ofpath = fpath+'.redump'
    new_entity_id = {}
    for id, val in id_val_dict.items():
        if val in new_entity_id.keys():
            new_entity_id[val].append(id)
        else:
            new_entity_id[val] = [id]
    json.dump(new_entity_id, open(ofpath, 'w', encoding='utf8'), indent=2, ensure_ascii=False)

