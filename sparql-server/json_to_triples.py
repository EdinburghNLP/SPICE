import os, json
from tqdm import tqdm
from rdflib import Literal

directory = '../wikidata_proc_json_2'
out_dir = 'ttl_files'
#WD="<http://www.wikidata.org/entity/"
WD="wd:"
WDT="wdt:"

num_p31_skipped = 0

def get_triple(s, p, o, sl=WD, pl=WDT, ol=WD):
    triple = "{wds}{s} {wdp}{p} {wdo}{o} .".format(s=s, p=p, o=o, wds=sl, wdp=pl, wdo=ol)
    return triple

def write_prefix(outf):
    prefixlist = open('prefix.ttl', 'r').readlines()
    for p in prefixlist:
        outf.write(p)


def read_wikidata_short_1(directory, out_dir):
    '''
    The outermost index indexes the ‘subject entities’. The value of each wikidata_short_1[key] is a dict. with keys as pid’s (id’s corresponding to the relations/predicates). The value of each wikidata_short_1[key][pid] is a list of object entities with the outer key as the subject entity.
    The (key, pid, object entity) is a usual KB triple.
    '''
    global num_p31_skipped
    filename = os.path.join(directory, 'wikidata_short_1.json')
    outfilename = os.path.join(out_dir, 'wikidata_short_1.ttl')
    print('Loading file ', filename)
    jobj = json.load(open(filename, 'r'))
    print('Loaded file ', filename)
    outf = open(outfilename, 'w')
    write_prefix(outf)
    for key in jobj.keys():
        for pid in jobj[key].keys():
            if pid == 'P31':
                num_p31_skipped += 1
                continue
            if len(jobj[key][pid]) == 0:
                t = get_triple(s=key, p = pid, o = 'Q0')
                outf.write(t + '\n')
            else:
                for obj in jobj[key][pid]:
                    triple = get_triple(s=key, p=pid, o=obj)
                    outf.write(triple + '\n')
    
    print(num_p31_skipped)
    return None

def read_wikidata_short_2(directory, out_dir):
    '''
    The outermost index indexes the ‘subject entities’. The value of each wikidata_short_1[key] is a dict. with keys as pid’s (id’s corresponding to the relations/predicates). The value of each wikidata_short_1[key][pid] is a list of object entities with the outer key as the subject entity.
    The (key, pid, object entity) is a usual KB triple.
    '''
    global num_p31_skipped
    filename = os.path.join(directory, 'wikidata_short_2.json')
    outfilename = os.path.join(out_dir, 'wikidata_short_2.ttl')
    print('Loading file ', filename)
    jobj = json.load(open(filename, 'r'))
    print('Loaded file ', filename)
    outf = open(outfilename, 'w')
    write_prefix(outf)
    for key in jobj.keys():
        for pid in jobj[key].keys():
            if pid == 'P31':
                num_p31_skipped += 1
                continue
            if len(jobj[key][pid]) == 0:
                t = get_triple(s=key, p = pid, o = 'Q0')
                outf.write(t + '\n')
            else:
                for obj in jobj[key][pid]:
                    triple = get_triple(s=key, p=pid, o=obj)
                    outf.write(triple + '\n')

    print('num_p31_skipped ', num_p31_skipped)
    return None
                    
def read_comp_wikidata_rev(directory, out_dir):
    '''
    The outermost index indexes the ‘object entities’. The value of each comp_wikidata_rev[key] is a dict. with keys as pid’s (id’s corresponding to the relations/predicates). The value of each comp_wikidata_rev[key][pid] is a list of subject entities with outer key as the object entity.
    The (key, inverse(pid), subject entity) is a usual KB triple.
    '''
    global num_p31_skipped
    filename = os.path.join(directory, 'comp_wikidata_rev.json')
    outfilename = os.path.join(out_dir, 'comp_wikidata_rev.ttl')
    print('Loading file ', filename)
    jobj = json.load(open(filename, 'r'))
    print('Loaded file ', filename)
    outf = open(outfilename, 'w')
    write_prefix(outf)
    for key in jobj.keys():
        for pid in jobj[key].keys():
            if pid == 'P31':
                num_p31_skipped += 1
                continue
            if len(jobj[key][pid]) == 0:
                triple = get_triple(s='Q0', p=pid, o=key)
                outf.write(triple + '\n')
            else:
                for obj in jobj[key][pid]:
                    # reversed
                    triple = get_triple(s=obj, p=pid, o=key)
                    outf.write(triple + '\n')

    print('num_p31_skipped ', num_p31_skipped)
    return None

def read_par_child_file(directory, out_dir):
    count_so = {}
    count_r = {}
    filename = os.path.join(directory, 'par_child_dict.json')
    outfilename = os.path.join(out_dir, 'par_child_dict.ttl')
    #outfilename_p0 = os.path.join(out_dir, 'par_child_dict_p0.ttl')
    print('Loading file ', filename)
    jobj = json.load(open(filename, 'r'))
    print('Loaded file ', filename)
    outf = open(outfilename, 'w')
    #outf_p0 = open(outfilename_p0, 'w')
    write_prefix(outf)
    #write_prefix(outf_p0)
    keys = jobj.keys()
    for parent in tqdm(keys):
        childs = jobj[parent]
        for child in childs:
            #triple = get_triple(s=child, p='P31', o=parent)
            triple = get_triple(s=child, p='P31', o=parent)
            outf.write(triple + '\n')
            #outf_p0.write(triple_p0 + '\n')

    return count_so, count_r


def read_items_wikidata_n(directory, out_dir):
    filename = os.path.join(directory, 'items_wikidata_n.json')
    outfilename = os.path.join(out_dir, 'items_wikidata_n.ttl')
    print('Loading file ', filename)
    jobj = json.load(open(filename, 'r'))
    print('Loaded file ', filename)
    outf = open(outfilename, 'w')
    all_triples = []
    write_prefix(outf)
    for k, v in tqdm(jobj.items()):
        v = Literal(v).n3()
        triple = get_triple(s = k, p = 'rdfs:label', o = v, sl=WD, pl='', ol='')
        outf.write(triple + '\n')

#########################

def read_child_par_dict_immed(directory, out_dir):
    name = 'child_par_dict_immed'
    filename = os.path.join(directory, name + '.json')
    outfilename = os.path.join(out_dir, name + '.ttl')

    print('Loading file ', filename)
    jobj = json.load(open(filename, 'r'))
    print('Loaded file ', filename)
    outf = open(outfilename, 'w')
    write_prefix(outf)

    for key, obj in jobj.items():
        triple = get_triple(s=key, p='P31', o=obj)
        outf.write(triple + '\n')
    
    outf.close()

def read_child_par_dict_save(directory, out_dir):
    '''
    Example:  "Q26521211 - Broad Leaze Farmhouse": "Q3947 - house", 
    '''
    name = 'child_par_dict_save'
    filename = os.path.join(directory, name + '.json')
    outfilename = os.path.join(out_dir, name + '.ttl')

    print('Loading file ', filename)
    jobj = json.load(open(filename, 'r'))
    print('Loaded file ', filename)
    outf = open(outfilename, 'w')
    write_prefix(outf)

    for key, obj in jobj.items():
        triple = get_triple(s=key, p='P31', o=obj)
        outf.write(triple + '\n')
    
    outf.close()

def read_child_all_parents_till_5_levels(directory, out_dir):
    '''
    Example
    "Q15140120 - Lexus RC": [
        "Q3231690 - automobile model"
        ],
    '''
    name = 'child_all_parents_till_5_levels'
    filename = os.path.join(directory, name + '.json')
    outfilename = os.path.join(out_dir, name + '.ttl')
    print('Loading file ', filename)
    jobj = json.load(open(filename, 'r'))
    print('Loaded file ', filename)
    outf = open(outfilename, 'w')
    write_prefix(outf)

    for key in jobj.keys():
        for obj in jobj[key]:
            triple = get_triple(s=key, p='P31', o=obj)
            outf.write(triple + '\n')

    outf.close()

read_wikidata_short_1(directory, out_dir)
read_wikidata_short_2(directory, out_dir)
read_comp_wikidata_rev(directory, out_dir)
read_par_child_file(directory, out_dir)
read_items_wikidata_n(directory, out_dir)

#read_child_par_dict_immed(directory, out_dir)
#read_child_par_dict_save(directory, out_dir)
#read_child_all_parents_till_5_levels(directory, out_dir)

