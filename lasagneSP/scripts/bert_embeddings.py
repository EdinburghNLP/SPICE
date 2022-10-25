import re
import json
import torch
import flair
import sys
sys.path.append("/home/s1959796/debug_lasange/lasagne_sparql/lasagne-baseline")
from dataset import CSQADataset
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, BertEmbeddings, DocumentPoolEmbeddings

# import constants
from myconstants import *

# set device
torch.cuda.set_device(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flair.device = DEVICE

# load bert model
bert = DocumentPoolEmbeddings([BertEmbeddings('bert-base-uncased')])

# read nodes from dataset
graph_nodes = list(CSQADataset().get_vocabs()[GRAPH].stoi.keys())

print(len(graph_nodes))

# read entity and relation labels
#ROOT_PATH = '/home/s1959796/debug_lasange/LASAGNE-master'
id_entity = json.loads(open(f'{ROOT_PATH}/knowledge_graph/items_wikidata_n.json').read())
id_relation = json.loads(open(f'{ROOT_PATH}/knowledge_graph/filtered_property_wikidata4.json').read())

# create embeddings
na_node = Sentence(graph_nodes[0])
pad_node = Sentence(graph_nodes[1])
bert.embed(na_node)
bert.embed(pad_node)
node_embeddings = {
    graph_nodes[0]: na_node.embedding.detach().cpu().tolist(),
    graph_nodes[1]: pad_node.embedding.detach().cpu().tolist()
}
for node in graph_nodes[2:]:
    node_label = Sentence(id_entity[node] if node.startswith('Q') else id_relation[node])
    bert.embed(node_label)
    node_embeddings[node] = node_label.embedding.detach().cpu().tolist()

with open(f'{ROOT_PATH}/knowledge_graph/node_embeddings.json', 'w') as outfile:
    json.dump(node_embeddings, outfile, indent=4)
