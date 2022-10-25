# SPICE dataset

Dataset description


# Annotations on the SPICE dataset

## Entity Neighborhood Sub-Graphs

Input is a SPICE dataset, this script will extract entity neighborhood sub-graphs for gold entities. Will output a SPICE dataset copy annotated with entity neighborhood sub-graphs (added json field ```'local_subgraph'```, each local_subgraph for each turn is constructed based on the entities in the previous question, previous answer and current question).

``` bash
python precompute_local_subgraphs.py \
    --partition train  \
    --read_folder ${SPICE_CONVERSATIONS} \
    --write_folder ${ANNOTATED_SPICE_CONVERSATIONS} \
    --json_kg_folder ${PATH_JSON_KG}  
```

Once annotations are done for gold entities, it's possible to add entity neighborhood sub-graphs for NER/NEL entities (e.g., AllenNLP). For this you need to specify the ```--nel_entities``` flag and ```--allennlpNER_folder``` that contains the conversations annotated with AllenNLP NER/NEL (see instructions for this script below).

This script also generates the global vocabulary file, it will generate a file named ```expansion_vocab.json``` in folder ```ANNOTATED_SPICE_CONVERSATIONS```.
``` bash
python precompute_local_subgraphs.py --write_folder ${SPICE_CONVERSATIONS} --task vocab
```



## Type Sub-Graphs

Input is a SPICE dataset, will find KG type candidates mentioned in utterances, link to types in the KG and extract a set of relations for each of them. Will output a SPICE dataset copy annotated with type sub-graphs (the added json field is ```'type_subgraph'```).

``` bash
python precompute_local_types.py \
    --partition train \
    --read_folder ${SPICE_CONVERSATIONS} \
    --write_folder ${ANNOTATED_SPICE_CONVERSATIONS} \
    --json_kg_folder ${PATH_JSON_KG}  
```


## AllenNLP -based NER

Allennlp based NER-NEL scripts are present [here](./ner/allennlp_ner)

## String Match -based NER

String based NER-NEL script are present [here](./ner/strner)


