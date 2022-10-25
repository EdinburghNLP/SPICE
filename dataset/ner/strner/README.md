## Scripts for string match based ner and nel
Tested for python version 3.8.13 and pyahocorasick 1.4.4


To avoid encoding issues.
```
Redump json files: 
python redump_ascii_disamb_list.py
```

### Generate entity count file for disambibuation
```
#This will create a json with entity count
python unnormalized_entity_counts.py -data_path path
```

### Run automation creation and tagging
```
# create list of files to annotate
python createlist.py
bash str_tag.sh
```
