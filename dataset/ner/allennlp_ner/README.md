### Tag based on allennlp NER and elastic search

- Start elastic_search server in localhost and port 9200

```
python createlist.py
python nel.py  \
   -data_path "data_path/" \
  -save_path "tag_data_ner" \
  -file_path "trainlist.txt" \
  -dataset 'train' \
  -start 0  \
  -end -1 
```