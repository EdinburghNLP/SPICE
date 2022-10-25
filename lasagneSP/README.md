This director contains lasagneSP model adapted from https://github.com/endrikacupaj/LASAGNE.


### Inverted index on Wikidata entities
For building an inverted index on wikidata entities we use [elastic](https://www.elastic.co/) search. Consider the script file [csqa_elasticse.py](scripts/csqa_elasticse.py) for doing so.

## BERT embeddings
Before training the framework, we need to create BERT embeddings for the knowledge graph (entity) types and relations. You can do that by running.
```
python scripts/bert_embeddings.py
```

## Train lasagneSP
```
python train.py --data_path /preprocessed_data
```

## Inference
Inference is performed per question-type.
```
python inference.py --question_type QTYPE --model_path experiments/snapshots/model_path.pth.tar --data_path /preprocessed_data
```
Where QTYPE is in ("Clarification" "Comparative Reasoning (All)" "Comparative Reasoning (Count) (All)" "Logical Reasoning (All)" "Quantitative Reasoning (All)" "Quantitative Reasoning (Count) (All)" "Simple Question (Coreferenced)" "Simple Question (Direct)" "Simple Question (Ellipsis)" "Verification (Boolean) (All)" "Simple Question (Coreferenced)" "Verification (Boolean) (All)")

## Evaluation
To eexcute and evalute the inferred files, run the following script in evaluation folder.
```
bash execute_all.sh
python summarise_results.py --file_path out_dir
```


## License
The repository is under [MIT License](LICENCE).

