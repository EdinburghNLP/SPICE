# BertSP Baseline

Here you will find the commands that are needed to prepare the data, train, validate and do inference with BertSP model.

## Preprocess Data
You need to run the following two steps.

### Prepare data to be used by the BertSP baseline

Below an example command to prepare the SPICE dataset with the data (and format) as required by the BertSP model.
This will produce a new set of .json files.

```commandline
python src/preprocess.py \
  -mode format_to_lines \
  -save_path ${JSON_PATH_DATA} \
  -n_cpus 20 \
  -use_bert_basic_tokenizer false \
  -data_path ${DATA_PATH} \
  -tgt_dict ${TGT_DICT} \
  -shard_size 5000 \
  -dataset ${DATASET} \
  -log_file ${HOME}/${LOGFILE} \
  -kb_graph ${HOME}'/knowledge_graph' \
  -nentities 'strnel' \
  -types 'linked'
```

For the ```-nentities``` flag there are four possible modes: *gold* (it will use all available gold entity identifier 
annotations), *lgnel* (will use gold entity identifier whose names can be found in the user utterances), *allennel* 
(will use entity identifiers that were linked through AllenNLP NER + ElasticSearch, this requires the input data, 
```-data_path```, to be annotated accordingly, see script for this annotation [here](../dataset)), and *strnel* 
(will use entity identifiers that were linked through String Match with KG symbols, this requires the input data, 
```-data_path```, to be annotated accordingly, see script for this annotation [here](../dataset))

For the ```-types``` flag possible values are: *gold* (will use gold type annotations) and *linked* will use
types that were linked through String Match with KG symbols).

#### Prepare data for generalisation splits

To prepare data for the generalisation spplits instead of the original train/valid/test, you need to add to the 
*preprocess.py* command above the flag ```-mapsplits``` and the file that contains the conversation IDs that
goes into each of the generalisation splits, i.e., flag ```-mapfile```.

You can find more about the files for the generalisation splits in the dataset description folder [here](../dataset).

### Generate binary files

Takes the data prepared in the previous step and generates binary .pt files.

```commandline
python src/preprocess.py \
  -mode format_to_bert \
  -raw_path ${JSON_PATH} \
  -save_path ${BERT_DATA_PATH}  \
  -data_path ${DATA_PATH} \
  -tgt_dict ${TGT_DICT} \
  -lower \
  -n_cpus 20 \
  -dataset ${DATASET} \
  -log_file ${HOME}/${LOGFILE}
```

## Train BertSP

Run the following command to train BertSP. This will train for 100k steps and save checkpoints.

```commandline
python src/train.py  \
  -mode train \
  -tgt_dict ${TGT_DICT} \
  -bert_data_path ${BERT_DATA_PATH} \
  -dec_dropout 0.2  \
  -sep_optim true \
  -lr_bert 0.00002 \
  -lr_dec 0.001 \
  -save_checkpoint_steps 2000 \
  -batch_size 1 \
  -train_steps 100000 \
  -report_every 50 \
  -accum_count 5 \
  -use_bert_emb true \
  -use_interval true \
  -warmup_steps_bert 20000 \
  -warmup_steps_dec 10000 \
  -max_pos 512 \
  -max_length 512  \
  -min_length 10  \
  -beam_size 1 \
  -alpha 0.95 \
  -visible_gpus 0,1,2,3 \
  -label_smoothing 0 \
  -model_path ${MODEL_PATH} \
  -log_file ${LOG_PATH}/${LOGFILE}
```

### Validate Checkpoints

Run the following command for checkpoint selection. The command runs one checkpoint at a time.

```commandline
python src/train.py \
    -mode validate \
    -valid_from ${MODEL_PATH}/${CHECKPOINT} \
    -batch_size 10 \
    -test_batch_size 10 \
    -tgt_dict ${TGT_DICT} \
    -bert_data_path ${BERT_DATA_PATH} \
    -log_file logs/base_bert_sparql_csqa_val  \
    -model_path ${MODEL_PATH}  \
    -sep_optim true  \
    -use_interval true \
    -visible_gpus 1  \
    -max_pos 512  \
    -max_length 512  \
    -min_length 20  \
    -test_split ${SPLIT} \
    -log_file $logf"/stats.${STEP}.log"
```

## Run Inference with BertSP

The following script will generate a .json file with Sparql predictions (parse) for each user utterance.
The format of this file is according to the format required by the evaluation scripts [here](../evaluation).

```commandline
python src/train.py \
    -mode test \
    -test_from ${MODEL_PATH}/${CHECKPOINT} \
    -batch_size 1 \
    -test_batch_size 1 \
    -tgt_dict ${TGT_DICT} \
    -bert_data_path ${BERT_DATA_PATH} \
    -log_file logs/base_bert_sparql_csqa_val  \
    -model_path ${MODEL_PATH}  \
    -test_split test \
    -sep_optim true  \
    -use_interval true \
    -visible_gpus 1  \
    -max_pos 512  \
    -max_length 512  \
    -min_length 10  \
    -alpha 0.95 \
    -beam_size 5 \
    -dosubset ${TEST_VALID_SUBSET} \
    -result_path results/${MODEL_NAME}/baseline.${TEST_VALID_SUBSET_STR}
```

Note that the flag ```-dosubset``` is used run inference on a subset of files from the test split.
The pre-processing (discussed at the beginning of this README) will shard conversations from each split into .json/.pt
files (e.g., *json_data.valid.41.json*). The ```-dosubset``` flag allows to give a regular expression to specify
the a range of shard IDs to run inference on, e.g., for our configuration of shards (0 to 49) we can use the following
to do inference on shards with IDs starting with *4*.
```
TEST_VALID_SUBSET='4[0-9]+'
TEST_VALID_SUBSET_STR='40-49'
```
Predictions will be saved in .json with the shard IDs, e.g. *baseline.40-49.96000.test_Logical Reasoning (All).json*
If the option ```-dosubset``` is not used only a single file containing predictions for all shards will be created, e.g.,
*baseline.96000.test_Logical Reasoning (All).json*.

## Evaluation

For evaluation of generated outputs see the evaluation scripts [here](../evaluation).