
# Evaluation

## Run evaluation script on predicted Sparqls (i.e., models' outputs)

The following script runs per question type. It will print a table with averaged results per question type and sub-type.
```--file_path``` indicates the json file with models' outputs. If the file name contains a "*" combined with 
```--read_all```, all matching files will be read and combined for evaluation (e.g., we can read models' outputs saved 
in several files such as *baseline.0-9.100000.test_Simple Question (Direct).json*, 
*baseline.10-19.100000.test_Simple Question (Direct).json*, etc.). See format of models' predictions files below.

```commandline
python run_subtype_lf.py \
    --file_path "results/${MODEL_NAME}/baseline.*.${CHECKPOINT}.${SPLIT}_Simple Question (Direct).json" \
    --read_all \
    --question_type "Simple Question (Direct)" \
    --em_only \
    --out_eval_file ${PATH_AND_NAME}
```

```--out_eval_file``` gives the name of the file where evaluation results details will be saved. This is a dictionary
where entries are the evaluated aspects (i.e., question type, question sub-types, and linguistic phenomena -- Ctx=-1, 
Ctx<-1, ellipsis, and multiple entities--). For each entry, the different computed metrics will be stored (e.g., exact 
match, f1-score, etc.).

When using the flag ```--context_dist_file``` it will also aggregate metrics per 'context distance'. 
This can be used for question types involving Coreference. If specified, it will aggregate scores for all coreference
questions where the antecedent is at distance 1, distance 2, and so on.
For each conversation turn, we have precomputed a file that contains the distance information used for this flag,
namelly, *CSQA_v9_skg.v7_context_distance_test.log*. Each line of the file is of the form:
```
test#QA_81#QA_16#2      2       Which french administrative division was that person born at ?
```
The first column indicates the turn and conversation identifier (see next section). The second, the distance in the 
conversation where the referent of the question is mentioned (here 2 means that the referent is in 2 turns back, that 
is introduced in turn 0). Finally, the user question.

#### Expected input json file containing models' predictions

The format is the following. See below descriptions of the fields.
```
[
    {
        "question_type": "Simple Question (Direct)",
        "description": "Simple Question|Single Entity",
        "question": "What is the official language of Paraguay ?",
        "answer": "Spanish",
        "actions": "SELECT ?x WHERE { wd: Q34770 wdt: P37 ?x . ?x wdt: P31 wd: Q34770 . }",
        "results": [
            "Q1321"
        ],
        "sparql_delex": "SELECT ?x WHERE { wd: Q733 wdt: P37 ?x . ?x wdt: P31 wd: Q34770 . }",
        "turnID": "test#QA_282#QA_72#3"
    },
    
...

]
```

* question_type: the question type
* description: the question sub-type
* question: user utterance
* answer: system utterance
* actions: predicted Sparql
* results: GOLD results/answer
* sparql_delex: GOLD Sparql
* turnID: Identifier of the conversation (e.g., ```test#QA_282#QA_72#3``` means turn position 3 in conversation 
of file test/QA_282/QA_72.json). Note that turn positions start from 0.

#### Summarising evaluation results 

The following script reads the evaluation details generated for each question type generated as described above (i.e., 
run_subtype_lf.py) and generates a final summary aggregation overall aspects across all questions. (Currently summarises linguistic phenomena).
```--file_path``` is the path to the folder that contains all .json files generated for each question with
*run_subtype_lf.py*.

```commandline
python summarise_results.py --file_path ${PATH_AND_NAME}
```
