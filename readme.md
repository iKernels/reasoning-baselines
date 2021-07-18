This repository contains the implementation of the baseline models for FEVER fact-checking described in the following paper:

Kateryna Tymoshenko and Alessandro Moschitti. (2021). *Strong and Light Baseline Models for Fact-Checking Joint Inference*. Findings of ACL.

# Table of contents
* [The task](#the-task)
* [Installation](#installation)
* [The input data](#the-input-data)
  * [Download the official FEVER corpus](#download-the-official-fever-corpus)
  * [Evidence reasoning input data](#evidence-reasoning-input-data)
  * [Input data format description](#input-data-format-description)
* [Running the pipeline](#running-the-pipeline)
  * [Training](#training)
    * [Training examples](#training-examples)
  * [Predicting](#predicting)
    * [Running prediction example](#running-prediction-example)
  * [Evaluating or generating input for the standard evaluator](#evaluating-or-generating-input-for-the-standard-evaluator)
    * [Evaluating output of a single experiment (fever_one_system_eval.py)](#evaluating-output-of-a-single-experiment-fever_one_system_evalpy)
      * [Evaluation example](#evaluation-example)
  * [Evaluating output of a batch of experiments (generate_eval_table.py)](#evaluating-output-of-a-batch-of-experiments-generate_eval_tablepy)
* [Reproducing lines 5-22 from Table 4 (Tymoshenko and Moschitti, 2021)](#reproducing-lines-5-22-from-table-4-tymoshenko-and-moschitti-2021)
  * [KGAT model](#kgat-model)
  * [Local Models](#local-models)
  * [Global models](#global-models)
* [References](#references)

# The task

In FEVER, given a claim, *C*, and a collection of approximately
five million Wikipedia pages, *W*, the task is to
predict whether *C* is supported (```SUPPORTS```) or refuted
(```REFUTES```) by *W*, or whether there is not enough
information (```NOT ENOUGH INFO```) in *W* to support or refute *C*.
If C is classified as ```SUPPORTS``` or ```REFUTES```, the respective evidence
should be provided.

The overall task is complex, as one needs to:
1. Retrieve the documents that contain the evidence **(document retrieval)**;
2. Select relevant evidence **(evidence selection)**;
3. Label the claim given the evidence **(evidence reasoning)**.

In our work, we focus only on the last step of **evidence reasoning**.  Formally, given a claim, *C*, and a list of top *K* evidence sentences, *(E_1;...;E_K)*, selected by the **evidence selection** component from the documents retrieved by the **document retrieval block**, our components predict the claim label
(```SUPPORTS```/```REFUTES```/```NOT ENOUGH INFO```).

The figure below illustrates how the FEVER pipeline is applied to a specific claim and which parts of it the models proposed in this repository correspond to:

![FEVER pipeline when applied to a specific claim](images/pipeline.png?raw=true)


For the full task description please refer to the dataset and shared task description papers:
* Thorne, J., Vlachos, A., Cocarascu, O., Christodoulopoulos, C., & Mittal, A. (2018). [The Fact Extraction and VERification (FEVER) Shared Task.](https://www.aclweb.org/anthology/W18-5501) Proceedings of the First Workshop on Fact Extraction and VERification (FEVER), 1–9.
* Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). [FEVER: a large-scale dataset for Fact Extraction and VERification.](https://doi.org/10.18653/v1/N18-1074). Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 809–819.



# Installation

Create a condas environment and install `huggingface`, `allennlp` and `pandas` within it.

```bash
git clone https://github.com/iKernels/reasoning-baselines.git
cd reasoning-baselines
conda create --name ikrnbsl python=3.6.6
conda activate ikrnbsl
python -m pip install -r requirements.txt
```

In the explanations below we will assume that the path to the ``reasoning-baselines`` folder is stored in the ```${base_fld}``` variable.

# The input data

## Download the official FEVER corpus
Download the original gold-standard FEVER reference data from the official [FEVER task web-site](https://fever.ai/resources.html): [train set (train.jsonl)](https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl), [development set (shared_task_dev.jsonl)](https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.json),  [test set (shared_task_test.jsonl)](https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl).

Below we will assume that you have downloaded the gold-standard reference files to the folder `${gold_reference_dir}`.

## Evidence reasoning input data
The models in this repo only predict the label of the claim given a set of evidence pieces retrieved by **document (DocIR)** and **evidence selection (ES)** engines.

We re-use the output of the **DocIR** and **ES** components by the authors of:
*Liu, Z., Xiong, C., & Sun, M. (2020). [Kernel Graph Attention Network for Fact Verification.](https://doi.org/10.18653/v1/2020.acl-main.655) In ACL.*

The original source code of the (Liu et al., 2020) pipeline is available at [https://github.com/thunlp/KernelGAT](https://github.com/thunlp/KernelGAT) and they made available their data and all the checkpoints as a [zip-archive](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT.zip).

In (Tymoshenko and Moschitti, 2021), we run our experiments on the (Liu et al., 2020)'s **evidence reasoning** training, development, validation and test data located in the `KernelGAT/data/` folder of the above archive called `bert_train.json`, `bert_dev.json`, `bert_eval.json` and `bert_test.json`, respectively.

In the explanations below we will assume that you have placed the above files into the ``${er_data}`` folder.

**Running Local experiments**. If you wish to run **Local** experiments (i.e. train/predict on separate ```(claim, evidence_i)``` pairs instead of ```(claim, evidence_1, ... evidence_K)``` tuples) you need to add evidence labels to the ``${er_data}`` files and store them in another folder. Store the path to this folder in the ```${er_local_data}``` variable.

You can do it as follows:
```bash
mkdir ${er_local_data}
python -m fever_scorer.add_labels_to_kgat_evidence --gold_standard_reference ${gold_reference_dir}/train.jsonl --input_file ${er_data}/bert_train.json --output_file ${er_local_data}/bert_train.json
python -m fever_scorer.add_labels_to_kgat_evidence --gold_standard_reference ${gold_reference_dir}/shared_task_dev.jsonl --input_file ${er_data}/bert_dev.json --output_file ${er_local_data}/bert_dev.json
python -m fever_scorer.add_labels_to_kgat_evidence --input_file ${er_data}/bert_eval.json --output_file ${er_local_data}/bert_eval.json
python -m fever_scorer.add_labels_to_kgat_evidence --input_file ${er_data}/bert_test.json --output_file ${er_local_data}/bert_test.json
```

### Input data format description
If you wish to use your own input data, please ensure that they are stored in a file where each line is a json record following the same format as the `${base_fld}/data` files:
```json
 { "id": <integer-claim-id>,  "evidence": [["<source-page-name>", <sentence-id>, "<sentence text>", <evidence-ranking-score>], ... [<source-page-name>, <sentence-id>, "<sentence text>", <evidence-ranking-score>]], "claim": "<claim text>", "label": "<claim label which can be SUPPORTS, REFUTES, NOT ENOUGH INFO>"}
 ```
 The evidence pieces in the `evidence` field should be sorted on their `<evidence-ranking-score>` in the decreasing order.

 For example:
 ```json
 {"id": 75397, "evidence": [ ["Nikolaj_Coster-Waldau", 7,  "He then played Detective John Amsterdam in the short lived Fox television series New Amsterdam LRB 2008 RRB ...", 1.0], ["Nikolaj_Coster-Waldau", 8, "He became widely known to a broad audience for his current role as Ser Jaime Lannister , in the HBO series Game of Thrones .", 0.1474965512752533], ["Nikolaj_Coster-Waldau", 9, "In 2017 , he became one of the highest paid actors on television and earned 2 million per episode of Game of Thrones .", -0.23199528455734253]], "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.", "label": "SUPPORTS"}
```

**Note:** to run the **Local** models you need to add the label of a specific evidence to each evidence record in training/dev/test files, so that it will be ```["<source-page-name>", <sentence-id>, "<sentence text>", <evidence-ranking-score>, "<evidence-label>"]``` instead of ```["<source-page-name>", <sentence-id>, "<sentence text>", <evidence-ranking-score>]```. The original `bert_train.json`, `bert_dev.json`, `bert_eval.json` and `bert_test.json` do not contain this information, but you can use the ```fever_scorer.add_labels_to_kgat_evidence``` script to convert them to the desired format (See the *Running Local experiments* in the [data](evidence-reasoning-input-data) section).

# Running the pipeline

## Training
Set the paths to the training and validation input files:
```bash
export FVR_TRAIN_PATH=<path_to_the_train_file> # path to bert_train.json
export FVR_VALID_PATH=<path_to_the_validation_file> # path to bert_dev.json
```
Specify which huggingface transformer model implementation you wish to use as an encoder by setting the `TRANSFORMER_MODEL` environment variable:
```bash
export TRANSFORMER_MODEL=roberta-base # or other huggingface transformer model. Please ensure that it is compatible with the configuration file of your choice (see information about configuration files below).
```

Use ```scripts/train.sh``` to launch training as follows:

```
sh scripts/train.sh ${config_file} ${model_dest_path} ${random_seed} ${overrides}
```
Above:
* ```${config_file}``` - is the standard allennlp experiment configuration ```jsonnet``` file. You may find configuration files for **Local**, **MaxPool**, **Concat** and **WgtSum** baselines in the `config/baselines` folder, and that for kgat in `config/kgat`.
  * Note: we have experimented running ```(concat|local|maxpool|wgt_sum).jsonnet``` with ```TRANSFORMER_MODEL``` set to ```roberta-base``` and ```bert-base-cased```; ```(concat|local|maxpool|wgt_sum)_local.jsonnet``` with ```TRANSFORMER_MODEL``` set to ```roberta-large```.
* ```${model_dest_path}``` - specifies where you want to store you model
* ```${random_seed}``` - random seed to use in your experiments
* ```${overrides}``` - overrides of the parameters set in the ```jsonnet``` file which will be passed as the ```-o``` parameter to the ```allennlp train``` command. Execute ```allennlp train --help``` to learn more about the ```-o/--overrides``` option.
  * For example, if you want to change the learning rate to `${lr}` without modifying the jsonnet file set the following overrides value: ```"trainer: {optimizer: {lr: ${lr}}}"```. Alternatively, you can simply create a new `jsonnet` file.


### Training examples
```bash
export TRANSFORMER_MODEL=roberta-base
sh scripts/train.sh config/baselines/concat.jsonnet models/fever/concat_roberta-base_s42 42
sh scripts/train.sh config/baselines/maxpool.jsonnet models/fever/maxpool_roberta-base_s42 42
sh scripts/train.sh config/baselines/wgt_sum.jsonnet models/fever/wgt_sum_roberta-base_s42 42
sh scripts/train.sh config/kgat/kgat.jsonnet models/kgat/wgt_sum_roberta-base_s42 42
```


## Predicting
Use `scripts/predict.sh` to run the prediction.
It generates the `allennlp predict` command and runs it.
Use the script as follows:

```bash
export CUDA_DEVICE=0 # set the cuda device; -1 for CPU
predict.sh <data_file> <model_dir> <batch_size> <output_file> <overrides (optional)> <weights file (optional)>
```
Above:
* `<data_file>` - input data file
* `<model_dir>` - folder containing the model pretrained with allennlp, `model.tar.gz`
* `<batch_size>` - batch size
* `<output_file>` - path to the output file
* `<overrides>` (optional) - overrides defined similarly to `train.sh`
* `<weights_file>` (optional) - path to the `.th` weights file if you want use the weights file other than those stored in `model.tar.gz`.

A line of the prediction file will correspond to one specific example and contain the following fields:
* `label_logits`: an array of logits corresponding to the "SUPPORTS", "REFUTES" and "NOT ENOUGH INFO" classes, respectively
* `probs`: an array of softmaxed logits corresponding to the "SUPPORTS", "REFUTES" and "NOT ENOUGH INFO" classes, respectively
* `qid`: claim ID
* `aid`: list of evidence ids that consist of their source Wikipedia page name and the sentence number concatenated with "_".
For example:
```json
{"label_logits": [0.9566327333450317, -1.7717119455337524, 1.126547932624817], "probs": [0.44433942437171936, 0.029027512297034264, 0.5266330242156982], "qid": "91198", "aid": ["Colin_Kaepernick_6", "Colin_Kaepernick_8", "Colin_Kaepernick_7", "Colin_Kaepernick_5", "Colin_Kaepernick_0"]}
```

### Running prediction example
To predict using the models we trained using the `train.sh` examples above and use CUDA run:
```
export CUDA_DEVICE=0 
eval_file=<path_to_the evaluation file> # path to bert_eval.json 
sh scripts/predict.sh ${eval_file} models/fever/concat_roberta-base_s42 64 output/fever/concat_roberta-base_s42.json
sh scripts/predict.sh ${eval_file} models/fever/maxpool_roberta-base_s42 64 output/fever/maxpool_roberta-base_s42.json
sh scripts/predict.sh ${eval_file} models/fever/wgt_sum_roberta-base_s42 64 output/fever/wgt_sum_roberta-base_s42.json
```

## Evaluating or generating input for the standard evaluator
To evaluate you need:
*  the official evaluation script from [https://github.com/sheffieldnlp/fever-scorer](https://github.com/sheffieldnlp/fever-scorer). You can download it by executing the following command:
```bash
    wget https://raw.githubusercontent.com/sheffieldnlp/fever-scorer/master/src/fever/scorer.py -O fever_scorer/scorer.py
```
* The original gold-standard FEVER reference data for the split you are evaluating on stored in `${gold_reference_dir}$` (See [here](#download-the-official-fever-corpus) how to download the data). Note that the test set is unlabeled and you can evaluate your predictions on test only by submitting your output in a specific format (not the output of `predict.sh`!) to [codalab](https://competitions.codalab.org/competitions/18814).

##### Evaluating output of a single experiment (`fever_one_system_eval.py`)
```bash
fever_one_system_eval.py [-h] [--gold_standard_fever GOLD_STANDARD_FEVER] [--allennlp_prediction_folder ALLENNLP_PREDICTION_FOLDER] [--allennlp_prediction_file ALLENNLP_PREDICTION_FILE] [--only_convert] [--logits_field LOGITS_FIELD] [--output_file OUTPUT_FILE]
```

Here:
* `GOLD_STANDARD_FEVER`: original gold-standard FEVER reference data file
* `ALLENNLP_PREDICTION_FOLDER`: folder containing the json file with the predictions produced by the `predict.sh` script
* `ALLENNLP_PREDICTION_FILE`: name of the json file containing the predictions produced by the `predict.sh` script
* `LOGITS_FIELD`: name of the json field in the `ALLENNLP_PREDICTION_FILE` which contains class logits predicted for an example. Default: `label_logits`.
*  `OUTPUT_FILE` (optional) file where to store the predictions in the format required by the official FEVER scorer. If you do not specify this option, nothing will be stored. If you need to generate the data to feed the official
*  `--only_convert`: the flag indicates that you only wish to convert the predictions in the `ALLENNLP_PREDICTION_FILE` to the official scorer format and do not need to compute the evaluation scores. Use this option (along with `--output_file`) when generating input for the official FEVER scorer (not our evaluation scripts). The properly formatted predictions will be written to the path indicated by `--output_file`.

###### Evaluation example
For example:
```bash
gold_standard_path=${gold_reference_dir}/shared_task_dev.jsonl
prediction_folder=output/fever
prediction_file=concat_roberta-base_s42.json 
python -m fever_scorer.fever_one_system_eval --gold_standard_fever ${gold_standard_path} --allennlp_prediction_folder ${prediction_folder} --allennlp_prediction_file ${prediction_file} --output_file output/fever_formatted/${prediction_file}
```
will produce:
```
FEVER score = 77.09
Label accuracy = 79.25
Evidence precision = 27.29
Evidence recall = 94.37
Evidence F1 = 42.34
```

Additionally, the `output/fever_formatted/concat_roberta-base_s42.json` file will contain the predictions in format required by the official FEVER evaluator and the CodaLab leaderboard.

##### Evaluating output of a batch of experiments (`generate_eval_table.py`)
The script generates a table with evaluation of outputs of multiple models.
```bash
generate_eval_table.py [-h] [--gold_standard_fever GOLD_STANDARD_FEVER] [--allennlp_prediction_folder ALLENNLP_PREDICTION_FOLDER] [--allennlp_prediction_file_pattern ALLENNLP_PREDICTION_FILE_PATTERN] [--logits_field LOGITS_FIELD]
```
Here:
* `GOLD_STANDARD_FEVER`: original gold-standard FEVER reference data file
* `ALLENNLP_PREDICTION_FOLDER`: folder containing the json file with the predictions produced by the `predict.sh` script
* `ALLENNLP_PREDICTION_FILE_PATTERN`: the script will evaluate on files in `ALLENNLP_PREDICTION_FOLDER` names of which match the regex pattern ALLENNLP_PREDICTION_FILE_PATTERN. If not specified, the script will evaluate on all the files in `ALLENNLP_PREDICTION_FOLDER`.
* `LOGITS_FIELD`: name of the json field in the `ALLENNLP_PREDICTION_FILE` which contains class logits predicted for an example. Default is `label_logits`.


For example:
```bash
gold_standard_path=${gold_reference_dir}/shared_task_dev.jsonl
prediction_folder=output/fever
python -m fever_scorer.generate_eval_table --gold_standard_fever ${gold_standard_path} --allennlp_prediction_folder ${prediction_folder} 
```

The output will look as follows:

| title |  FEVER |  LA  |  Ev P  |  Ev R  |  Ev F1 |
|---|---|---|---|---|---|
|concat_roberta-base_s42.json|77.09|79.25|27.29|94.37|42.34|
|kgat_roberta-base_s42.json|**77.66**|79.98|27.29|94.37|42.34|
|maxpool_roberta-base_s42.json|77.48|79.82|27.29|94.37|42.34|
|wgt_sum_roberta-base_s42.json|77.62|**80.01**|27.29|94.37|42.34|

Please note that your results will (insignificantly) differ from those in the table above.

# Reproducing lines 5-22 from Table 4 (Tymoshenko and Moschitti, 2021)
To reproduce lines 5-22 from Table 4 (Tymoshenko and Moschitti, 2021) you need to install the **reasoning-baseline** repository, download the necessary data and set the ``${er_data}``/``${er_local_data}`` variables as follows:
1. Install the `reasoning-baselines` repository following the [installation](#installation) instructions.
2. Download:
   * the official FEVER gold standard corpus (instructions [here](#download-the-official-fever-corpus)) and save it to the ```${base_fld}/gold``` folder;
   * the evidence reasoning step input retrieved by (Liu at al., 2020) (instructions [here](#evidence-reasoning-input-data)). Store its location in the ```${er_data}``` variable. If running the **Local** experiment, convert the evidence reasoning data as instructed [here](#evidence-reasoning-input-data) and store the converted data location in the ```{er_local_data}``` variable.
   * the official fever evaluation script (instructions [here](#evaluating-or-generating-input-for-the-standard-evaluator)).

The commands in the table below should reproduce the results from lines 5-22 of Table 4 in (Tymoshenko and Moschitti, 2021). Please note, that your results will insignificantly differ from those published in the paper and below.

By default, the commands will run on cuda device 0 with the learning rate of 2e-5. To change the cuda device or learning rate use flags ``-c`` and ``-l``, respectively.

Run ```sh scripts/paper/table4_global_commands.sh -h``` and ```sh scripts/paper/table4_local_commands.sh -h``` to see more options for running the global and local experiments, correspondingly.

## KGAT model
Note that the original KGAT software was made available by their authors in [https://github.com/thunlp/KernelGAT](https://github.com/thunlp/KernelGAT). If you wish to run their original software, please refer to the official **KernelGAT** repository.

The commands below launch the original KGAT model code integrated into the `reasoning-baselines` AllenNLP pipeline by us. More specifically, we took the original KGAT model code from the official repository and integrated it into the AllenNLP model interface. The original KGAT code is distributed under the MIT license (see [ikernels_core/models/kgat](https://github.com/iKernels/reasoning-baselines/tree/main/ikernels_core/models/kgat) for more details).

|Line|Learning rate|Fever|LA|LRM|Command|
|---|---|---|---|---|---|
|5:|lr=2e-5|74.87|77.15|bert-base-cased|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/kgat kgat bert-base-cased```|
|6:|lr=2e-5|77.66|79.98|roberta-base|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/kgat kgat roberta-base```|
|7:|lr=2e-5|78.66|80.77|roberta-large|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/kgat kgat_large roberta-large```|
|8:|lr=3e-5|75.28|77.48|bert-base-cased|```sh scripts/paper/table4_global_commands.sh -d ${er_data} -l 3e-5 config/kgat kgat bert-base-cased```|
|9:|lr=3e-5|77.75|80.06|roberta-base|```sh scripts/paper/table4_global_commands.sh -d ${er_data} -l 3e-5 config/kgat kgat roberta-base```|

## Local Models
|Line|Aggr. Heuristic|Fever|LA|LRM|Command|
|---|---|---|---|---|---|
|10:|Heuristic 1|73.05|75.11|bert-base-cased|```sh scripts/paper/table4_local_commands.sh -d ${er_local_data} config/baselines local bert-base-cased```|
|12:|Heuristic 2|71.79|73.66|bert-base-cased|Same as above. The script above will produce two outputs with both heuristics.|
|11:|Heuristic 1|75.62|77.85|roberta-base|```sh scripts/paper/table4_local_commands.sh -d ${er_local_data} config/baselines local roberta-base```|
|13:|Heuristic 2|73.98|75.96|roberta-base|Same as above. The script above will produce two outputs with both heuristics.|

## Global models
|Line|Model|Fever|LA|LRM|Command|
|---|---|---|---|---|---|
|14:|Concat|74.23|76.51|bert-base-cased|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/baselines concat bert-base-cased```|
|15:|Concat|77.09|79.25|roberta-base|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/baselines concat roberta-base```|
|16:|Concat|78.27|80.31|roberta-large|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/baselines concat_large roberta-large```|
|17:|MaxPool|74.72|76.99|bert-base-cased|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/baselines maxpool bert-base-cased```|
|18:|MaxPool|77.48|79.82|roberta-base|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/baselines maxpool roberta-base```|
|19:|MaxPool|78.85|81.16|roberta-large|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/baselines maxpool_large roberta-large```|
|20:|WgtSum|74.48|76.85|bert-base-cased|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/baselines wgt_sum bert-base-cased```|
|21:|WgtSum|77.62|80.01|roberta-base|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/baselines wgt_sum roberta-base```|
|22:|WgtSum|79.02|81.3|roberta-large|```sh scripts/paper/table4_global_commands.sh -d ${er_data} config/baselines wgt_sum_large roberta-large```|


# References
* (Liu et al, 2020) Liu, Z., Xiong, C., & Sun, M. (2020). [Kernel Graph Attention Network for Fact Verification.](https://doi.org/10.18653/v1/2020.acl-main.655) In ACL.
* (Tymoshenko and Moschitti, 2021) Kateryna Tymoshenko and Alessandro Moschitti. (2021). *Strong and Light Baseline Models for Fact-Checking Joint Inference*. Findings of ACL.
