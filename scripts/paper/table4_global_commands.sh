#!/usr/bin/env bash

Help()
{
   echo "This scripts trains model, predicts on development set and runs evaluation"
   echo
   echo "Syntax: table4_global_commands.sh [options] config_folder config_file_name transfomer_model_name"
   echo "config_folder    folder with configuration files"
   echo "config_file_name    configuration filename (without jsonnet extension)"
   echo "transfomer_model_name    name of the transfomer language representation model to use"

   echo "options:"
   echo "-d    folder which contains the bert_train.json, bert_dev.json, bert_eval.json files [default: ./data]"
   echo "-m    folder where to store the trained model [default: models]"
   echo "-p    folder where to store the resulting predictions [default: output]"
   echo "-l    learning rate [default: 2e-5]"
   echo "-g    folder with tht gold standard FEVER dev reference file shared_task_dev.jsonl [default: ./gold_standard]"
   echo "-c    cuda device [default: 0]"
   echo "-h    print this help"
   echo
}

#set the paths below to your preferred locations
data_folder=./data  #where train/dev/test files with retrieved evidence are located
model_folder=models #where you want to store the models
prediction_folder=output #folder to which you want to output the predictions
gold_reference_dir=./gold_standard #folder which contains gold original FEVER files
gold_standard_path=${gold_reference_dir}/shared_task_dev.jsonl
cuda_device=0
learning_rate=2e-5

while getopts d:m:p:g:c:h:l: option
do
        case "${option}"
                in
                d) data_folder=${OPTARG};;
                m) model_folder=${OPTARG};;
                p) prediction_folder=${OPTARG};;
                g) gold_standard_path=${OPTARG};;
                c) cuda_device=${OPTARG};;
                l) learning_rate=${OPTARG};;
                h) Help; exit 0;;
                \?) echo "Wrong parameter ${OPTARG}"; Help; exit 0;;
        esac
done
shift $((OPTIND -1))

model_name=${3}
config_folder=${1}
config_file_name=${2}

export FVR_TRAIN_PATH=${data_folder}/bert_train.json
export FVR_VALID_PATH=${data_folder}/bert_dev.json
export TRANSFORMER_MODEL=${model_name}

#training

experiment_name=${config_file_name}_${model_name}_s42_lr${learning_rate}

echo "Storing model to ${model_folder}/${experiment_name}"
echo sh scripts/train.sh ${config_folder}/${config_file_name}.jsonnet \
 ${model_folder}/${experiment_name} 42 "trainer: {optimizer: {lr: ${learning_rate}}, cuda_device: ${cuda_device}}"
sh scripts/train.sh ${config_folder}/${config_file_name}.jsonnet \
 ${model_folder}/${experiment_name} 42 \
 "trainer: {optimizer: {lr: ${learning_rate}}, cuda_device: ${cuda_device}}"
#  "trainer: {optimizer: {lr: ${learning_rate}}, cuda_device: ${cuda_device}}, data_loader: {batches_per_epoch: null}, dataset_reader: {lines_to_read: 1000, max_sentences_to_keep: 3}"

#predicting
export CUDA_DEVICE=${cuda_device}
echo sh scripts/predict.sh ${data_folder}/bert_eval.json \
 ${model_folder}/${experiment_name} 64 ${prediction_folder}/${experiment_name}.json

sh scripts/predict.sh ${data_folder}/bert_eval.json \
 ${model_folder}/${experiment_name} 64 \
 ${prediction_folder}/${experiment_name}.json

#evaluating
python -m fever_scorer.generate_eval_table --gold_standard_fever ${gold_standard_path} \
 --allennlp_prediction_folder ${prediction_folder} \
 --allennlp_prediction_file_pattern "${experiment_name}"


