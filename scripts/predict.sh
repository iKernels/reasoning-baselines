#!/usr/bin/env bash

data_file=${1}
model_dir=${2}
batch_size=${3}
output_file=${4}

if [ "$#" -lt 4 ]; then
  echo "Usage: predict.sh data_file model_dir batch_size output_file {overrides optional} {weights file
  optional}"
  exit 1
fi

overrides=""
if [ "$#" -gt 4 ]; then
    overrides="${5}"
fi

wgt_file=""
if [ "$#" -gt 5 ]; then
    wgt_file=" --weights-file ${6}"
fi


packages="--include-package ikernels_core.readers --include-package ikernels_core.models
--include-package ikernels_core.modules --include-package ikernels_core.evaluators"

model_file="${model_dir}/model.tar.gz"

params="--use-dataset-reader --batch-size ${batch_size} --silent --cuda-device ${CUDA_DEVICE}${wgt_file}"

echo allennlp predict ${packages} ${params} --output-file ${output_file} ${model_file} ${data_file} -o \"{${overrides}}\"

allennlp predict ${packages}  ${params} --output-file ${output_file} ${model_file} ${data_file} -o "{${overrides}}"

echo "Wrote to ${output_file}"
