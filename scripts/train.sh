#!/usr/bin/env bash

config_file=${1}
model_dir=${2}
current_seed=${3}

if [ "$#" -lt 3 ]; then
  echo "Usage: train.sh config_file model_dir seed [optional: additional-config-file-overrides]"
  exit 1
fi

packages="--include-package ikernels_core.readers --include-package ikernels_core.models
--include-package ikernels_core.modules --include-package ikernels_core.evaluators"

overrides="{random_seed: ${current_seed}70,  numpy_seed: ${current_seed}7, pytorch_seed: ${current_seed}}"

if [ "$#" -eq 4 ]; then
    overrides="{random_seed: ${current_seed}70,  numpy_seed: ${current_seed}7, pytorch_seed: ${current_seed}, ${4}}"
fi

#echo "Writing to ${model_dir}"
#if [ -d "${model_dir}" ]; then
#    echo "removing ${model_dir}"
#    rm -r ${model_dir};
#fi

echo allennlp train -f ${config_file} --serialization-dir ${model_dir} ${packages} -o "${overrides}" -f
allennlp train -f ${config_file} --serialization-dir ${model_dir} ${packages} -o "${overrides}" -f