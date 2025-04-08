#!/bin/bash

CAGBirds=("blk521" "org512" "org545" "pur567")
CaMKIIBirds=("grn573" "blk603")

for bird in ${CAGBirds[@]}; do
    python /hdd/miles/ouroboros/kernel_comparison.py --audio_path="/home/miles/isilon/All_Staff/birds/mooney/CAGBirds/$bird/data"\
    --audio_filetype="[0-9]*[0-9][0-9]/denoised/*.wav" --seg_filetype="[0-9]*[0-9][0-9]/denoised_segments/*.txt"\
    --model_path="/home/miles/isilon/All_Staff/miles/models/ouroboros/kernelboros_train_all" --nEpochs=40 --seed=92\
    --voctype="juvieboros_$bird" --max_pairs=2000 --expand_factor=4 --n_layers=2 --mult_factor_kernels=10
done

for bird in ${CaMKIIBirds[@]}; do
    python /hdd/miles/ouroboros/kernel_comparison.py --audio_path="/home/miles/isilon/All_Staff/birds/mooney/CaMKIIBirds/$bird/data"\
    --audio_filetype="[0-9]*[0-9][0-9]/denoised/*.wav" --seg_filetype="[0-9]*[0-9][0-9]/denoised_segments/*.txt"\
    --model_path="/home/miles/isilon/All_Staff/miles/models/ouroboros/kernelboros_train_all" --nEpochs=40 --seed=92\
    --voctype="juvieboros_$bird" --max_pairs=2000 --expand_factor=4 --n_layers=2 --mult_factor_kernels=10
done