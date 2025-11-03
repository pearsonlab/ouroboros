#!/bin/bash

CAGBirds=("blk521" "org512" "org545" "pur567")
CaMKIIBirds=("grn573" "blk603")

for bird in ${CAGBirds[@]}; do
    python /hdd/miles/ouroboros/hyperparameter_cv.py --audio_path="/home/mmartinez/servers/pearson/birds/mooney/CAGBirds/$bird/data"\
    --audio_subdir="[0-9]*[0-9][0-9]/synchro_cleaned_v2"  --seg_subdir="[0-9]*[0-9][0-9]/denoised_segments" \
    --audio_filetype="_cleaned.wav" --seg_filetype="nrd.txt"\
    --model_path="/home/miles/isilon/All_Staff/miles/models/ouroboros/juveniles_11_3_25_$bird" --nEpochs=40 --seed=92\
    --voctype="juvieboros_$bird" --max_pairs=3000 --context_len=0.3 --n_kernels=10
done

for bird in ${CaMKIIBirds[@]}; do
    python /hdd/miles/ouroboros/hyperparameter_cv.py --audio_path="/home/mmartinez/servers/pearson/birds/mooney/CaMKIIBirds/$bird/data"\
    --audio_subdir="[0-9]*[0-9][0-9]/synchro_cleaned_v2"  --seg_subdir="[0-9]*[0-9][0-9]/denoised_segments" \
    --audio_filetype="_cleaned.wav" --seg_filetype="nrd.txt"\
    --model_path="/home/miles/isilon/All_Staff/miles/models/ouroboros/juveniles_11_3_25_$bird" --nEpochs=40 --seed=92\
    --voctype="juvieboros_$bird" --max_pairs=3000 --n_kernels=10 --nEpochs=100 --context_len=0.3
done