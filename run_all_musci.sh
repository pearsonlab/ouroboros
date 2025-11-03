#!/bin/bash

musciBirds=("blk445" "blk446" "blk447" "org666" "red453")

for bird in ${musciBirds[@]}; do 
    python /hdd/miles/ouroboros/run_muscimol_experiments.py \
    --audio_path="/home/miles/isilon/All_Staff/birds/mooney/muscimol/Microdialysis/Muscimol/$bird/" \
    --model_path="/home/miles/isilon/All_Staff/miles/models/ouroboros/musci_11_2_25_$bird/"
done