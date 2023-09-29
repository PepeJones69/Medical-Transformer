#! /bin/bash
cd /work/scratch/schulz/Medical-Transformer

python=/work/scratch/schulz/miniconda3/envs/medt/bin/python

$python /work/scratch/schulz/Medical-Transformer/train.py --train_dataset data_original/train --val_dataset data_original/val --direc runs/MedT_Original --batch_size 4 --epoch 400 --save_freq 10 --modelname "MedT" --learning_rate 0.001 --imgsize 256 --crop 256 --gray "yes"
