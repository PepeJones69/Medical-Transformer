#! /bin/bash
cd /work/scratch/schulz/Medical-Transformer

python=/work/scratch/schulz/miniconda3/envs/medt/bin/python

$python /work/scratch/schulz/Medical-Transformer/train.py --train_dataset data_3L/train --val_dataset data_3L/val --direc runs/MedT_3L_NoBackground --batch_size 4 --epoch 400 --save_freq 10 --modelname "MedT" --learning_rate 0.001 --imgsize 256 --gray "yes" --n_classes 4
