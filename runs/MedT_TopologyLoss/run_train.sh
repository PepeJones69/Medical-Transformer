#! /bin/bash
cd /work/scratch/schulz/Medical-Transformer

python=/work/scratch/schulz/miniconda3/envs/medt/bin/python

$python /work/scratch/schulz/Medical-Transformer/train.py --train_dataset data/train --val_dataset data/val --direc results --batch_size 4 --epoch 400 --save_freq 10 --modelname "MedT" --learning_rate 0.001 --imgsize 256 --gray "yes" --loss "LogNLL_Topology"
