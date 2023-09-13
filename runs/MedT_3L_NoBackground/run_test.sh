#! /bin/bash
cd /work/scratch/schulz/Medical-Transformer

python=/work/scratch/schulz/miniconda3/envs/medt/bin/python

$python /work/scratch/schulz/Medical-Transformer/test.py --loaddirec runs/MedT_3L_NoBackground/final_model.pth --val_dataset data_3L/val --direc runs/MedT_3L_NoBackground --batch_size 4 --modelname "MedT" --imgsize 256 --gray "yes" --n_classes 4
