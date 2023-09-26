#! /bin/bash
cd /work/scratch/schulz/Medical-Transformer

python=/work/scratch/schulz/miniconda3/envs/medt/bin/python

$python /work/scratch/schulz/Medical-Transformer/test.py --loaddirec runs/MedT_Original/final_model.pth --val_dataset data_new_mov_avg --direc runs/mov_avg --batch_size 4 --modelname "MedT" --imgsize 256 --gray "yes"
