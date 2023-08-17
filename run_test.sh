#! /bin/bash
cd /work/scratch/schulz/Medical-Transformer

python=/work/scratch/schulz/miniconda3/envs/medt/bin/python

$python /work/scratch/schulz/Medical-Transformer/test.py --loaddirec resultsfinal_model.pth --val_dataset data/val --direc results --batch_size 4 --modelname "gatedaxialunet" --imgsize 256 --gray "yes"
