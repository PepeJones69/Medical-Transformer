#! /bin/bash
cd /work/scratch/schulz/Medical-Transformer

python=/work/scratch/schulz/miniconda3/envs/medt/bin/python

$python /work/scratch/schulz/Medical-Transformer/test.py --loaddirec runs/MedT_TopologyLoss/final_model.pth --val_dataset data/val --direc runs/MedT_TopologyLoss --batch_size 4 --modelname "MedT" --imgsize 256 --gray "yes"
