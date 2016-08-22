#!/bin/bash
#SBATCH -o /home/sxg755/logs/out.o%j
#SBATCH --time=02:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 2048
#NOTE : Only to be run from news-shot-classification directory

srun python main.py $1
rm $1

