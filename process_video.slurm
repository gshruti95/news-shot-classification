#!/bin/bash
#SBATCH -o /home/sxg755/logs/out.o%j
#SBATCH --time=03:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 4096
#NOTE : Only to be run from news-shot-classification directory

srun python ShotClass-01.py $1
rm $1

