#!/bin/bash
#SBATCH -J split_stacks
#SBATCH -o split_stacks.out
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 10:00:00
#SBATCH --mem=64G

source /home/spsalmon/env_directory/nuclei/bin/activate
python3 split_tiff_stacks.py
deactivate