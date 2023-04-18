#!/bin/bash
#SBATCH -J split_nuclei
#SBATCH -o split_nuclei.out
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 10:00:00
#SBATCH --mem=64G

source /home/spsalmon/env_directory/nuclei/bin/activate
python3 split_nuclei.py
deactivate