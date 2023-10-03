#!/bin/bash
#SBATCH -J stitch
#SBATCH -o stitch.out
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 10:00:00
#SBATCH --mem=64G
source ~/env_directory/nuclei/bin/activate
python3 stitch_back_stacks.py
deactivate