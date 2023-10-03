#!/bin/bash
#SBATCH -J intensity
#SBATCH -o intensity.out
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 05:00:00
#SBATCH --mem=64G

source /home/spsalmon/env_directory/nuclei/bin/activate
python3 tracking_intensity.py
deactivate