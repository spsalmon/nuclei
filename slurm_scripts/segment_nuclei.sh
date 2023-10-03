#!/bin/bash
#SBATCH -J nuclei_seg
#SBATCH -o nuclei_seg.out
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# TO BE CHANGED TO YOUR INPUT AND OUTPUT DIRECTORIES

MCH_DIR="/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/ch2_WBT281/"
OUTPUT_DIR="/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/ch2_WBT281_masks_stardist_boris/"

# CHOOSE THE METHOD YOU WANT TO USE AND SPECIFY THE PATH TO YOUR STARDIST MODEL IF NEEDED

METHOD="stardist"
PATH_TO_MODEL="/home/spsalmon/ASTRO_CLEAN_NUCLEI/stardist/models/stardist_200_bigger_patch/"

source /home/spsalmon/env_directory/nuclei/bin/activate
python3 ../src/segmentation/segment_nuclei.py --mCh_dir "$MCH_DIR" --output_dir "$OUTPUT_DIR" --method "$METHOD" --model_path "$PATH_TO_MODEL"
deactivate