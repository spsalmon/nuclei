import os
from skimage.measure import label
import numpy as np
import tifffile as ti
from skimage.util import img_as_ubyte

# Define the input and output directories
input_dir = "/mnt/external.data/TowbinLab/spsalmon/nuclei_segmentation_stardist_training/database_200/annotations/"
output_dir = "/mnt/external.data/TowbinLab/spsalmon/nuclei_segmentation_stardist_training/database_200/labels/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all the binary masks in the input directory
for filename in os.listdir(input_dir):
    # if os.path.exists(os.path.join(output_dir, filename)):
    #     os.remove(os.path.join(input_dir, filename))
    # Load the binary mask as a numpy array
    if filename.endswith(".tiff"):
        binary_mask = ti.imread(os.path.join(input_dir, filename))
        
        labels = img_as_ubyte(label(binary_mask).astype(int))
        # Save the labeled image with the same filename, but with the -labels.tiff replaced with .tiff
        output_filename = filename.replace("-labels.tiff", ".tiff")
        output_path = os.path.join(output_dir, output_filename)
        ti.imwrite(output_path, binary_mask, compression="zlib")