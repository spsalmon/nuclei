from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
# matplotlib.rcParams["image.interpolation"] = None


from glob import glob
from tifffile import imread, imwrite
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D
import os
from tqdm import tqdm

# from joblib import Parallel, delayed

np.random.seed(6)
lbl_cmap = random_label_cmap()

def list_images_of_point(images, point):
	"""
	Given a list of images and a point, returns a list of image names that contain the point in their file names.
	
	Parameters:
		images (list): List of image file names.
		point (str): Point to search for in image file names.
	
	Returns:
		list: List of image names that contain the point in their file names.
	"""
	
	# Initialize empty list to store image names
	image_names = []
	
	# Iterate through list of images
	for image in images:
		# Check if point is in the image file name
		if point in os.path.basename(image):
			# If point is found, append image name to list
			image_names.append(image)
	
	# Return list of image names
	return image_names

def segment_nuclei_stardist(image_path:str) -> None:
	nuclei_image = imread(image_path)
	if nuclei_image.ndim > 2:
		# Create an empty array of the same shape as the input image for storing the binary masks of segmented nuclei
		nuclei_mask_stack = np.zeros_like(nuclei_image, dtype="uint8")
		# Perform nuclei segmentation on each plane in the stack
		for index, plane in enumerate(nuclei_image):
			img = normalize(plane, 1,99.8, axis=axis_norm)
			img = img/np.max(img)
			labels, _ = model.predict_instances(img, verbose = False, show_tile_progress=False)

			# Store the mask in the output array
			nuclei_mask_stack[index, :, :] = (labels > 0).astype(int)

		print(f'DONE ! {os.path.basename(image_path)}')
		# Save the mask
		imwrite(os.path.join(output_dir, os.path.basename(image_path)), nuclei_mask_stack, compression='zlib')
	else:
		img = normalize(nuclei_image, 1,99.8, axis=axis_norm)
		img = img/np.max(img)
		labels, _ = model.predict_instances(img, verbose = False, show_tile_progress=False)
		# Save the mask
		imwrite(os.path.join(output_dir, os.path.basename(image_path)), labels, compression='zlib')

input_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/ch2_WBT281/"
output_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/ch2_WBT281_mask_stardist/"

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

images_path = sorted([os.path.join(input_dir, x) for x in os.listdir(input_dir)])
images_path = list_images_of_point(images_path, point="Point0017")

X = list(map(imread,images_path))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
	print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

model = StarDist2D(None, name='stardist_200_bigger_patch', basedir='models')

for image_path in tqdm(images_path):
	segment_nuclei_stardist(image_path)



