import numpy as np
import os
from itertools import repeat
from tifffile import imread, imwrite


import cv2
from skimage.util import img_as_ubyte, img_as_uint
from scipy.ndimage import binary_fill_holes
from skimage import measure

from joblib import Parallel, delayed

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D
from tqdm import tqdm
from csbdeep.utils import Path, normalize

import argparse


def segment_nuclei_plane_canny(img : np.ndarray) -> np.ndarray:
	"""
	Segment the nuclei out of a single plane image using the Canny edge detection algorithm.

	Args:
		img (np.ndarray): A single plane image as a numpy array.
	Returns:
		np.ndarray: The binary mask of the segmented nuclei as a numpy array.
	"""
	# Normalize the image intensities to [0,1]
	normalized_nuclei_image = np.divide(img, np.max(img))
	# Convert the normalized image to 16-bit unsigned integer format
	nuclei_image = img_as_uint(normalized_nuclei_image)

	# Apply Gaussian blur on the image
	blurred_nuclei_image = cv2.GaussianBlur(nuclei_image, (0,0), 1.5)
	# Detect edges using Canny edge detection
	nuclei_edges = cv2.Canny(img_as_ubyte(blurred_nuclei_image), 0.1*256, 0.2*256) 

	# Define a cross-shaped structuring element for morphological operations
	cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	# Perform morphological closing to fill gaps in the edges
	nuclei_edges = cv2.morphologyEx(nuclei_edges, cv2.MORPH_CLOSE, cross_kernel)

	# Because images were just translated to eliminate the shift, the small black border creates an edge we don't want, we thus have to remove it
	nuclei_edges[0:7, :] = 0

	# Fill the edges
	nuclei_mask = img_as_ubyte(binary_fill_holes(nuclei_edges))
	# Smooth the mask using median filtering
	nuclei_mask = cv2.medianBlur(nuclei_mask, 3)
	# Erode the binary mask a little bit
	nuclei_mask = cv2.morphologyEx(nuclei_edges, cv2.MORPH_ERODE, cross_kernel)
	

	# Label the connected components in the binary mask and calculate their properties
	labels = measure.label(nuclei_mask)
	props = measure.regionprops(labels, np.zeros_like(labels))

	# Keep only the labels (nuclei) that satisfy certain criteria, such as minimum area
	labels_to_keep = []
	for idx in range(0, labels.max()):
		label_i = props[idx].label
		# eccentricity = (props[idx]['eccentricity'])
		# solidity = (props[idx]['solidity'])
		area = (props[idx]['area'])
		# if solidity > 0.93 and eccentricity < 0.75 and area > 30:
		if area > 30:
			labels_to_keep.append(label_i)

	# Create a new binary mask containing only the nuclei with the selected labels
	good_nuclei_mask = np.isin(labels, labels_to_keep)

	return good_nuclei_mask

def segment_nuclei_plane_stardist(img: np.ndarray, model: StarDist2D) -> np.ndarray:
	"""
	Segment the nuclei out of a single plane image using a trained StarDist model.

	Args:
		img (np.ndarray): A single plane image as a numpy array.
		model (StarDist2D): The trained StarDist model.
	Returns:
		np.ndarray: The binary mask of the segmented nuclei as a numpy array.
	"""
	
	# Normalize the image
	img = normalize(img, 1, 99.8, axis=(0, 1))
	img = img / np.max(img)
	
	# Predict labels using the StarDist model
	labels, _ = model.predict_instances(img, verbose=False, show_tile_progress=False)

	# Convert the labels into a binary mask.
	nuclei_mask = (labels > 0).astype(int)
	return nuclei_mask

def segment_nuclei_canny(image_path: str, output_dir: str) -> None:
	"""
	Segment the nuclei out of an image (single plane or z-stack) using the Canny edge detection algorithm.
	
	Args:
		image_path (str): Path to the z-stack image.
		output_dir (str): Path to the directory where the segmented nuclei masks will be saved.
	
	Returns:
		None
	"""
	
	# Load the image.
	nuclei_image = imread(image_path)
	# Check if image is a z-stack or a single plane.
	if nuclei_image.ndim > 2:
	# Create an empty array of the same shape as the input image for storing the binary masks of segmented nuclei
		nuclei_mask_stack = np.zeros_like(nuclei_image, dtype="uint8")

	# Perform nuclei segmentation on each plane in the stack
		for index, plane in enumerate(nuclei_image):

			# Store the mask in the output array
			nuclei_mask_stack[index, :, :] = segment_nuclei_plane_canny(plane)

		print(f'DONE ! {os.path.basename(image_path)}')
		# Save the mask
		imwrite(os.path.join(output_dir, os.path.basename(image_path)), nuclei_mask_stack, compression='zlib')
	else:
		nuclei_mask = segment_nuclei_plane_canny(nuclei_image)
		imwrite(os.path.join(output_dir, os.path.basename(image_path)), nuclei_mask, compression='zlib')

def segment_nuclei_stardist(image_path:str, output_dir:str, model:StarDist2D) -> None:
	"""
	Segment the nuclei out of an image (single plane or z-stack) using a trained StarDist model.
	
	Args:
		image_path (str): Path to the z-stack image.
		output_dir (str): Path to the directory where the segmented nuclei masks will be saved.
		model (StarDist2D): The trained StarDist model.
	Returns:
		None
	"""
	# Load the image.
	nuclei_image = imread(image_path)
	# Check if image is a z-stack or a single plane.
	if nuclei_image.ndim > 2:
		# Create an empty array of the same shape as the input image for storing the binary masks of segmented nuclei
		nuclei_mask_stack = np.zeros_like(nuclei_image, dtype="uint8")
		# Perform nuclei segmentation on each plane in the stack
		for index, plane in enumerate(nuclei_image):
			nuclei_mask_stack[index, :, :] = segment_nuclei_plane_stardist(plane, model)
		print(f'DONE ! {os.path.basename(image_path)}')
		# Save the mask
		imwrite(os.path.join(output_dir, os.path.basename(image_path)), nuclei_mask_stack, compression='zlib')
	else:
		# Perform nuclei segmentation on the image (single plane)
		nuclei_mask = segment_nuclei_plane_stardist(nuclei_image, model)
		imwrite(os.path.join(output_dir, os.path.basename(image_path)), nuclei_mask, compression='zlib')

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


def get_args() -> argparse.Namespace:
	"""
	Parses the command-line arguments and returns them as a namespace object.

	Returns:
		argparse.Namespace: The namespace object containing the parsed arguments.
	"""
	# Create a parser and set the formatter class to ArgumentDefaultsHelpFormatter
	parser = argparse.ArgumentParser(description='Segment the nuclei out of images using Canny edge detection or StarDist.',
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Add the arguments to the parser
	parser.add_argument('--mCh_dir', type=str, required=True,
						help='Directory containing the z-stack images.')
	parser.add_argument('--output_dir', type=str, required=True,
						help='Directory where the segmented nuclei masks will be saved.')
	parser.add_argument('--method', type=str, choices=['canny', 'stardist'], required=True,
						help='Segmentation method to use.')
	parser.add_argument('--model_path', type=str,
						help='Path to the StarDist model directory (only required if using StarDist segmentation).')


	# Parse the arguments and return the resulting namespace object
	return parser.parse_args()


if __name__ == '__main__':
	
	args = get_args()
	images_mCh = sorted([os.path.join(args.mCh_dir, x) for x in os.listdir(args.mCh_dir)])
	images_mCh = list_images_of_point(images_mCh, 'Point0013')
	method = args.method
	output_dir = args.output_dir

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Run the segmentation using the Canny edge detection algorithm on the images
	if method == "canny":
		Parallel(n_jobs=-1, prefer="processes")(delayed(segment_nuclei_canny)(image, output_dir) for image in tqdm(images_mCh))
	# Run the segmentation using StarDist on the images
	elif method == "stardist":
		if args.model_path is None:
			raise ValueError("Please specify the path to the StarDist model directory using the --model_path argument.")
		model = StarDist2D(None, name=os.path.basename(os.path.normpath(args.model_path)), basedir=os.path.abspath(os.path.join(args.model_path, os.pardir)))
		print(model)
		for image in tqdm(images_mCh):
			segment_nuclei_stardist(image, output_dir, model)