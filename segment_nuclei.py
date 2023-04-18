import numpy as np
import os
from itertools import repeat
import tifffile as ti

import cv2
from skimage.util import img_as_ubyte, img_as_uint
from scipy.ndimage import binary_fill_holes
from skimage import measure

from joblib import Parallel, delayed



def segment_nuclei_stack_canny(mCh_image: str, output_dir: str) -> None:
	"""
	Segment the nuclei out of a z-stack image using the Canny edge detection algorithm.
	
	Args:
		mCh_image (str): Path to the z-stack image.
		output_dir (str): Path to the directory where the segmented nuclei masks will be saved.
	
	Returns:
		None
	"""
	
	# Load the 3D volume image (stack) of nuclei from the given input image
	nuclei_stack = ti.imread(mCh_image)
	# Create an empty array of the same shape as the input image for storing the binary masks of segmented nuclei
	nuclei_mask_stack = np.zeros_like(nuclei_stack, dtype="uint8")

	# Perform nuclei segmentation on each plane in the stack
	for index, stack in enumerate(nuclei_stack):

		# Normalize the image intensities to [0,1]
		normalized_nuclei_image = np.divide(stack, np.max(stack))
		# Convert the normalized image to 16-bit unsigned integer format
		nuclei_image = img_as_uint(normalized_nuclei_image)

		# Apply Gaussian blur on the image
		blurred_nuclei_image = cv2.GaussianBlur(nuclei_image, (0,0), 3)
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
		# Store the mask in the output array
		nuclei_mask_stack[index, :, :] = good_nuclei_mask

	print(f'DONE ! {os.path.basename(mCh_image)}')
	# Save the mask
	ti.imwrite(os.path.join(output_dir, os.path.basename(mCh_image)), nuclei_mask_stack, compression='zlib')

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

if __name__ == '__main__':
	mCh_dir = "/mnt/external.data/TowbinLab/bgusev/nuclei_tracking/daf16_ch2_examples/"
	images_mCh = sorted([os.path.join(mCh_dir, x) for x in os.listdir(mCh_dir)])
	# images_mCh = list_images_of_point(images_mCh, "Point0006")

	output_dir = "/mnt/external.data/TowbinLab/bgusev/nuclei_tracking/daf16_ch2_examples_masks/"

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	Parallel(n_jobs=-1, prefer="processes")(delayed(segment_nuclei_stack_canny)(image, output_dir) for image in images_mCh)