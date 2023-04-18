import numpy as np
import re
from math import floor
import os
import multiprocessing as mp
from itertools import repeat
import tifffile as ti
from math import floor, ceil

import btrack

from btrack import datasets

import matplotlib
from joblib import Parallel, delayed

import pandas as pd
from skimage import measure

import logging

matplotlib.rcParams['figure.figsize'] = [16, 16]
logging.basicConfig(level=logging.NOTSET)


def extract_split(s):
    """Extract the last occurence of "split_x-y" where x and y are two numbers in a string """
    pattern = r'split_\d+-\d+'
    matches = re.findall(pattern, s)
    if matches:
        return matches[-1]
    else:
        return None
    
def crop_to_rectangle(image):
	"""
	Crop the image into a square. This is sometimes useful, especially
	in methods employing FFT.
	"""
	dims = image.shape

	if dims[0] > dims[1]:
		diff = 0.5*(dims[0]-dims[1])
		image = image[int(floor(diff)): -int(ceil(diff)), :]
	elif dims[1] > dims[0]:
		diff = 0.5*(dims[1]-dims[0])
		image = image[:, int(floor(diff)): -int(ceil(diff))]
	return image


def focus_with_normalized_variance(image):
	mean = np.mean(image)
	var = np.var(image)
	return np.array([var/mean])


def get_square_bbox(mask, expand=10):
	# Creation of the bounding box.
	y, x = np.where(mask != 0)

	ymin = int(np.min(y))
	ymax = int(np.max(y))
	xmin = int(np.min(x))
	xmax = int(np.max(x))

	w = ymax-ymin
	h = xmax-xmin

	ymin -= expand
	xmin -= expand

	if ymin < 0:
		ymin = 0
	if xmin < 0:
		xmin = 0

	w += expand
	h += expand

	square_dim = max(w, h)

	# For some reason, the next steps only work with a image whose dimensions can be divided by four.
	while (square_dim % 4 != 0):
		square_dim += 1

	# If the bounding box would come out of the image, moove the bounding box's origin (xmin, ymin).
	if (xmin + square_dim) > mask.shape[1]:
		xmin = (xmin - ((xmin + square_dim) - mask.shape[1]))

	if (ymin + square_dim) > mask.shape[0]:
		ymin = (ymin - ((ymin + square_dim) - mask.shape[0]))

	return [ymin, xmin, square_dim]


def get_nuclei_intensity(inputs):
	idx, images_gfp, images_mCh, masks_mCh = inputs
	image_gfp = images_gfp[idx]
	mask_mCh = masks_mCh[idx]
	image_mCh = images_mCh[idx]

	logging.info(os.path.basename(image_gfp))

	gfp = ti.imread(image_gfp)
	mask = ti.imread(mask_mCh)
	mCh = ti.imread(image_mCh)

	if np.count_nonzero(mask) == 0 :
		return np.nan
	# Tracking the nuclei

	FEATURES = [
		"area",
		"major_axis_length",
		"minor_axis_length",
		"orientation",
		"solidity",
		"intensity_max",
		"intensity_mean",
		"intensity_min",
		"eccentricity"
	]

	objects = btrack.utils.segmentation_to_objects(
		mask, mCh, properties=tuple(FEATURES))

	# initialise a tracker session using a context manager
	with btrack.BayesianTracker() as tracker:

		# configure the tracker using a config file
		tracker.configure(datasets.cell_config())
		tracker.max_search_radius = 50
		tracker.tracking_updates = ["MOTION", "VISUAL"]
		tracker.features = FEATURES

		# append the objects to be tracked
		tracker.append(objects)

		# set the tracking volume
		tracker.volume = ((0, 2048), (0, 2048))

		# track them (in interactive mode)
		tracker.track(step_size=100)

		# generate hypotheses and run the global optimizer
		tracker.optimize()

		# store the tracks
		tracks = tracker.tracks

		# store the configuration
		cfg = tracker.configuration

	# Create a stack of tracked objects
	labels = measure.label(mask)
	tracked_objects = np.zeros_like(labels, dtype="uint8")

	for cell in tracks:
		cell_id = cell['ID']
		print(cell['ID'])
		print(cell['t'])
		for t in cell['t']:
			# print(f't = {t}')
			arg = np.argwhere(np.array(cell['t']) == t).squeeze()

			# print(int(cell['y'][arg]), int(cell['x'][arg]))
			cell_label = labels[t, int(cell['y'][arg]), int(cell['x'][arg])]

			if cell_label != 0:
				tracked_objects[t, :, :][labels[t, :, :]
										 == cell_label] = cell_id

	total_gfp = 0
	total_area = 0

	for obj in np.unique(tracked_objects)[1:]:
		obj_position = (tracked_objects == obj)
		list_focus = []
		for t, stack_track in enumerate(obj_position):
			if np.max(stack_track > 0):
				stack_mCh = mCh[t, :, :]
				stack_gfp = gfp[t, :, :]
				ymin, xmin, square_dim = get_square_bbox(stack_track, expand=8)
				stack_mCh = stack_mCh[ymin:ymin +
									  square_dim, xmin:xmin+square_dim]
				stack_gfp = stack_gfp[ymin:ymin +
									  square_dim, xmin:xmin+square_dim]
				stack_track = stack_track[ymin:ymin +
										  square_dim, xmin:xmin+square_dim]

				stack_focus = [
					t, float(focus_with_normalized_variance(stack_mCh).squeeze())]

				list_focus.append(stack_focus)

		sorted_list_focus = sorted(
			list_focus, key=lambda x: x[1], reverse=True)
		best_stack_for_object = sorted_list_focus[0][0]

		best_obj_position = obj_position[best_stack_for_object, :, :]
		best_gfp_for_object = gfp[best_stack_for_object, :, :]

		gfp_of_object = np.multiply(best_obj_position, best_gfp_for_object)
		area_object = np.sum(best_obj_position)

		if area_object > 85:
			total_gfp += np.sum(gfp_of_object)
			total_area += area_object

	try:
		result = total_gfp/total_area
		return result
	except ZeroDivisionError:
		return np.nan


def parallel_intensity(list_indexes, images_gfp, images_mCh, masks_mCh):
	logging.info(list_indexes)
	logging.info(len(images_gfp))
	logging.info(len(masks_mCh))
	return Parallel(n_jobs=10, prefer="processes")(delayed(get_nuclei_intensity)(inp) for inp in zip(list_indexes, repeat(images_gfp), repeat(images_mCh), repeat(masks_mCh)))


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
	gfp_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/ch1_WBT281/"
	mCh_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/ch2_WBT281/"
	mCh_mask_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/ch2_WBT281_mask_split_5/"
	
	output_dir = "./"
	if not os.path.exists(output_dir):
			os.makedirs(output_dir)

	images_gfp = sorted([os.path.join(gfp_dir, x)
						 for x in os.listdir(gfp_dir)])
	images_mCh = sorted([os.path.join(mCh_dir, x)
						 for x in os.listdir(mCh_dir)])
	
	images_mCh_mask = sorted([os.path.join(mCh_mask_dir, x)
							  for x in os.listdir(mCh_mask_dir)])

	unique_splits = np.unique([extract_split(x) for x in images_mCh_mask])

	for split in unique_splits:
		logging.info(split)

		output_csv = output_dir + f'nuclei_GFP_intensity_WBT281_{split}.csv'
		
		images_mCh_mask_split = sorted(list_images_of_point(images_mCh_mask, split))

		point_range = range(0, 23)

		header = ["Point"]
		max_timepoints = 0
		for point in point_range:
			point = str(point)

			while len(point) < 4:
				point = "0" + point
			point = "Point" + point
			logging.info(point)
			images_of_point_gfp = sorted(list_images_of_point(images_gfp, point))

			nb_timepoints = len(images_of_point_gfp)

			if nb_timepoints > max_timepoints:
				max_timepoints = nb_timepoints

		for time in range(max_timepoints):
			print(time)
			time = str(time)
			while len(time) < 5:
				time = "0" + time
			time = "Time" + time
			header.append(time)

		required_list_length = len(header)
		output_dataframe = pd.DataFrame(columns=header)

		for point in point_range:
			point = str(point)

			while len(point) < 4:
				point = "0" + point
			point = "Point" + point

			output_line = [point]
			logging.info(point)
			images_of_point_gfp = sorted(list_images_of_point(images_gfp, point))
			images_of_point_mCh = sorted(list_images_of_point(images_mCh, point))
			masks_of_point_mCh = sorted(
				list_images_of_point(images_mCh_mask_split, point))
			logging.info('Computing intensities')
			mean_gfp_in_nuclei = parallel_intensity(
				range(len(images_of_point_gfp)), images_of_point_gfp, images_of_point_mCh, masks_of_point_mCh)
			
			output_line.extend(mean_gfp_in_nuclei)

			while len(output_line) < required_list_length:
				output_line.append(np.nan)
			
			output_dataframe.loc[len(output_dataframe)] = output_line
			output_dataframe.to_csv(output_csv, index=False)