import numpy as np
import tifffile as ti
import skan
from skimage import morphology
import os
import cv2
import sys
import time
from joblib import Parallel, delayed

def get_nuclei_position_and_split(idx, number_of_splits, list_mask, list_bbox, list_nuclei, output_dir):
    """Uses the mask obtained from the GFP channel, the bounding box masks predicted with YOLO and the nuclei masks to split the different nuclei into thirds
    depending on their position in the worm's body."""

    # Opens the images corresponding to the given index
    mask = (ti.imread(list_mask[idx])>0)
    bbox = ti.imread(list_bbox[idx])
    nuclei = ti.imread(list_nuclei[idx])

    nuclei_basename = os.path.basename(list_nuclei[idx])
    print(f'{idx}, {nuclei_basename}')
    basename_split = nuclei_basename.split('_')
    name = basename_split[0:-2]

    savenames = []

    for i in range(number_of_splits):
        split_savename = name.copy()
        split_id = f'split_{i+1}-{number_of_splits}'
        split_savename.append(split_id)
        split_savename.append(basename_split[-1])
        split_savename = '_'.join(split_savename)
        savenames.append(split_savename)

    splits_stack = np.zeros((number_of_splits, nuclei.shape[0], nuclei.shape[1]), dtype='uint8')

    try:
        # To speed up the analysis, if the nuclei masks or the bounding box masks are empty, the rest of the analysis is skipped
        if np.max(nuclei) == 0:
            for i in range(number_of_splits):
                ti.imwrite(os.path.join(output_dir, savenames[i]), splits_stack[i], compression='zlib')
            return
        if np.max(bbox) == 0:
            for i in range(number_of_splits):
                ti.imwrite(os.path.join(output_dir, savenames[i]), splits_stack[i], compression='zlib')
            return
        
        # Computes the gfp mask's skeleton
        skeleton0 = morphology.skeletonize(mask)

        # Skeleton analysis using the Skan library
        skeleton_analysis = skan.csr.Skeleton(skeleton0, source_image=mask_dir)
        # Gets the longest path's index
        longest_path_index = np.argmax(skeleton_analysis.path_lengths())

        # Gets the coordinates of the first endpoint of the longest path
        endpoint0_coordinates = (skeleton_analysis.path_coordinates(longest_path_index)[0][0], skeleton_analysis.path_coordinates(longest_path_index)[0][1])

        # Gets the indexes of the two endpoints of the longest path
        endpoint0_index = np.sort(skeleton_analysis.path(longest_path_index))[0]
        endpoint1_index = np.sort(skeleton_analysis.path(longest_path_index))[-1]

        # Assigns the head index as depending on where endpoint0 is in the image
        if bbox[endpoint0_coordinates] == 7:
            head_index = endpoint0_index
        elif bbox[endpoint0_coordinates] == 3:
            head_index = endpoint1_index
        else:
            return
        
        # Initializes the list of nuclei centers
        nuclei_centers = []

        # Gets the length of the skeleton's longest path
        longest_path_graph = skeleton_analysis.graph[skeleton_analysis.path(longest_path_index), :]
        longest_path_length = longest_path_graph.sum()/2

        # Finds the contours of the nuclei mask, calculates the contours' moments and the nuclei centers
        contours = cv2.findContours(nuclei, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        for c in contours:
            M = cv2.moments(c)
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00']) 

            nuclei_centers.append([y,x])

        nuclei_centers = np.array(nuclei_centers)

        split_limits = list(np.linspace(0, 1, number_of_splits+1))

        splits_nuclei = []
        for i in range(number_of_splits):
            splits_nuclei.append([])

        # For each nuclei center 
        for center in nuclei_centers:
            # Computes the distance between the nuclei center and all the points of the skeleton
            distance_array = np.linalg.norm(skeleton_analysis.path_coordinates(longest_path_index) - center, axis=1)
            # Gets the index of the closest skeleton point to the nuclei center
            argmin = np.argmin(distance_array)

            # Gets the sub path graph, the path in the skeleton between the head and the point which is the closest to the nuclei center
            sub_path_graph = longest_path_graph[skeleton_analysis.path(longest_path_index)[min(argmin, head_index):max(argmin, head_index)], :]

            # Calculates the distance between the head and the point and normalizes it by the longest path's length
            normalized_distance_from_head = (sub_path_graph.sum()/2)/longest_path_length

            # Transposes the normalized distance to the interval containing normal nuclei positions (between 0.1 and 0.8)
            normalized_distance_from_head = (normalized_distance_from_head - 0.1)/(0.8 - 0.1)

            split_of_center = 0
            # Depending on the normalized distance, assigns the nuclei center to one of the splits
            if normalized_distance_from_head >= 0 and normalized_distance_from_head <= 1:
                while(normalized_distance_from_head) > split_limits[split_of_center+1]:
                    split_of_center += 1
                splits_nuclei[split_of_center].append(center)
        
                
        print(splits_nuclei)
        # # Gets the nuclei mask's connected components
        _, labels = cv2.connectedComponents(nuclei)

        splits_labels = []
        for i in range(number_of_splits):
            splits_labels.append([])

        for i, split in enumerate(splits_nuclei):
            for center in split:
                splits_labels[i].append(labels[center[0], center[1]])

        for i in range(number_of_splits):
            splits_stack[i] = np.isin(labels, splits_labels[i]).astype("uint8")

        for i in range(number_of_splits):
            ti.imwrite(os.path.join(output_dir, savenames[i]), splits_stack[i], compression='zlib')
        return
    except:

        for i in range(number_of_splits):
            ti.imwrite(os.path.join(output_dir, savenames[i]), splits_stack[i], compression='zlib')
        return


gfp_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/ch1_WBT281_frames/"
mask_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/ch1_WBT281_seg_frames/"
bbox_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/yolo_masks/"
nuclei_mask_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/ch2_WBT281_mask_cleaned_frames/"
output_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/ch2_WBT281_mask_split_frames_5/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

list_gfp = sorted([os.path.join(gfp_dir, x) for x in os.listdir(gfp_dir)])
list_mask = sorted([os.path.join(mask_dir, x) for x in os.listdir(mask_dir)])
list_bbox = sorted([os.path.join(bbox_dir, x) for x in os.listdir(bbox_dir)])
list_nuclei = sorted([os.path.join(nuclei_mask_dir, x) for x in os.listdir(nuclei_mask_dir)])

indexes = list(range(len(list_gfp)))

Parallel(n_jobs=32, prefer="threads")(delayed(get_nuclei_position_and_split)(idx, 5, list_mask, list_bbox, list_nuclei, output_dir) for idx in indexes)



