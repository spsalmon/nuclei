import numpy as np
import os
import multiprocessing as mp
import skimage.io as skio
from itertools import repeat
import tifffile as ti


frames_folder = r"/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/ch2_WBT281_mask_split_frames_5/"
output_folder = r"/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/ch2_WBT281_mask_split_5/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def stitch_back(time_point, all_frames):

    frames = get_frames_of_timepoint(time_point, all_frames)
    frames.sort(key=sorting_key)

    max_x = skio.imread(frames[0]).shape[0]
    max_y = skio.imread(frames[0]).shape[1]
    for f in frames:
        frame_shape = skio.imread(f).shape
        if(frame_shape[0] > max_x):
            max_x = frame_shape[0]
        if(frame_shape[1] > max_y):
            max_y = frame_shape[1]


    video_shape = (len(frames), max_x, max_y)
    print(video_shape)
    video = np.zeros(video_shape, 'uint16')

    video_name = output_folder+'_'.join(os.path.basename(frames[0]).split('_')[0:-1])+".tiff"
    print(video_name)
    for i, f in enumerate(frames):
        print(f)
        print(i)
        frame = skio.imread(f).astype('uint16')
        video[i, 0:frame.shape[0], 0:frame.shape[1]] = frame
    ti.imwrite(video_name, video, compression="zlib")
            
        

def sorting_key(file):
    file = os.path.splitext(os.path.basename(file))[0]
    frame_number = file.split('_')[-1]
    return int(frame_number)

def get_frames_of_timepoint(time_point, frames):
    list_frames = [frame  for frame in frames if time_point in frame]
    return list_frames

def get_all_time_points(files):
    time_point_list = []
    for file in files:
        time_point = '_'.join(os.path.basename(file).split('_')[0:-1])
        if time_point not in time_point_list and (not time_point == "error_reports"):
            time_point_list.append(time_point)
    return time_point_list

if __name__ == '__main__':
    mp.freeze_support()
    frames = [os.path.join(frames_folder, x) for x in sorted(os.listdir(frames_folder))]
    print(get_all_time_points(frames))
    time_point_list = get_all_time_points(frames)

    # for tp in time_point_list:
    #     print(tp)
    #     print()
    pool = mp.Pool(32)    
    pool.starmap(stitch_back, zip(time_point_list, repeat(frames)))
    pool.close()