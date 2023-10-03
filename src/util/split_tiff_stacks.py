import tifffile as ti
import os
from joblib import Parallel, delayed
import cv2
from skimage.util import img_as_ubyte, img_as_uint
import numpy as np

def split_tiff_stack(stack_path, output_dir):
    stack = ti.imread(stack_path)
    stack_basename = os.path.basename(os.path.splitext(stack_path)[0])
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8,8))
    for idx, img in enumerate(stack):

        # img_norm = np.divide(img, np.max(img))
        # img = img_as_ubyte(img_norm)
        output_path = os.path.join(output_dir, f'{stack_basename}_{idx}.tiff')
        ti.imwrite(output_path, img, compression = "zlib")

if __name__ == "__main__":
    images_dir = "/mnt/external.data/TowbinLab/spsalmon/test_coco_database/labels/nuclei_masks/"
    # output_dir = f"{images_dir[:-1]}_frames/"
    output_dir = "/mnt/external.data/TowbinLab/spsalmon/test_coco_database/labels/nuclei_masks_frames/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = sorted([os.path.join(images_dir, x) for x in os.listdir(images_dir)])
    
    Parallel(n_jobs = -1, prefer="processes")(delayed(split_tiff_stack)(image, output_dir) for image in images)