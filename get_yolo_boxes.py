import numpy as np
import os
import tifffile as ti
from skimage.util import img_as_ubyte
from ultralytics import YOLO
import time

def get_yolo_pred(image, model):
    """Predict the bounding boxes of an image (str) from a YOLO model"""
    # Load image
    img = ti.imread(image)

    # Normalize image pixel values
    img = img/np.max(img)

    # Convert image pixel values to unsigned 8-bit integer
    img = img_as_ubyte(img)

    # Create a new 3-channel array of zeros with the same dimensions as the image
    img_for_pred = np.zeros((img.shape[0], img.shape[1], 3))

    # Fill the new array with the image pixel values for each channel
    img_for_pred[:, :, 0] = img
    img_for_pred[:, :, 1] = img
    img_for_pred[:, :, 2] = img

    # Use the YOLO's predict() function to obtain the bounding box predictions
    results = model.predict(source=img_for_pred, save=False, save_txt=False, verbose=False)

    return results

def process_yolo_pred(results, correct_classes = [0, 1]):
    """Process the output of a YOLO model, converts bounding boxes into masks"""
    
    # Get bounding boxes and class names from the YOLO results
    boxes = results[0].boxes
    class_names = results[0].names

    # Get the original image shape from the YOLO results
    orig_img_shape = results[0].orig_img.shape

    # Create a new output mask with the same dimensions as the original image
    output_mask = np.zeros((orig_img_shape[0], orig_img_shape[1]), dtype="uint8")

    # Check if the predicted classes match the correct classes (here, one head and one tail)
    if not np.array_equal(np.sort(boxes.cls.cpu().numpy()), np.array(correct_classes)):
        return output_mask
    
    # Loop through each bounding box
    for box in boxes:

        # Get the class name and coordinates of the box
        box_class = class_names[int(box.cls)]
        xmin, ymin, xmax, ymax = (box.xyxy).cpu().numpy().squeeze().astype(int)
        
        # If the box is a "head", set the corresponding pixels in the output mask to 7
        if box_class == "head":
            output_mask[xmin:xmax, ymin:ymax] = 7
        # If the box is a "tail", set the corresponding pixels in the output mask to 3
        if box_class == "tail":
            output_mask[xmin:xmax, ymin:ymax] = 3
    
    # Transpose the output mask and return it
    return output_mask.T

gfp_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/test_yolo_speed/"

images_gfp = sorted([os.path.join(gfp_dir, x) for x in os.listdir(gfp_dir)])

output_dir = "/mnt/external.data/TowbinLab/ngerber/20200317_wBT280_wBT281_LIPSI_daf16_GFP_20C/analysis/nuclei_coordinate_test/yolo_masks/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = YOLO("/home/spsalmon/nuclei_tracking/runs/detect/train8/weights/best.pt")
start_time = time.time()
for i, image in enumerate(images_gfp):
    print(i)
    results = get_yolo_pred(image, model)
    output_mask = process_yolo_pred(results)

    # ti.imwrite(os.path.join(output_dir, os.path.basename(image)), output_mask, compression="zlib")

print(f'time : {time.time() - start_time}')