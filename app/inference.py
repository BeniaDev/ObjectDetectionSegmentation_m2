import datetime
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage

from MRCNN.mrcnn.model import log

# import the necessary packages
import imutils
import cv2
import time

from pathlib import Path

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import MRCNN.mrcnn.model as modellib
from MRCNN.mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import MRCNN.rocks as rocks

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
ROCKS_MODEL_PATH = os.path.join("./model/mask_rcnn_rocks_0003.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data/dataset/val/")


class InferenceConfig(rocks.RocksConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def evaluate_test_dataset(dataset_path: Path("../data/dataset/val/")):
    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(ROCKS_MODEL_PATH, by_name=True)

    # Load validation dataset
    dataset = rocks.RocksDataset()
    dataset.load_rocks(dataset_path, "val")
    dataset.prepare()

    # image_id = random.choice(dataset.image_ids)
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))

        # Run object detection
        results = model.detect([image], verbose=1)

        # Display results
        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names=dataset.class_names,
                                    scores=r['scores'], ax=ax, title="Predictions", image_id=info["id"])
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash



def run_demo_video(video_path: Path("../data/demo_video.avi")):
    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(ROCKS_MODEL_PATH, by_name=True)

    # Video capture
    vcapture = cv2.VideoCapture("/Users/eugeneborisov/Documents/University/3rd_year/ML/ObjectDetectionSegmentation_m2/data/demo/demo_video_0.avi")
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # Define codec and create video writer
    file_name = "demo_rocks_output_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                              cv2.VideoWriter_fourcc(*'XVID'),
                              fps, (width, height))

    count = 0
    success = True
    while success:
        print("frame: ", count)
        # Read next image
        success, image = vcapture.read()
        if success:
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects
            r = model.detect([image], verbose=0)[0]
            print(r["class_ids"])
            splash = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names="rocks",
                                        scores=r['scores'], ax=get_ax(), title="Predictions", return_image=True)

            # RGB -> BGR to save image to video
            splash = splash[..., ::-1]
            # Add image to video writer
            vwriter.write(splash)
            count += 1
    vwriter.release()

    #TODO: add logging of model saving




