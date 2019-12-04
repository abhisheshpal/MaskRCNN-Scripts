import os
import sys
import random
import math
import numpy as np

import colorsys
import cv2

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
VIDEO_FILE = os.path.abspath(sys.argv[1])

# video properties
VIDEO_DIMENSIONS = (1920,1080)
VIDEO_FPS = 30

# classes to annotate
CLASSES = [3,4,6,8]

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Run detection
OUTPUT_FILE = VIDEO_FILE.replace('.', '_maskRCNN.')
inputV = cv2.VideoCapture(VIDEO_FILE)
outputV = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc('M','J','P','G'), VIDEO_FPS, VIDEO_DIMENSIONS)

print('Starting...')

boxesArray = []
while (True):
    ret, frame = inputV.read()

    if ret != True:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = model.detect([frame], verbose=1)[0]

    boxes = r['rois']
    masks = r['masks']
    class_ids = r['class_ids']
    scores = r['scores']
    N = boxes.shape[0]
    colors = random_colors(N)
    # masked_image = image.astype(np.uint32).copy()
    masked_image = frame.copy()
    frameBoxes = []
    for i in range(N):
        if not class_ids[i] in CLASSES:
            print("Skipping: {}".format(class_ids[i]))
            print(scores[i])
            continue
        else:
            print("Found: {}".format(class_ids[i]))
            print(scores[i])
        color = colors[i]
        # cv2_color = tuple(reversed(color))
        cv2_color = tuple([int(x*255) for x in color])

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        frameBoxes.append((x1,y1,x2,y2))

        # Bounding Box
        cv2.rectangle(masked_image, (x1,y1), (x2,y2), cv2_color)

        # Label
        # caption = "{} {:.3f}".format("person", scores[i])
        # cv2.putText(masked_image, caption, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cv2_color, 2)

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

    boxesArray.append(frameBoxes)
    del frameBoxes
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    outputV.write(masked_image)

inputV.release()
outputV.release()

pklDest = VIDEO_FILE[:VIDEO_FILE.rfind('.')+1] + pkl
with open(pklDest, 'w+') as pklFile:
    pickle.dump(boxesArray, pklFile)

