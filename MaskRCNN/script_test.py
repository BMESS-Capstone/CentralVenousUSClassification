from mrcnnconfig import *
from mrcnnmodel import *
from mrcnnutils import *
from mrcnnvisualize import *
from mrcnnsubclass import VesselConfig, VesselDataset
import numpy as np
import imgaug.augmenters as aug
import argparse
from keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time

tf.logging.set_verbosity(tf.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class InferenceConfig(VesselConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.95
    IMAGE_CHANNEL_COUNT = 1
    MEAN_PIXEL = np.array([70.0])
    BACKBONE = "resnet50"

train_start = 0
train_end = 1819
val_start = 1819
val_end = 1958
test_start = 1958
test_end = 2439

# For non-random test set
#test_indices = np.arange(start=1901-434, stop=1901)
test_indices = np.arange(start=0, stop=203)

orig_shape = (374, 589)
X_path = 'datasets/X_test.npy'
Y_path = 'datasets/Y_test.npy'
X_train_path = 'datasets/X_train.npy'
Y_train_path = 'datasets/Y_train.npy'
model_path = "models/setD/vessels20210512T1002/mask_rcnn_vessels_0002.h5"

# Load the model in inference mode
inference_config = InferenceConfig()
model = MaskRCNN(mode="inference", config=inference_config, model_dir="models/vessels20210512T1002/")

# Load trained weights
print()
print("Loading Trained Weights...")
model.load_weights(model_path, by_name=True)
print('...Model loaded')
# Training dataset
dataset_train = VesselDataset(X_train_path, Y_train_path)
dataset_train.load_images(train_start, train_end)
dataset_train.prepare()

# Create test dataset
test_dataset = VesselDataset(X_path, Y_path)
test_dataset.load_images(0, 481) # Indices 110-220 are in the test set
test_dataset.prepare()

#loop through images
while(1):
    cmd = input("Enter Command: ")
    if(cmd=="predict"):

        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(18.5, 10.5)
        # Test on a random image
        image_id = random.choice(test_dataset.image_ids)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            load_image_gt(test_dataset, inference_config,
                                   image_id, use_mini_mask=False)

        log("original_image", original_image)
        log("image_meta", image_meta)
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)

        #displays the correct image

        display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                    test_dataset.class_names, figsize=(8, 8), ax = ax[0])
        start = time.time()
        results = model.detect([original_image], verbose=1)
        finish = time.time()
        r = results[0]
        display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                    test_dataset.class_names, r['scores'],figsize=(8, 8), ax = ax[1])
        
        print('Completed prediction in ' + str(finish-start) + ' seconds')
        ax[0].set_title('Ground Truth', fontdict={'fontsize': 14, 'fontweight': 'medium'})
        ax[1].set_title('Predicted ', fontdict={'fontsize': 14, 'fontweight': 'medium'})

        fig.show()
    if (cmd == "exit"):
        break
