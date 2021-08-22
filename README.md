# cvClassification

This code was adapated from the paper: https://doi.org/10.1007/s11548-020-02248-2 from their Github repository found at https://github.com/VASST/AIVascularSegmentation 

Their code was heavily based on the well-known architecture found at this repository: https://github.com/matterport/Mask_RCNN

Please go to this OneDrive folder to download the numpy image arrays they used for training and testing: https://1drv.ms/u/s!Akxm4gUER2IFag1CJ1LtM_qB6HY?e=0aqATl 

Please go to this link for the training and test splits that I used. You may also find my trained models at this link: https://drive.google.com/drive/folders/1HhsIkitnZiNBbyLZqi43U5MXLASa0feU?usp=sharing 


Below is a breakdown of the files: 

train_mrcc - file to train a Mask R-CNN algorithm to segment the CA and IJV from neck US images. The data used to train can be found under the PARTITION FOR CROSS-VALIDATION, this provides the start and end locations for a four-fold cross validation.

train_unet - file to train a U-Net algorithm to segment the CA and IJV from neck US images. The data used to train can be found under the PARTITION FOR CROSS-VALIDATION, this provides the start and end locations for a four-fold cross validation.

test_mrcnn and test_unet - files to run to obtain test results.

Unet - file that executes the U-net segmentation algorithm.

mrcnnconfig - Mask R-CNN Base Configurations class.

mrcnnmodel - The main Mask R-CNN model implementation.

mrcnnmodeldist - delete?

mrcnnparrallel_model - Mask R-CNN Multi-GPU Support for Keras.

mrcnnsubclass - ?

mrcnnutils - Mask R-CNN Common utility functions and classes.

mrcnnvisualize - Mask R-CNN Display and Visualization Functions.

