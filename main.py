from pre_classification import pre_classification
from sample_selection import sample_selection
from sdae import sdae_test
from sdae import post
import os
import cv2
import numpy as np


img_path='/test_img'
#img1_n='farmland1.png'
#img2_n='farmland2.png'
img1_n='forest1.png'
img2_n='forest2.png'
#img1_n='weihe1.png'
#img2_n='weihe2.png'

print('#################    starting pre-classification    ####################')
img1_binary,img2_binary=pre_classification.edge_binarize(img1_n,img2_n)
pc_map=pre_classification.pre_classification(img1_n,img2_n,img1_binary,img2_binary)

print('#################    starting sample selection    ####################')
sample_img=np.copy(pc_map)
sample_selection.super_segmentation(img1_n,img2_n)
sample_selection.sample_selection(img1_n,img2_n)

print('#################    starting classification    ####################')
cd_map=classification(img1_n,img2_n,sample_img)
cd_map=post_process(cd_map,img1_n)
