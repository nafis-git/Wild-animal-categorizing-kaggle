
# coding: utf-8

# Dogs-vs-cats classification with CNNs
#
# In this script, we'll train a convolutional neural network (CNN,
# ConvNet) to classify images of dogs from images of cats using
# PyTorch. This script is largely based on the blog post [Building
# powerful image classification models using very little
# data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
# by François Chollet.
#
# **Note that using a GPU with this script is highly recommended.**

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob

import cv2
import torch
from tqdm import tqdm_notebook
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from torchvision import transforms
from skimage import io
ImageFile.LOAD_TRUNCATED_IMAGES = True

#%matplotlib inline



def kaggle_commit_logger(str_to_log, need_print = True):
    if need_print:
        print(str_to_log)
    os.system('echo ' + str_to_log)



import json
with open(r'/scratch/project_2000859/mohamman/metadata/iwildcam2020_train_annotations.json') as json_file:
    train_data = json.load(json_file)


df_train = pd.DataFrame({'id': [item['id'] for item in train_data['annotations']],
                                'category_id': [item['category_id'] for item in train_data['annotations']],
                                'image_id': [item['image_id'] for item in train_data['annotations']],
                                'file_name': [item['file_name'] for item in train_data['images']]})

df_train.head()
