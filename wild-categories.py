
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import glob



import torch
from tqdm import tqdm_notebook
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True



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

print("df_train", df_train.head())


df_image = pd.DataFrame.from_records(train_data['images'])

indices = []
for _id in df_image[df_image['location'] == 537]['id'].values:
    indices.append( df_train[ df_train['image_id'] == _id ].index )

for the_index in indices:
    df_train = df_train.drop(df_train.index[the_index])


indices = []
for i in df_train['file_name']:
    try:
        Image.open('/scratch/project_2000859/mohamman/train/' + i)
    except:
        print(i)
        df_train.drop(df_train.loc[df_train['file_name']==i].index, inplace=True)


with open(r'/scratch/project_2000859/mohamman/metadata/iwildcam2020_test_information.json') as f:
    test_data = json.load(f)

df_test = pd.DataFrame.from_records(test_data['images'])
print("df_test",df_test.head())


batch_size = 64
IMG_SIZE = 64

N_EPOCHS = 1

ID_COLNAME = 'file_name'
ANSWER_COLNAME = 'category_id'
TRAIN_IMGS_DIR = r'../scratch/project_2000859/mohamman/train/'
TEST_IMGS_DIR = r'../scratch/project_2000859/mohamman/test/'


train_df, test_df = train_test_split(df_train[[ID_COLNAME, ANSWER_COLNAME]],
                                     test_size = 0.15,
                                     shuffle = True
                                    )


print("train_df.head",train_df.head(5))


CLASSES_TO_USE = df_train['category_id'].unique()
print("CLASSES_TO_USE",CLASSES_TO_USE)


CLASSMAP = dict(
    [(i, j) for i, j
     in zip(CLASSES_TO_USE, range(NUM_CLASSES))
    ]
)
print ("CLASSMAP", CLASSMAP)


REVERSE_CLASSMAP = dict([(v, k) for k, v in CLASSMAP.items()])
