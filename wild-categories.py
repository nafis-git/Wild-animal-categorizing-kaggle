
# coding: utf-8

# In[9]:


get_ipython().system('module load pytorch')


get_ipython().system('pip install torch')


# In[12]:


import torch


# In[14]:


get_ipython().system(' pip install pillow')


# In[16]:


import PIL


# In[19]:


import os
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import PIL
import glob

from tqdm import tqdm_notebook
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True



import json
with open(r'/scratch/project_2000859/mohamman/metadata/iwildcam2020_train_annotations.json') as json_file:
    train_data = json.load(json_file)


df_train = pd.DataFrame({'id': [item['id'] for item in train_data['annotations']],
                                'category_id': [item['category_id'] for item in train_data['annotations']],
                                'image_id': [item['image_id'] for item in train_data['annotations']],
                                'file_name': [item['file_name'] for item in train_data['images']]})


print("df_train")
df_train.head()


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
print("df_test")
df_test.head()


im = Image.open(r"/scratch/project_2000859/mohamman/test/88037cce-21bc-11ea-a13a-137349068a90.jpg")
im.show()



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
print("train_df.head")
train_df.head(5)




CLASSES_TO_USE = df_train['category_id'].unique()
print("CLASSES_TO_USE",CLASSES_TO_USE)


# In[36]:


NUM_CLASSES = len(CLASSES_TO_USE)
NUM_CLASSES



CLASSMAP = dict(
    [(i, j) for i, j
     in zip(CLASSES_TO_USE, range(NUM_CLASSES))
    ]
)
print ("CLASSMAP", CLASSMAP)



model = models.densenet121(pretrained='imagenet')




new_head = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.classifier = new_head




normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_augmentation = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    normalizer,
])

val_augmentation = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    normalizer,
])




class IMetDataset(Dataset):
    
    def __init__(self,
                 df,
                 images_dir,
                 n_classes = NUM_CLASSES,
                 id_colname = ID_COLNAME,
                 answer_colname = ANSWER_COLNAME,
                 label_dict = CLASSMAP,
                 transforms = None
                ):
        self.df = df
        self.images_dir = images_dir
        self.n_classes = n_classes
        self.id_colname = id_colname
        self.answer_colname = answer_colname
        self.label_dict = label_dict
        self.transforms = transforms
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):        
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_name = img_id # + self.img_ext
        img_path = os.path.join(self.images_dir, img_name)
          
        img = Image.open(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.answer_colname is not None:              
            label = torch.zeros((self.n_classes,), dtype=torch.float32)
            label[self.label_dict[cur_idx_row[self.answer_colname]]] = 1.0

            return img, label

        else:
            return img, img_id  



train_dataset = IMetDataset(train_df, TRAIN_IMGS_DIR, transforms = train_augmentation)
test_dataset = IMetDataset(test_df, TRAIN_IMGS_DIR, transforms = val_augmentation)



BS = 24

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)



def cuda(x):
    return x.cuda(non_blocking=True)



def f1_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 1, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))



def train_one_epoch(model, train_loader, criterion, optimizer, steps_upd_logging = 250):
    model.train();
    
    total_loss = 0.0
    
    train_tqdm = tqdm_notebook(train_loader)
    
    
    for step, (features, targets) in enumerate(train_tqdm):
        try:        
            features, targets = cuda(features), cuda(targets)

            optimizer.zero_grad()

            logits = model(features)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % steps_upd_logging == 0:
                logstr = f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}'
                train_tqdm.set_description(logstr)
                kaggle_commit_logger(logstr, need_print=False)
                
        except:
            pass
        
    return total_loss / (step + 1)




def validate(model, valid_loader, criterion, need_tqdm = False):
    model.eval();
    
    test_loss = 0.0
    TH_TO_ACC = 0.5
    
    true_ans_list = []
    preds_cat = []
    
    with torch.no_grad():
        
        if need_tqdm:
            valid_iterator = tqdm_notebook(valid_loader)
        else:
            valid_iterator = valid_loader
        
        for step, (features, targets) in enumerate(valid_iterator):
            features, targets = cuda(features), cuda(targets)

            logits = model(features)
            loss = criterion(logits, targets)

            test_loss += loss.item()
            true_ans_list.append(targets)
            preds_cat.append(torch.sigmoid(logits))

        all_true_ans = torch.cat(true_ans_list)
        all_preds = torch.cat(preds_cat)
                
        f1_eval = f1_score(all_true_ans, all_preds).item()

    logstr = f'Mean val f1: {round(f1_eval, 5)}'
    kaggle_commit_logger(logstr)
    return test_loss / (step + 1), f1_eval




criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)


get_ipython().run_cell_magic('time', '', '\nTRAIN_LOGGING_EACH = 500\n\ntrain_losses = []\nvalid_losses = []\nvalid_f1s = []\nbest_model_f1 = 0.0\nbest_model = None\nbest_model_ep = 0\n\nfor epoch in range(1, N_EPOCHS + 1):\n    ep_logstr = f"Starting {epoch} epoch..."\n    kaggle_commit_logger(ep_logstr)\n    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, TRAIN_LOGGING_EACH)\n    train_losses.append(tr_loss)\n    tr_loss_logstr = f\'Mean train loss: {round(tr_loss,5)}\'\n    kaggle_commit_logger(tr_loss_logstr)\n\n    valid_loss, valid_f1 = validate(model, test_loader, criterion)  \n    valid_losses.append(valid_loss)    \n    valid_f1s.append(valid_f1)       \n    val_loss_logstr = f\'Mean valid loss: {round(valid_loss,5)}\'\n    kaggle_commit_logger(val_loss_logstr)\n    sheduler.step(valid_loss)\n\n    if valid_f1 >= best_model_f1:    \n        best_model = model        \n        best_model_f1 = valid_f1        \n        best_model_ep = epoch     ')





