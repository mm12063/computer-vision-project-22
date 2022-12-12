import file_locs
from os import listdir
from os.path import isfile, join
import os
from PIL import Image
import numpy as np
from os.path import exists
from tqdm import tqdm
import torch
import torchvision.transforms as T
import torchvision
import random
import file_locs
import pandas as pd

DATASETS_TO_UPDATE = [
    file_locs.TRAIN_DS,
    # file_locs.VAL_DS,
    # file_locs.TEST_DS
]

torch.manual_seed(42)


df = pd.read_csv(file_locs.TRAIN_CSV, index_col=None)
print(df.head())

# print(df['ID'].where(df['ODE'] == 1))



def balance_data(ds_loc, filename, num_to_create, starting_id):

    START_DIR = f"{ds_loc}resized/"
    DEST_DIR = f"{START_DIR}creations/"

    # REMOVE THIS
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    # filename = "578"

    full_loc = f"{START_DIR}{filename}.png"
    im = Image.open(full_loc)

    # # TODO - REMOVE
    # im.save(f"{DEST_DIR}{filename}.png", 'png')

    print(f"Creating {num_to_create}")
    for i in range(0, num_to_create):
        THRESH = 0.6
        trans = [
            T.RandomHorizontalFlip(p=1),
            T.RandomVerticalFlip(p=1),
            T.RandomRotation([30, 330]),
            T.RandomAdjustSharpness(sharpness_factor=5, p=.4),
            T.RandomAutocontrast(p=0.6)
        ]

        rand_num = random.uniform(0, 1)
        if rand_num > THRESH:
            trans.append(
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
            )

        rand_num = random.uniform(0, 1)
        if rand_num > THRESH:
            trans.append(
                T.ElasticTransform(alpha=20.0)
            )

        transforms = T.RandomApply(torch.nn.ModuleList(trans), p=1)

        new_id = starting_id + 1
        new_filename = f"{new_id}.png"
        random_image = transforms(im)
        random_image.save(f"{DEST_DIR}{new_filename}", 'png')

        row = df.loc[img_id-1].copy()
        row["ID"] = new_id
        df.loc[len(df.index)] = row
        starting_id = new_id







update = 0
for col_name in df:
    TOTAL_TARGET = 10
    num_existing = df[col_name].sum()
    highest_id = df["ID"].max()

    if num_existing < TOTAL_TARGET:
        num_to_create = TOTAL_TARGET - num_existing
        print(f"{col_name} – num_existing: {num_existing}")
        img_id = df.loc[df[col_name] == 1]['ID'].iloc[:1]


        if len(img_id):
            img_id = img_id.values[0]
            print(f"{col_name}: {img_id}")
            balance_data(file_locs.TRAIN_DS, img_id, num_to_create, highest_id)
            update += 1
        else:
            print(f"{col_name}: "+"#"*10+" No images "+"#"*10)


    if update >= 2:
        break
print(df.tail(15))

df.to_csv("./test.csv", index=False)






# For image of interest IMAGE
# Store how many images IMAGE currently has: num_existing
# Save the row for the IMAGE: IMAGE_ROW
# Save the highest ID number
# For each new image
#     Add a new row to bottom of table with IMAGE_ROW incrementing the ID in table




