import os
from PIL import Image
import torch
import torchvision.transforms as T
import file_locs
import pandas as pd
import matplotlib.pyplot as plt
import shutil

new_loc_x = "fundus_ds/Training_Set/Training_Set/RFMiD_Training_Labels_w_upsampling_newen.csv"
df = pd.read_csv(file_locs.UPSAMPLED_TRAIN_CSV, index_col=None)

image_size = 256
image_dir = f"resized_complete/{image_size}/"

torch.manual_seed(42)
TOTAL_TARGET = 200

def check_updated_ds():
    df = pd.read_csv("fundus_ds/Training_Set/Training_Set/RFMiD_Training_Labels_w_DOWN.csv", index_col=None)
    DROP_VALS = ['ID', 'Disease_Risk']
    df = df.drop(labels=DROP_VALS, axis=1)

    total = 0
    totals = dict()
    for col_name in df:
        print(col_name)
        tot = df[col_name].sum()
        totals[col_name] = tot
        total += tot

    totals = dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))
    print(totals)
    print(len(totals))
    print(f"Mean class images {total/len(totals)}")

    res_dic = {}
    fig = plt.figure(figsize=(10, 5))
    for i, disease in enumerate(totals):
        print(disease, totals[disease])

        res_dic[f'{disease}_{i}'] = totals[disease]

    dis = list(res_dic.keys())
    values = list(res_dic.values())

    fig = plt.figure(figsize=(10, 5))
    plt.bar(dis, values, color ='maroon',
            width=0.4)

    plt.xticks([])
    plt.xlabel("Diseases")
    plt.ylabel("Num images")
    plt.title("Training images per disease after downsampling")
    plt.show()

    plt.bar(res_dic)
    plt.show()


def balance_data(ds_loc, img_id, num_to_create, starting_id):

    START_DIR = f"{ds_loc}{image_dir}"
    DEST_DIR = f"{START_DIR}"

    for i in range(0, num_to_create):
        THRESH = 0.6
        trans = [
            T.RandomHorizontalFlip(p=1),
            T.RandomVerticalFlip(p=1),
            T.RandomRotation([30, 330]),
            # T.RandomAdjustSharpness(sharpness_factor=2.5, p=0.6),
            # T.RandomAutocontrast(p=0.6)
        ]

        # rand_num = random.uniform(0, 1)
        # if rand_num > THRESH:
        #     trans.append(
        #         T.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 1.5))
        #     )

        # rand_num = random.uniform(0, 1)
        # if rand_num > THRESH:
        #     trans.append(
        #         T.ElasticTransform(alpha=10.0)
        #     )

        transforms = T.RandomApply(torch.nn.ModuleList(trans), p=1)

        full_loc = f"{START_DIR}{img_id}.png"

        im = Image.open(full_loc)
        new_id = starting_id + 1
        new_filename = f"{new_id}.png"
        random_image = transforms(im)
        loc_and_name = f"{DEST_DIR}{new_filename}"
        random_image.save(loc_and_name, 'png')

        # Make sure the image was created before adding to the DF
        if os.path.exists(loc_and_name):
            row = df.loc[img_id-1].copy()
            row["ID"] = new_id
            df.loc[len(df.index)] = row
            starting_id = new_id


def begin_upsampling():
    totals = dict()
    for col_name in df:
        totals[col_name] = df[col_name].sum()

    totals = dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))
    for disease in totals:
        print(disease, totals[disease])

    for col_name in df:
        if col_name == "ID" or col_name == "Disease_Risk":
            continue
        num_existing = df[col_name].sum()
        highest_id = df["ID"].max()

        if num_existing < TOTAL_TARGET:
            num_to_create = TOTAL_TARGET - num_existing

            # Get the row ID containing the fewest number of other diseases
            lowest = float('inf')
            img_id = 0
            row_ids = df.loc[df[col_name] == 1]['ID']
            for id in row_ids:
                row_sum = df.loc[id-1].sum() - id
                if row_sum < lowest:
                    lowest = row_sum
                    img_id = id

            if len(row_ids):
                balance_data(file_locs.TRAIN_DS, img_id, num_to_create, highest_id)
            else:
                print(f"{col_name}: "+"#"*10+" No images "+"#"*10)

    df.to_csv(new_loc_x, index=False)


def move_non_represented_diseases_into_training_set(dis_name, sample_from_csv, sample_from_loc):
    sample_loc_df = pd.read_csv(sample_from_csv, index_col=None)
    train_df = pd.read_csv(file_locs.UPSAMPLED_TRAIN_CSV, index_col=None)

    highest_id = train_df["ID"].max()

    lowest = float('inf')
    img_id = 0
    row_ids = sample_loc_df.loc[sample_loc_df[dis_name] == 1]['ID']
    for id in row_ids:
        row_sum = sample_loc_df.loc[id - 1].sum() - id
        if row_sum < lowest:
            lowest = row_sum
            img_id = id

    print("img_id", img_id)

    START_DIR = f"{sample_from_loc}"
    DEST_DIR = f"{file_locs.TRAIN_DS}{image_dir}"

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    # Move the image into the train ds
    full_loc = f"{START_DIR}{img_id}.png"
    print(full_loc)

    im = Image.open(full_loc)

    new_id = highest_id + 1
    new_filename = f"{new_id}.png"
    new_full_loc = f"{DEST_DIR}{new_filename}"

    im.save(new_full_loc, 'png')

    # Also move the image into the train resized ds
    # LARGE_IMG_SIZE = (4288, 2848)
    # if im.size == LARGE_IMG_SIZE:
    #     im = im.resize((int(LARGE_IMG_SIZE[0]*.5), int(LARGE_IMG_SIZE[1]*.5)), Image.Resampling.LANCZOS)
    # im.save(f"{file_locs.TRAIN_DS}{image_dir}{new_filename}", 'png')

    if os.path.exists(new_full_loc):
        row = sample_loc_df.loc[img_id - 1].copy()
        row["ID"] = new_id
        train_df.loc[len(train_df.index)] = row

    train_df.to_csv(file_locs.UPSAMPLED_TRAIN_CSV, index=False)
    print(train_df.tail())


def downsample():
    print()
    print("*"*30)
    print()
    df = pd.read_csv(new_loc_x, index_col=None)

    totals = dict()
    for col_name in df:
        totals[col_name] = df[col_name].sum()

    totals = dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))
    for disease in totals:
        print(disease, totals[disease])

    chosen = []
    for col_name in df:
        if col_name == "ID" or col_name == "Disease_Risk":
            continue

        num_existing = df[col_name].sum()

        if num_existing > TOTAL_TARGET:
            num_to_remove = num_existing - TOTAL_TARGET

            for x in range(0, num_to_remove):
                # Get the row ID containing the fewest number of other diseases
                lowest = float('inf')
                img_id = 0

                img_nums = df.loc[(df[col_name] == 1)]['ID']

                for img_num in img_nums:
                    row_sum = df.loc[img_num - 1].sum() - img_num
                    if row_sum < lowest:
                        lowest = row_sum
                        img_id = img_num

                if len(img_nums):
                    chosen.append(img_id)
                    df = df[df["ID"] != img_id]

    for id in chosen:
        filename = f"{id}.png"
        source = "fundus_ds/Training_Set/Training_Set/Training/resized_complete/256_upsample/" + filename
        destination = "fundus_ds/Training_Set/Training_Set/Training/resized_complete/256_upsample_moved/"
        if os.path.isfile(source):
            shutil.move(source, destination)
            print('Moved:', source)

    df.to_csv("fundus_ds/Training_Set/Training_Set/Training/RFMiD_Training_Labels_w_DOWN.csv", index=False)


    total = 0

    DROP_VALS = ['ID', 'Disease_Risk']

    df = df.drop(labels=DROP_VALS, axis=1)

    totals = dict()
    for col_name in df:
        print(col_name)
        tot = df[col_name].sum()
        totals[col_name] = tot
        total += tot

    totals = dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))

    res_dic = {}
    fig = plt.figure(figsize=(10, 5))
    for i, disease in enumerate(totals):
        print(disease, totals[disease])

        res_dic[f'{disease}_{i}'] = totals[disease]

    courses = list(res_dic.keys())
    values = list(res_dic.values())

    fig = plt.figure(figsize=(10, 5))

    plt.bar(courses, values,
            width=0.4)

    plt.xticks([])

    plt.xlabel("Diseases")
    plt.ylabel("Num images")
    plt.title("Training images per disease after downsampling")
    plt.show()


def create_upsampling_cvs(orig_csv):
    if not os.path.exists(file_locs.UPSAMPLED_TRAIN_CSV):
        df = pd.read_csv(orig_csv, index_col=None)
        df.to_csv(file_locs.UPSAMPLED_TRAIN_CSV, index=False)


def main():
    upsample = True
    if upsample:
        create_upsampling_cvs(file_locs.TRAIN_CSV)

        move_non_represented_diseases_into_training_set(dis_name="ODPM",
                                                        sample_from_csv=file_locs.VAL_CSV,
                                                        sample_from_loc=f"{file_locs.VAL_DS}{image_dir}")

        move_non_represented_diseases_into_training_set(dis_name="HR",
                                                        sample_from_csv=file_locs.TEST_CSV,
                                                        sample_from_loc=f"{file_locs.VAL_DS}{image_dir}")

        begin_upsampling()
        check_updated_ds()

    downsample()

if __name__ == '__main__':
    main()

