import file_locs
from os import listdir
from os.path import isfile, join
import os
from PIL import Image, ImageEnhance
import numpy as np
from os.path import exists
from tqdm import tqdm
import subprocess

DATASETS_TO_UPDATE = [
    file_locs.TRAIN_DS,
    file_locs.VAL_DS,
    file_locs.TEST_DS
]

PUSH_TO_SERVER = False
FINAL_SIZE = 256

def resize_in_one_shot(ds_loc, intense):
    START_DIR = f"{ds_loc}"
    DEST_DIR = f"{START_DIR}resized_complete/grayscale/{FINAL_SIZE}/{intense}/"

    LARGE_IMG_SIZE = (4288, 2848)

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    files = [f for f in listdir(START_DIR) if isfile(join(START_DIR, f))]
    for filename in tqdm(files):

        full_loc = ds_loc + filename

        if (filename == ".DS_Store") or exists(DEST_DIR + filename):
            continue

        # Resize the large camera images
        im = Image.open(full_loc)
        if im.size == LARGE_IMG_SIZE:
            im = im.resize((int(LARGE_IMG_SIZE[0]*.5), int(LARGE_IMG_SIZE[1]*.5)), Image.Resampling.LANCZOS)


        # Auto crop
        MIN = 1200
        im_crop = auto_crop(im)
        width, height = im_crop.size
        if width > MIN and height > MIN:
            im = im_crop


        # Crop to same size
        CROP_SIZE = 1350
        width, height = im.size

        left = (width - CROP_SIZE) / 2
        top = (height - CROP_SIZE) / 2
        right = (width + CROP_SIZE) / 2
        bottom = (height + CROP_SIZE) / 2

        im = im.crop((left, top, right, bottom))


        # Resize all
        width, height = im.size
        if width == FINAL_SIZE and height == FINAL_SIZE:
            continue
        im = im.resize((int(FINAL_SIZE), int(FINAL_SIZE)), Image.Resampling.LANCZOS).convert('L')

        # Grayscale
        enhancer = ImageEnhance.Contrast(im)
        factor = intense
        im = enhancer.enhance(factor)

        # Save the final image
        im.save(f"{DEST_DIR}{filename}", 'png')


def auto_crop(image):
    THRESH = 80
    im = np.array(image)
    im[im < THRESH] = 0
    y_nonzero, x_nonzero, _ = np.nonzero(im)
    cropped = image.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))
    if cropped.size != (0, 0):
        return cropped
    else:
        return image


for ds_loc in DATASETS_TO_UPDATE:
    print(f"Resizing and cropping image for DS:")
    print(ds_loc)
    intensities = [1.5, 1.75, 2.0]
    for intense in intensities:
        resize_in_one_shot(ds_loc, intense)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Compress and push to Greene
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if PUSH_TO_SERVER:

        SERV_PROJ_ROOT = "/scratch/mm12063/cv_project_22/data/"

        ins = f"find fundus_ds/Training_Set/Training_Set/Training/resized_complete/{FINAL_SIZE}/ -name '*.DS_Store' -type f -delete"
        print(ins)
        os.system(ins)

        ins = f"find fundus_ds/Evaluation_Set/Evaluation_Set/Validation/resized_complete/{FINAL_SIZE}/ -name '*.DS_Store' -type f -delete  "
        print(ins)
        os.system(ins)

        # Compress train dirs
        ins = f"tar -cvzf train_{FINAL_SIZE}.tar.gz fundus_ds/Training_Set/Training_Set/Training/resized_complete/{FINAL_SIZE}/"
        print(ins)
        os.system(ins)

        # SCP train dirs
        ins = f"scp train_{FINAL_SIZE}.tar.gz mm12063@greene-dtn.hpc.nyu.edu:/scratch/mm12063/cv_project_22/data/"
        print(ins)
        os.system(ins)


        # Compress val dirs
        ins = f"tar -cvzf val_{FINAL_SIZE}.tar.gz fundus_ds/Evaluation_Set/Evaluation_Set/Validation/resized_complete/{FINAL_SIZE}/"
        print(ins)
        os.system(ins)

        # SCP val dirs
        ins = f"scp val_{FINAL_SIZE}.tar.gz mm12063@greene-dtn.hpc.nyu.edu:/scratch/mm12063/cv_project_22/data/"
        print(ins)
        os.system(ins)


        # Extract train dirs
        ins = f"gunzip /scratch/mm12063/cv_project_22/data/train_{FINAL_SIZE}.tar.gz"
        print(ins)
        subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

        ins = f"tar -xvf {SERV_PROJ_ROOT}train_{FINAL_SIZE}.tar -C {SERV_PROJ_ROOT}"
        print(ins)
        subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()


        # Extract val dirs
        ins = f"gunzip {SERV_PROJ_ROOT}val_{FINAL_SIZE}.tar.gz"
        print(ins)
        subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

        ins = f"tar -xvf {SERV_PROJ_ROOT}val_{FINAL_SIZE}.tar -C {SERV_PROJ_ROOT}"
        print(ins)
        subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

        # Set additional permissions
        ins = f"setfacl -R -m 'u:mya6510:r-x' {SERV_PROJ_ROOT}fundus_ds/Evaluation_Set/Evaluation_Set/Validation/resized_complete/{FINAL_SIZE}/"
        print(ins)
        subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

        ins = f"setfacl -R -m 'u:mya6510:r-x' {SERV_PROJ_ROOT}fundus_ds/Training_Set/Training_Set/Training/resized_complete/{FINAL_SIZE}/"
        print(ins)
        subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()


        print("Uploaded Locations:\n")
        print(f"{SERV_PROJ_ROOT}fundus_ds/Training_Set/Training_Set/Training/resized_complete/{FINAL_SIZE}/")
        print()
        print(f"{SERV_PROJ_ROOT}fundus_ds/Evaluation_Set/Evaluation_Set/Validation/resized_complete/{FINAL_SIZE}/")

        print("*"*20)
