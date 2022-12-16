import os
import subprocess

TRAIN = True
VAL = False

FINAL_SIZE = 256
SUB_DIR = "resized_complete"
PREFIX = "downsample_"

SERV_PROJ_ROOT = "/scratch/mm12063/cv_project_22/data/"

if TRAIN:
    ins = f"find fundus_ds/Training_Set/Training_Set/Training/{SUB_DIR}/{PREFIX}{FINAL_SIZE}/ -name '*.DS_Store' -type f -delete"
    print(ins)
    os.system(ins)

    # Compress train dirs
    ins = f"tar -cvzf train_{PREFIX}{FINAL_SIZE}.tar.gz fundus_ds/Training_Set/Training_Set/Training/{SUB_DIR}/{PREFIX}{FINAL_SIZE}/"
    print(ins)
    os.system(ins)

    # SCP train dirs
    ins = f"scp train_{PREFIX}{FINAL_SIZE}.tar.gz mm12063@greene-dtn.hpc.nyu.edu:/scratch/mm12063/cv_project_22/data/"
    print(ins)
    os.system(ins)

    # Extract train dirs
    ins = f"gunzip /scratch/mm12063/cv_project_22/data/train_{PREFIX}{FINAL_SIZE}.tar.gz"
    print(ins)
    subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

    ins = f"tar -xvf {SERV_PROJ_ROOT}train_{PREFIX}{FINAL_SIZE}.tar -C {SERV_PROJ_ROOT}"
    print(ins)
    subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

    # Set train permission
    ins = f"setfacl -R -m 'u:mya6510:r-x' {SERV_PROJ_ROOT}fundus_ds/Training_Set/Training_Set/Training/{SUB_DIR}/{PREFIX}{FINAL_SIZE}/"
    print(ins)
    subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()


if VAL:
    ins = f"find fundus_ds/Evaluation_Set/Evaluation_Set/Validation/{SUB_DIR}/{PREFIX}{FINAL_SIZE}/ -name '*.DS_Store' -type f -delete  "
    print(ins)
    os.system(ins)

    # Compress val dirs
    ins = f"tar -cvzf val_{PREFIX}{FINAL_SIZE}.tar.gz fundus_ds/Evaluation_Set/Evaluation_Set/Validation/{SUB_DIR}/{PREFIX}{FINAL_SIZE}/"
    print(ins)
    os.system(ins)

    # SCP val dirs
    ins = f"scp val_{PREFIX}{FINAL_SIZE}.tar.gz mm12063@greene-dtn.hpc.nyu.edu:/scratch/mm12063/cv_project_22/data/"
    print(ins)
    os.system(ins)

    # Extract val dirs
    ins = f"gunzip {SERV_PROJ_ROOT}val_{PREFIX}{FINAL_SIZE}.tar.gz"
    print(ins)
    subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

    ins = f"tar -xvf {SERV_PROJ_ROOT}val_{PREFIX}{FINAL_SIZE}.tar -C {SERV_PROJ_ROOT}"
    print(ins)
    subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

    # Set val permission
    ins = f"setfacl -R -m 'u:mya6510:r-x' {SERV_PROJ_ROOT}fundus_ds/Evaluation_Set/Evaluation_Set/Validation/{SUB_DIR}/{PREFIX}{FINAL_SIZE}/"
    print(ins)
    subprocess.Popen(f'ssh greene {ins}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()


print("Locations:\n")
if TRAIN:
    print(f"{SERV_PROJ_ROOT}fundus_ds/Training_Set/Training_Set/Training/{SUB_DIR}/{PREFIX}{FINAL_SIZE}/")
    print()

if VAL:
    print(f"{SERV_PROJ_ROOT}fundus_ds/Evaluation_Set/Evaluation_Set/Validation/{SUB_DIR}/{PREFIX}{FINAL_SIZE}/")

