# Computer Vision Project Fall '22

1. First, update the `file_locs.py` file to store your fundus images directory locations. It should look similar to this:

``` 
TRAIN_DS = "./fundus_ds/Training_Set/Training_Set/Training/"
TRAIN_CSV = "./fundus_ds/Training_Set/Training_Set/RFMiD_Training_Labels.csv"
VAL_DS = "./fundus_ds/Evaluation_Set/Evaluation_Set/Validation/"
VAL_CSV = "./fundus_ds/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv"
TEST_DS = "./fundus_ds/Test_Set/Test_Set/Test/"
TEST_CSV = "./fundus_ds/Test_Set/Test_Set/RFMiD_Testing_Labels"
```

2. Then run `pre_processing.py`. This will create the smaller cropped images we'll use to train. Once this has finished running (it'll take some time...), then...


3. Ensure that you load in the smaller images as the training/validation dataset in `main.py`:

```
train_dataset = CustomDataSet(file_locs.TRAIN_DS + "resized/auto_crop/same_size/", y_values=y_train_vals, transform=transforms)
val_dataset = CustomDataSet(file_locs.VAL_DS + "resized/auto_crop/same_size/", y_values=y_val_vals, transform=transforms)
```


4. Once 2 has completed, you can run the `main.py` script 



====================================

### To run the slurm job:
1. Update the `basic_slum.s` file with your Singularity and personal details. 
2. On Greene, I created a `/scratch/NET_ID/cv_proj/slurm_scripts/` to put the slurm scripts in. Copy `basic_slum.s` to this directory.  
3. Create a `test.py` in `/scratch/NET_ID/cv_proj/` – I just put a single print statement in it to test.
4. Go back into `/scratch/NET_ID/cv_proj/slurm_scripts/` and run `sbatch basic_slum.s`
5. Then run `squeue -u $USER` to see your job sitting in the queue. PD means pending. R means running...
6. The job will disappear from the queue once it's completed. You should also receive an email to tell you it's done.
7. In `/scratch/NET_ID/cv_proj/slurm_scripts/` run `ls -la` to see the output file that's been created.
8. Run `cat NEW_OUTPUT_FILE.out` to see the output from the `test.py`
    
    
    
