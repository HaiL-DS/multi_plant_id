#!/bin/bash
#SBATCH --mail-user=hl9h@virginia.edu
#SBATCH --mail-type=BEGIN,END   ##Other option: ALL
#SBATCH -A shakeri_ds6050
#SBATCH --partition=gpu		##Define GPU partition
#SBATCH --gres=gpu:1		##Specify desired number of GPUs
#SBATCH --constraint=a6000	##Optional: specify type of GPU
#SBATCH --ntasks=1		##Specify number of tasks 

#SBATCH --cpus-per-task=4	##Specify number of CPUs per task
#SBATCH --mem=66G		##Specify amount of CPU storage needed
#SBATCH -t 0-20:00:00		##Specify time constraint in Days-Hours:Minutes:Seconds format
#SBATCH -J trainhead_resnet_script		##Name the job
#SBATCH -o trainhead_resnet_script-%A.out		##Provide a name for the .out file- this is where any output printed to console will be stored after the job finishes
#SBATCH -e trainhead_resnet_script-%A.err		##Provide a name for the .err file- this is where any error messages and output will be printed during the job


module purge		##Purge any existing modules on the compute resources
module load miniforge		##Load miniforge for python
##source activate /home/hl9h/.conda/envs/dsvenv		##Load your virtual environment
python train_head_resnet.py     ##Run your .py file