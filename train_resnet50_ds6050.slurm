#!/bin/bash
#SBATCH --mail-user=ezq9qu@virginia.edu
#SBATCH --mail-type=ALL   ##Other option: ALL
#SBATCH -A shakeri_ds6050
#SBATCH --partition=gpu		##Define GPU partition
#SBATCH --gres=gpu:2		##Specify desired number of GPUs
#SBATCH --constraint=a6000	##Optional: specify type of GPU
#SBATCH --ntasks=1		##Specify number of tasks 

#SBATCH --cpus-per-task=4	##Specify number of CPUs per task
#SBATCH --mem=80G		##Specify amount of CPU storage needed
#SBATCH -t 3-00:00:00		##Specify time constraint in Days-Hours:Minutes:Seconds format
#SBATCH -J finetune_resnet_script		##Name the job
#SBATCH -o finetune_resnet_script-%A.out		##Provide a name for the .out file- this is where any output printed to console will be stored after the job finishes
#SBATCH -e finetune_resnet_script-%A.err		##Provide a name for the .err file- this is where any error messages and output will be printed during the job


module purge		##Purge any existing modules on the compute resources
module load miniforge		##Load miniforge for python
##source activate /scratch/ezq9qu/llm_course		##Load your virtual environment
python fine_tune_resnet.py	##Run your .py file