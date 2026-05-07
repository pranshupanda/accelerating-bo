#!/bin/bash -l

#SBATCH --partition=a100-4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00          # A period of 24 hours is the maximum for one scheduling.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>  # Replace this with your own x500. Emails will be sent to notify the start and end of the program
#SBATCH --output=terminal_output.txt  # This is a new file automatically created to record all the terminal outputs from this run.
#SBATCH --mem=20G

source ~/.bashrc
conda activate gpuenv              # Replace these with your own environment

PYTHONPATH="." python batch64.py
