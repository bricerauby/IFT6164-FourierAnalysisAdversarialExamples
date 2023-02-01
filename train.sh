#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --account=def-jeproa
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --job-name=CIFAR10-low_freq_random_075
##SBATCH --array=1-10
#SBATCH --output=output_dir/%j-%x.out


module load python/3.9
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision matplotlib tqdm argparse
pip list
cd ~/IFT6164-FourierAnalysisAdversarialExamples
nvidia-smi
python pdg_adversarial_training_low_freq_random_075.py
