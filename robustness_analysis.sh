#!/bin/bash
#SBATCH --time=2:59:00
#SBATCH --account=def-jeproa
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --job-name=pgd_adversarial_training

#SBATCH --output=output_dir/%j-%x.out


module load python/3.9
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision matplotlib tqdm
pip list
cd ~/IFT6164-FourierAnalysisAdversarialExamples
python robustness_analysis.py --checkpoint_path checkpoint/pgd_adversarial_training --norm_fourierHM 4
