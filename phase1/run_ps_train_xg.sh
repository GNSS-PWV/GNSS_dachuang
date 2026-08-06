#!/bin/bash
#SBATCH --job-name=ps_train
#SBATCH --comment="train_ps(cnn+bilstm+attention+xgboost)"

#SBATCH --partition=intel
#SBATCH --time=2-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8g

#SBATCH --output=/share/home/u23114/tj23114/packages/dachuang_pwv/PS/%x_%j.out
#SBATCH --error=/share/home/u23114/tj23114/packages/dachuang_pwv/PS/%x_%j.err

source activate kokolo

python /share/home/u23114/tj23114/packages/dachuang_pwv/PS/PS_train_xg.py