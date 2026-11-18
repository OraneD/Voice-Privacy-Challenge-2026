#!/bin/bash
#SBATCH --job-name=ECAPA_SSL
#SBATCH --account=yjs@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00



source env.sh

python run_evaluation.py --config "$1" --force_compute False