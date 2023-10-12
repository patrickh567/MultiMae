#!/bin/bash
#SBATCH --job-name=multimae_finetuning
#SBATCH --partition=ckpt
#SBATCH --account=krishna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a40:8
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pdh4@uw.edu

OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 run_finetuning_depth.py \
--config cfgs/finetune/depth/ft_nyu_2000e_multimae-b.yaml \
--finetune ./tools/multimae_ckpt.bin \
--data_path /mmfs1/gscratch/sciencehub/vision_datasets/NYU_ksm/train \
--eval_data_path /mmfs1/gscratch/sciencehub/vision_datasets/NYU_ksm/test
