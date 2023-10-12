#!/bin/bash
#SBATCH --job-name=multimae_finetuning
#SBATCH --partition=ckpt
#SBATCH --account=krishna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a40:8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pdh4@uw.edu

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_finetuning_semseg.py \
--config cfgs/finetune/semseg/ade/ft_ade_64e_multimae-b_rgb.yaml \
--finetune ./tools/multimae_ckpt.bin \
--data_path /mmfs1/gscratch/sciencehub/vision_datasets/ADE20k/train \
--eval_data_path /mmfs1/gscratch/sciencehub/vision_datasets/ADE20k/test

#srun --job-name=linprobe \
#--partition=ckpt \
#--account=krishna \
#--nodes=1 \
#--ntasks-per-node=8 \
#--gpus=a40:8 \
#--gpus-per-node=8 \
#--cpus-per-task=4 \
#--gpus=8 \
#--mem=256G \
#--time=10:00:00 \
#--mail-type=ALL \
#--mail-user=pdh4@uw.edu \
#torchrun --nproc_per_node=8 run_finetuning_semseg.py \
#--config cfgs/finetune/semseg/ade/ft_ade_64e_multimae-b_rgb.yaml \
#--finetune ./tools/multimae_ckpt.bin \
#--data_path /mmfs1/gscratch/sciencehub/vision_datasets/ADE20k/train \
#--eval_data_path /mmfs1/gscratch/sciencehub/vision_datasets/ADE20k/test
