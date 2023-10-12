#!/bin/bash
#SBATCH --job-name=multimae_finetuning
#SBATCH --partition=gpu-a40
#SBATCH --account=sciencehub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=6
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pdh4@uw.edu

module load squashfuse
squashfuse_ll /data/imagenet/ILSVRC2012_img_train.sqsh ../scr/imagenet_train
squashfuse_ll /data/imagenet/ILSVRC2012_img_val.sqsh ../scr/imagenet_val

OMP_NUM_THREADS=1 torchrun --nproc_per_node=6 run_finetuning_cls.py \
--config cfgs/finetune/cls/ft_in1k_100e_multimae-b.yaml \
--finetune ./tools/multimae_ckpt.bin \
--data_path ../scr/imagenet_train \
--eval_data_path ../scr/imagenet_val \
--save_ckpt_freq 1
