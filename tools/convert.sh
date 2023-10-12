#!/bin/bash

python3 vit2multimae_converter.py --multimae_ckpt_path ./multimae_ckpt.bin --vit_ckpt_path ../../CLIP-ViT-B-16-DataComp.L-s1B-b8K/open_clip_pytorch_model.bin
#python3 vit2multimae_converter.py --multimae_ckpt_path ./ --vit_ckpt_path ../../vit_base_patch16_224.augreg2_in21k_ft_in1k/pytorch_model.bin
