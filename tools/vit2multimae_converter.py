# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math

import torch
from einops import rearrange

import open_clip
import timm

def vit_to_multimae(multimae_state_dict):
    """
    Converts timm ViT weights to MultiMAE weights.
    """
    state_dict = {}
    state_dict['global_tokens'] = multimae_state_dict['cls_token']
    for k,v in multimae_state_dict.items():
        if k == 'pos_embed':
            n = int(math.sqrt(v.shape[1]))
            pos_embed = rearrange(v[:,1:], 'b (n1 n2) d -> b d n1 n2', n1=n, n2=n)
            state_dict['global_tokens'] += v[:,0]
            state_dict['input_adapters.rgb.pos_emb'] = pos_embed
        elif k == 'patch_embed.proj.weight':
            state_dict['input_adapters.rgb.proj.weight'] = v
        elif k == 'patch_embed.proj.bias':
            state_dict['input_adapters.rgb.proj.bias'] = v
        elif 'blocks.' in k:
            state_dict[k.replace('blocks.', 'encoder.')] = v
    return state_dict

def openai_to_multimae(multimae_state_dict):
    """
    Converts OpenAI ViT weights to MultiMAE weights.
    """
    state_dict = {}
    for k,v in multimae_state_dict.items():
        if k == 'visual.class_embedding':
            state_dict['global_tokens'] = multimae_state_dict['visual.class_embedding'].reshape(1, 1, 768)
        elif k == 'visual.positional_embedding':
            v = v.reshape(1, 197, 768)
            n = int(math.sqrt(v.shape[1]))
            pos_embed = rearrange(v[:,1:], 'b (n1 n2) d -> b d n1 n2', n1=n, n2=n)
            state_dict['global_tokens'] += v[:,0]
            state_dict['input_adapters.rgb.pos_emb'] = pos_embed
        elif k == 'visual.conv1.weight':
            state_dict['input_adapters.rgb.proj.weight'] = v
            # Add bias of all zeros since bias is false in openai model
            #state_dict['input_adapters.rgb.proj.bias'] = torch.zeros(768)
        elif 'visual.transformer.resblocks.' in k:
            k = k.replace('visual.transformer.', '')
            k = k.replace('resblocks.', 'encoder.')
            if 'ln_1' in k:
                k = k.replace('ln_1', 'norm1')
            if 'ln_2' in k:
                k = k.replace('ln_2', 'norm2')
            if 'c_fc' in k:
                k = k.replace('c_fc', 'fc1')
            if 'c_proj' in k:
                k = k.replace('c_proj', 'fc2')
            if 'in_proj_' in k:
                k = k.replace('in_proj_', 'qkv.')
            if 'out_proj' in k:
                k = k.replace('out_proj', 'proj')
            state_dict[k] = v

    return state_dict

def print_keys(state_dict, name):
    with open(name, 'w') as f:
        for k in state_dict:
            f.write(f'{k}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="ViT to MultiMAE checkpoint converter")
    parser.add_argument(
        "--vit_ckpt_path", type=str,
        help="Path to converted ViT(MultiMAE) checkpoint"
    )
    parser.add_argument(
        "--multimae_ckpt_path", type=str,
        help="Path to MultiMAE checkpoint"
    )
    args = parser.parse_args()
    ckpt = torch.load(args.vit_ckpt_path, map_location=torch.device('cpu'))
    ckpt['model'] = openai_to_multimae(ckpt)
    #ckpt['model'] = vit_to_multimae(ckpt)
    #print(ckpt['model'].keys())
    #print_keys(ckpt['model'], 'open_ai_state_dict.txt')
    #print_keys(ckpt['model'], 'vit_state_dict.txt')
    #exit()

    print('Converting from ViT weights to MultiMAE weights...')
    #ckpt = openai_to_multimae(ckpt)
    torch.save(ckpt, args.multimae_ckpt_path)
    print(f'Saved converted weights at {args.multimae_ckpt_path}')
