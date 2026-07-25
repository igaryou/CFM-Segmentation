#!/usr/bin/env bash
set -euo pipefail

cd /home/igarashi_25/playground_2/CSDFM/CFM/src/segv3_aug

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --batch_size is per process: local=2, world_size=2, global=4.
# WandB stays disabled because --use_wandb is intentionally omitted.
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  main.py \
  --root /home/igarashi_25/datasets/cityscapes \
  --num_classes 20 \
  --image_size 256 512 \
  --batch_size 2 \
  --grad_accum_steps 1 \
  --epochs 1 \
  --max_iters 200 \
  --seed 42 \
  --num_workers 4 \
  --backbone unet \
  --prior_type image_gaussian \
  --source_backbone segformer \
  --source_segformer_variant b1 \
  --source_pretrained \
  --source_decoder_channels 256 \
  --source_lr 1e-5 \
  --fusion_channels 128 \
  --rrdb_blocks 3 \
  --rrdb_growth_channels 32 \
  --rrdb_blocks_mask 3 \
  --rrdb_growth_channels_mask 16 \
  --unet_base_channels 45 \
  --unet_channel_mults 1,2,4,4,8 \
  --num_res_blocks 2 \
  --time_emb_dim 512 \
  --attn_levels 3,4 \
  --num_heads 4 \
  --consistency_loss ecld \
  --consistency_weight 1.0 \
  --ecld_ec_weight 1.0 \
  --ecld_td_weight 1.0 \
  --ecld_time_weighting none \
  --amp \
  --amp_dtype bf16 \
  --amp_ecld \
  --result_dir /tmp/ecld_ddp_2gpu_256x512_debug
