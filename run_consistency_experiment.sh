#!/usr/bin/env bash
set -euo pipefail

# Copy-ready invocations:
#   CUDA_VISIBLE_DEVICES=0 bash run_consistency_experiment.sh psd
#   CUDA_VISIBLE_DEVICES=0 bash run_consistency_experiment.sh csd
#   CUDA_VISIBLE_DEVICES=0 bash run_consistency_experiment.sh ecld
#   CUDA_VISIBLE_DEVICES=0 bash run_consistency_experiment.sh esd

loss_type="${1:-}"
case "${loss_type}" in
  psd|csd|ecld|esd) ;;
  *)
    echo "usage: CUDA_VISIBLE_DEVICES=<gpu> bash $0 {psd|csd|ecld|esd}" >&2
    exit 2
    ;;
esac

result_dir="/home/igarashi_25/playground_2/CSDFM/CFM/result/V3_aug/unet_align015_aug_128x256_400ep_consistency_${loss_type}"
wandb_name="unet-align015-aug-128x256-400epoch-consistency-${loss_type}"
cd /home/igarashi_25/playground_2/CSDFM

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python /home/igarashi_25/playground_2/CSDFM/CFM/src/segv3_aug/main.py \
  --root /home/igarashi_25/datasets/cityscapes \
  --result_dir "${result_dir}" \
  --num_classes 20 \
  --image_size 128 256 \
  --epochs 400 \
  --seed 42 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --num_workers 8 \
  --use_ccdm_aug \
  --hflip_prob 0.5 \
  --color_jitter_brightness 0.2 \
  --color_jitter_contrast 0.2 \
  --color_jitter_saturation 0.2 \
  --color_jitter_hue 0.1 \
  --val_eval_epochs 50,100,150,200,250,300,350,400 \
  --val_eval_split val \
  --val_eval_num_steps 1 \
  --backbone unet \
  --lr 1e-4 \
  --optimizer adamw \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --warmup_epochs 10 \
  --eta_min 1e-6 \
  --amp \
  --amp_dtype bf16 \
  --eta 0.5 \
  --consistency_loss "${loss_type}" \
  --consistency_weight 1.0 \
  --consistency_eps 1e-6 \
  --consistency_time_eps 1e-4 \
  --ecld_ec_weight 4.0 \
  --ecld_td_weight 2.0 \
  --ecld_time_weighting none \
  --eps 0.05 \
  --label_smoothing 0.0 \
  --prior_type image_gaussian \
  --prior_noise_std 1.0 \
  --use_loss_align \
  --align_weight 0.15 \
  --var_weight 0.0 \
  --align_eps 1e-8 \
  --source_backbone segformer \
  --source_segformer_variant b2 \
  --source_pretrained \
  --source_decoder_channels 256 \
  --source_fixed_std 1.0 \
  --source_mu_tanh_scale 0.0 \
  --source_lr 1e-4 \
  --fusion_channels 128 \
  --rrdb_blocks 10 \
  --rrdb_growth_channels 32 \
  --rrdb_blocks_mask 3 \
  --rrdb_growth_channels_mask 16 \
  --unet_base_channels 64 \
  --unet_channel_mults 1,2,4,6,8 \
  --num_res_blocks 2 \
  --time_emb_dim 512 \
  --attn_levels 3,4 \
  --dropout 0.0 \
  --num_heads 4 \
  --endpoint_segformer_variant b2 \
  --endpoint_decoder_channels 256 \
  --endpoint_time_emb_dim 512 \
  --endpoint_drop_path_rate 0.1 \
  --use_wandb \
  --wandb_project CFM \
  --wandb_name "${wandb_name}" \
  --wandb_mode online \
  --wandb_log_interval 50 \
  --wandb_num_images 4
