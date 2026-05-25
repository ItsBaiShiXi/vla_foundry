#!/usr/bin/env python3
"""
Diffusion Policy Head size ablation study on ALOHA VLA.
VLM backbone: SmolVLM2-2.2B (frozen, fixed)
Diffusion head: 11M / 100M / 410M (varied)

Run from repo root: python experiments/create_configs.py
"""

import os

VLM = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

DATA = """
data:
  type: robotics
  processor: HuggingFaceTB/SmolVLM2-2.2B-Instruct
  image_size: 512
  img_num_tokens: 64
  seq_len: 512
  dataset_manifest:
    - tutorials/data/preprocessed/aloha/shards/manifest.jsonl
  dataset_statistics:
    - tutorials/data/preprocessed/aloha/shards/stats.json
  dataset_modality:
    - robotics
  dataset_weighting:
    - 1.0
  camera_names:
    - observation.images.top
  image_indices:
    - -1
    - 0
  action_fields:
    - action
  proprioception_fields:
    - observation.state
  language_instruction_types:
    - original
  pose_groups: []
  intrinsics_fields: []
  extrinsics_fields: []
  lowdim_past_timesteps: 1
  lowdim_future_timesteps: 14
  allow_multiple_epochs: true
  num_workers: 4
  prefetch_factor: 8
"""

DISTRIBUTED = """
distributed:
  fsdp: true
"""

configs = {
    "vla_head_11m.yaml": """save_path: tutorials/checkpoints/vla_head_11m

model:
  type: diffusion_policy
  action_dim: 14
  num_action_head_repeats: 8
  input_noise_std: 0.05
  diffusion_step_conditioning: add
  use_flow_matching_scheduler: true
  vision_language_backbone:
    type: vlm_backbone
    freeze: true
    num_vlm_layers_to_use: 1
    hf_pretrained: HuggingFaceTB/SmolVLM2-2.2B-Instruct
    resume_from_checkpoint: null
    resume_weights_only: false
  transformer:
    <<: !include vla_foundry/config_presets/models/transformer_11m.yaml
    is_causal: false
  noise_scheduler:
    num_timesteps: 1000
    beta_start: 0.0001
    beta_end: 0.02
    clamp_range: [-1.5, 1.5]
""" + DATA + DISTRIBUTED + """
hparams:
  <<: !include vla_foundry/config_presets/hparams/diffusion_policy.yaml
  per_gpu_batch_size: 8
  global_batch_size: 64
  warmup: 1000

total_train_samples: 500000
num_checkpoints: 10
""",

    "vla_head_100m.yaml": """save_path: tutorials/checkpoints/vla_head_100m

model:
  type: diffusion_policy
  action_dim: 14
  num_action_head_repeats: 8
  input_noise_std: 0.05
  diffusion_step_conditioning: add
  use_flow_matching_scheduler: true
  vision_language_backbone:
    type: vlm_backbone
    freeze: true
    num_vlm_layers_to_use: 1
    hf_pretrained: HuggingFaceTB/SmolVLM2-2.2B-Instruct
    resume_from_checkpoint: null
    resume_weights_only: false
  transformer:
    <<: !include vla_foundry/config_presets/models/transformer_100m.yaml
    is_causal: false
  noise_scheduler:
    num_timesteps: 1000
    beta_start: 0.0001
    beta_end: 0.02
    clamp_range: [-1.5, 1.5]
""" + DATA + DISTRIBUTED + """
hparams:
  <<: !include vla_foundry/config_presets/hparams/diffusion_policy.yaml
  per_gpu_batch_size: 8
  global_batch_size: 64
  warmup: 1000

total_train_samples: 500000
num_checkpoints: 10
""",

    "vla_head_410m.yaml": """save_path: tutorials/checkpoints/vla_head_410m

model:
  type: diffusion_policy
  action_dim: 14
  num_action_head_repeats: 8
  input_noise_std: 0.05
  diffusion_step_conditioning: add
  use_flow_matching_scheduler: true
  vision_language_backbone:
    type: vlm_backbone
    freeze: true
    num_vlm_layers_to_use: 1
    hf_pretrained: HuggingFaceTB/SmolVLM2-2.2B-Instruct
    resume_from_checkpoint: null
    resume_weights_only: false
  transformer:
    <<: !include vla_foundry/config_presets/models/transformer_410m.yaml
    is_causal: false
  noise_scheduler:
    num_timesteps: 1000
    beta_start: 0.0001
    beta_end: 0.02
    clamp_range: [-1.5, 1.5]
""" + DATA + DISTRIBUTED + """
hparams:
  <<: !include vla_foundry/config_presets/hparams/diffusion_policy.yaml
  per_gpu_batch_size: 4
  global_batch_size: 64
  warmup: 1000

total_train_samples: 500000
num_checkpoints: 10
""",
}

for filename, content in configs.items():
    path = filename
    with open(path, "w") as f:
        f.write(content)
    print(f"Created: {path}")

print("\nDone. Run training with:")
print("  torchrun --nproc_per_node=8 vla_foundry/main.py --config_path experiments/vla_head_11m.yaml")
print("  torchrun --nproc_per_node=8 vla_foundry/main.py --config_path experiments/vla_head_100m.yaml")
print("  torchrun --nproc_per_node=8 vla_foundry/main.py --config_path experiments/vla_head_410m.yaml")

