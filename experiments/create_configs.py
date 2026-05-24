#!/usr/bin/env python3
"""
Generate VLM size ablation experiment config files.
Run from the repo root: python experiments/create_configs.py
"""

import os

configs = {
    "vla_256m.yaml": """save_path: tutorials/checkpoints/vla_smolvlm_256m

model:
  <<: !include vla_foundry/config_presets/models/vla_smolvlm_256m.yaml
  action_dim: 14

data:
  type: robotics
  processor: HuggingFaceTB/SmolVLM2-256M-Video-Instruct
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

distributed:
  fsdp: true

hparams:
  <<: !include vla_foundry/config_presets/hparams/diffusion_policy.yaml
  per_gpu_batch_size: 8
  global_batch_size: 64
  warmup: 1000

total_train_samples: 500000
num_checkpoints: 10
""",
    "vla_500m.yaml": """save_path: tutorials/checkpoints/vla_smolvlm_500m

model:
  <<: !include vla_foundry/config_presets/models/vla_smolvlm_256m.yaml
  action_dim: 14
  vision_language_backbone:
    type: vlm_backbone
    freeze: false
    num_vlm_layers_to_use: 1
    hf_pretrained: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
    resume_from_checkpoint: null
    resume_weights_only: false

data:
  type: robotics
  processor: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
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

distributed:
  fsdp: true

hparams:
  <<: !include vla_foundry/config_presets/hparams/diffusion_policy.yaml
  per_gpu_batch_size: 8
  global_batch_size: 64
  warmup: 1000

total_train_samples: 500000
num_checkpoints: 10
""",
    "vla_2b.yaml": """save_path: tutorials/checkpoints/vla_smolvlm_2b

model:
  <<: !include vla_foundry/config_presets/models/vla_smolvlm_256m.yaml
  action_dim: 14
  vision_language_backbone:
    type: vlm_backbone
    freeze: false
    num_vlm_layers_to_use: 1
    hf_pretrained: HuggingFaceTB/SmolVLM2-2.2B-Instruct
    resume_from_checkpoint: null
    resume_weights_only: false

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

distributed:
  fsdp: true

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
    path = os.path.join("experiments", filename)
    with open(path, "w") as f:
        f.write(content)
    print(f"Created: {path}")

print("\nDone. Run training with:")
print("  torchrun --nproc_per_node=8 vla_foundry/main.py --config_path experiments/vla_256m.yaml")
print("  torchrun --nproc_per_node=8 vla_foundry/main.py --config_path experiments/vla_500m.yaml")
print("  torchrun --nproc_per_node=8 vla_foundry/main.py --config_path experiments/vla_2b.yaml")