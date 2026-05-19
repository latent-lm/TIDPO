#!/usr/bin/env python3
"""Run TI-DPO on Meta-Llama-3-8B-Instruct with UltraFeedback Binarized prefs."""

import subprocess
import sys
import os
from pathlib import Path


DEFAULT_OVERRIDES = [
    # Model/data.
    'model=meta_llama3_8b_instruct',
    'datasets=[h4_ultrafeedback_binarized]',
    'loss=tidpo',
    'exp_name=meta_llama3_8b_instruct_h4_ultrafeedback_tidpo',

    # TI-DPO paper v3, Table B13.
    'loss.alpha=0.5',
    'loss.beta=0.1',
    'loss.gamma=0.1',
    'loss.lambda_importance=0.7',

    # README TIDPO defaults not specified by Table B13.
    'loss.use_tidpo=true',
    'loss.if_tdpo2=true',
    'loss.enable_gradient_attribution=true',
    'loss.prior_sigma_div=8.0',
    'loss.alpha_triplet=0.01',
    'loss.reference_free=false',
    'loss.anchor_max_new_tokens=16',
    'loss.anchor_top_k=50',
    'loss.anchor_top_p=0.95',
    'loss.anchor_temperature=0.8',
    'loss.kl_coef=0.0',

    # Training/runtime config.
    'trainer=FSDPTrainer',
    'model.policy_dtype=float16',
    'model.fsdp_policy_mp=null',
    'model.reference_dtype=float16',
    'optimizer=RMSprop',
    'lr=5e-6',
    'warmup_steps=150',
    'batch_size=64',
    'eval_batch_size=4',
    'gradient_accumulation_steps=32',
    'max_grad_norm=10.0',
    'n_epochs=1',
    'n_eval_examples=2000',
    'eval_every=12800',
    'max_length=768',
    'max_prompt_length=384',
    'activation_checkpointing=true',
    'fsdp_cpu_offload=false',
    'sample_during_eval=false',
    'do_first_eval=false',
    'seed=42',
    'wandb.enabled=false',
]


def main() -> int:
    repo_dir = Path(__file__).resolve().parent
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:32')
    command = [sys.executable, '-u', 'train.py', *DEFAULT_OVERRIDES, *sys.argv[1:]]
    print(' '.join(command))
    return subprocess.run(command, cwd=repo_dir).returncode


if __name__ == '__main__':
    raise SystemExit(main())
