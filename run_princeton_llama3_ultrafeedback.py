#!/usr/bin/env python3
"""Run TI-DPO on Princeton Llama-3-Base-8B-SFT with Princeton Llama3 UltraFeedback."""

import subprocess
import sys
from pathlib import Path


DEFAULT_OVERRIDES = [
    'model=princeton_llama3_8b_sft',
    'datasets=[princeton_llama3_ultrafeedback]',
    'loss=tidpo',
    'exp_name=princeton_llama3_8b_sft_ultrafeedback_tidpo',
    'trainer=FSDPTrainer',
    'batch_size=8',
    'eval_batch_size=4',
    'n_epochs=1',
    'n_eval_examples=1961',
    'eval_every=10000',
    'max_length=2048',
    'max_prompt_length=1024',
    'gradient_accumulation_steps=1',
    'activation_checkpointing=true',
    'sample_during_eval=false',
    'do_first_eval=false',
    'wandb.enabled=false',
]


def main() -> int:
    repo_dir = Path(__file__).resolve().parent
    command = [sys.executable, '-u', 'train.py', *DEFAULT_OVERRIDES, *sys.argv[1:]]
    print(' '.join(command))
    return subprocess.run(command, cwd=repo_dir).returncode


if __name__ == '__main__':
    raise SystemExit(main())
