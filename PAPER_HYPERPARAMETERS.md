# TI-DPO Paper Hyperparameters

Source: *Token-Importance Guided Direct Preference Optimization*, ICLR 2026 (PDF in repo root).

## Table B12 — Final hyperparameter settings (page 21)

The paper's single, authoritative hyperparameter table is **Table B12** on **page 21**, in Appendix B.6. It lists only four values:

| Symbol | Name | Value | Page |
|---|---|---|---|
| α | TDPO KL Weight | 0.5 | p. 21 (Table B12) |
| β | DPO Temperature | 0.1 | p. 21 (Table B12) |
| γ | Triplet Loss Weight | 0.1 | p. 21 (Table B12) |
| λ | Hybrid Weight Mix (gradient attribution vs Gaussian prior) | 0.7 | p. 21 (Table B12) |

These map to the run-script overrides:
- `loss.alpha=0.5`
- `loss.beta=0.1`
- `loss.gamma=0.1`
- `loss.lambda_importance=0.7`

## Other quantitative settings mentioned in the paper

| Setting | Value | Page | Notes |
|---|---|---|---|
| Gaussian prior mean μ | (T−1)/2 | p. 20 (B.6) | T = sequence length |
| Gaussian prior std σ | T/4 | p. 20 (B.6) | Chosen so 4σ ≈ T covers ~95% of the sequence |
| Gradient-attribution target | max(L_{T−1}) | p. 5 (Eq. 5) | Max logit at the final position |
| Importance score | L1 norm of ∇_{e_i} L_target | p. 5 (Eq. 6) | Per-token sum over vocab gradient |

## Sensitivity-analysis ranges (not prescribed defaults, but reported tested values)

| Symbol | Tested values | Page |
|---|---|---|
| λ | 0, 0.1, 0.3, 0.5, **0.7**, 0.9, 1.0 (paper notes stable for λ ∈ [0.3, 0.7]) | p. 21 (Table B10) |
| α | 0.1, 0.2, **0.3**, 0.5 (Table B11 shows α=0.3 best on the small sweep, but Table B12 picks 0.5) | p. 21 (Table B11) |

## What the paper does NOT specify

The paper is silent on the following — they are **infrastructure choices**, not paper hyperparameters, and can be tuned freely without deviating from Table B12:

- Batch size, micro-batch size, gradient accumulation steps
- Learning rate, warmup steps, LR schedule, weight decay
- Optimizer (RMSprop / Adam / SGD / etc.)
- Sequence length (`max_length`), prompt length (`max_prompt_length`)
- Number of epochs (Algorithm 1 on p. 7 just says "for each epoch"; loss-curve plot in Table B9 on p. 20 runs to epoch 3.0, but no prescribed value)
- Triplet margin (`alpha_triplet`), KL coefficient (`kl_coef`), `prior_sigma_div` (Gaussian σ is fixed by formula above, not a tuned hyperparameter)
- Anchor-sampling settings (`anchor_max_new_tokens`, `anchor_top_k`, `anchor_top_p`, `anchor_temperature`)
- Mixed-precision policy, FSDP sharding strategy, CPU-offload, activation checkpointing
- Number/type of GPUs used

## Computational-cost note (page 23)

From Appendix D.1: TI-DPO costs approximately **2× standard DPO training time** because the hybrid weighting mechanism requires *one additional backward pass per sequence* for gradient attribution. Cost scales linearly with sequence length. Training-only overhead — inference unchanged.

## Codebase hyperparameter inventory

Default values shipped in this repo, with their source file/line. Run-script overrides in `run_meta_llama3_instruct_ultrafeedback.py` supersede these defaults.

### Loss hyperparameters — `config/loss/tidpo.yaml`

| Name | Default | Source | Maps to paper |
|---|---|---|---|
| `name` | `tidpo` | `config/loss/tidpo.yaml:2` | — |
| `use_tidpo` | `True` | `config/loss/tidpo.yaml:3` | — |
| `beta` | `0.2` | `config/loss/tidpo.yaml:4` | β (Table B12, p. 21) |
| `alpha` | `0.5` | `config/loss/tidpo.yaml:7` | α (Table B12, p. 21) |
| `if_tdpo2` | `true` | `config/loss/tidpo.yaml:8` | — |
| `gamma` | `0.001` | `config/loss/tidpo.yaml:10` | γ (Table B12, p. 21) |
| `alpha_triplet` | `0.001` | `config/loss/tidpo.yaml:11` | Triplet margin α_trp (Eq. 4, p. 4) |
| `reference_free` | `false` | `config/loss/tidpo.yaml:12` | — |
| `enable_gradient_attribution` | `True` | `config/loss/tidpo.yaml:13` | Toggle for Eq. 6 (p. 5) |
| `lambda_importance` | `0.2` | `config/loss/tidpo.yaml:16` | λ (Eq. 7, p. 5 / Table B12, p. 21) |
| `prior_sigma_div` | `8.0` | `config/loss/tidpo.yaml:17` | Sets σ of Gaussian prior (B.6, p. 20 — paper formula uses σ = T/4) |
| `anchor_max_new_tokens` | `64` | `config/loss/tidpo.yaml:20` | Anchor sampling (§4.3, p. 5) |
| `anchor_top_k` | `50` | `config/loss/tidpo.yaml:21` | Anchor sampling |
| `anchor_top_p` | `0.95` | `config/loss/tidpo.yaml:22` | Anchor sampling |
| `anchor_temperature` | `0.8` | `config/loss/tidpo.yaml:23` | Anchor sampling |
| `kl_coef` | `0.0` | `config/loss/tidpo.yaml:26` | Optional KL(ref ‖ policy) penalty |

These values are read with `getattr(self.config.loss, ...)` in `trainers.py`:
- `use_tidpo` — `trainers.py:650, 723`
- `alpha_triplet` — `trainers.py:680, 755, 884`
- `enable_gradient_attribution` — `trainers.py:1171`
- `lambda_importance` — `trainers.py:1172`
- `prior_sigma_div` — `trainers.py:1174`
- `anchor_*` — `trainers.py:903–905, 919, 949`
- `alpha`, `if_tdpo2` — `trainers.py:1299–1300`
- `gamma` — `trainers.py:1303`
- `kl_coef` — `trainers.py:713, 1339`

### Training/runtime hyperparameters — `config/config.yaml`

| Name | Default | Source | Paper? |
|---|---|---|---|
| `seed` | `0` | `config/config.yaml:2` | not specified |
| `batch_size` | `1` | `config/config.yaml:8` | not specified |
| `eval_batch_size` | `1` | `config/config.yaml:11` | not specified |
| `fsdp_cpu_offload` | `false` | `config/config.yaml:20` | not specified |
| `sample_during_eval` | `true` | `config/config.yaml:42` | not specified |
| `n_eval_model_samples` | `16` | `config/config.yaml:45` | not specified |
| `do_first_eval` | `true` | `config/config.yaml:48` | not specified |
| `lr` | `5e-6` | `config/config.yaml:54` | not specified |
| `gradient_accumulation_steps` | `2` | `config/config.yaml:59` | not specified |
| `max_grad_norm` | `10.0` | `config/config.yaml:62` | not specified |
| `max_length` | `256` | `config/config.yaml:65` | not specified |
| `max_prompt_length` | `128` | `config/config.yaml:68` | not specified |
| `n_epochs` | `1` | `config/config.yaml:71` | not specified (Table B9, p. 20 plots up to epoch 3) |
| `n_examples` | `null` | `config/config.yaml:74` | not specified |
| `n_eval_examples` | `64` | `config/config.yaml:77` | not specified |
| `trainer` | `BasicTrainer` | `config/config.yaml:80` | not specified |
| `optimizer` | `RMSprop` | `config/config.yaml:83` | not specified |
| `warmup_steps` | `150` | `config/config.yaml:86` | not specified |
| `activation_checkpointing` | `true` | `config/config.yaml:89` | not specified |
| `eval_every` | `10000` | `config/config.yaml:92` | not specified |

### Model hyperparameters — `config/model/meta_llama3_8b_instruct.yaml`

| Name | Default | Source |
|---|---|---|
| `name_or_path` | `meta-llama/Meta-Llama-3-8B-Instruct` | `config/model/meta_llama3_8b_instruct.yaml:1` |
| `tokenizer_name_or_path` | `null` | `config/model/meta_llama3_8b_instruct.yaml:2` |
| `block_name` | `LlamaDecoderLayer` | `config/model/meta_llama3_8b_instruct.yaml:4` |
| `use_auth_token` | `true` | `config/model/meta_llama3_8b_instruct.yaml:5` |
| `attn_implementation` | `null` | `config/model/meta_llama3_8b_instruct.yaml:6` |
| `policy_dtype` | `float32` | `config/model/meta_llama3_8b_instruct.yaml:8` |
| `fsdp_policy_mp` | `null` | `config/model/meta_llama3_8b_instruct.yaml:9` |
| `reference_dtype` | `float16` | `config/model/meta_llama3_8b_instruct.yaml:10` |

### Run-script overrides — `run_meta_llama3_instruct_ultrafeedback.py`

The active overrides currently applied (after the OOM tuning) live in the `DEFAULT_OVERRIDES` list at the top of the file:

| Override | Value | Notes |
|---|---|---|
| `loss.alpha` | `0.5` | **Table B12 (p. 21)** |
| `loss.beta` | `0.1` | **Table B12 (p. 21)** |
| `loss.gamma` | `0.1` | **Table B12 (p. 21)** |
| `loss.lambda_importance` | `0.7` | **Table B12 (p. 21)** |
| `loss.use_tidpo` | `true` | TI-DPO mode |
| `loss.if_tdpo2` | `true` | TDPO2-style position-wise KL |
| `loss.enable_gradient_attribution` | `true` | Eq. 6 |
| `loss.prior_sigma_div` | `8.0` | Gaussian σ control |
| `loss.alpha_triplet` | `0.01` | Triplet margin |
| `loss.kl_coef` | `0.0` | No extra KL |
| `loss.anchor_max_new_tokens` | `64` | Anchor length |
| `loss.anchor_top_k` | `50` | Anchor sampling |
| `loss.anchor_top_p` | `0.95` | Anchor sampling |
| `loss.anchor_temperature` | `0.8` | Anchor sampling |
| `trainer` | `FSDPTrainer` | 2× A6000 sharding |
| `optimizer` | `RMSprop` | infrastructure choice |
| `lr` | `5e-6` | infrastructure |
| `warmup_steps` | `150` | infrastructure |
| `batch_size` | `64` | infrastructure |
| `eval_batch_size` | `4` | infrastructure |
| `gradient_accumulation_steps` | `32` | gives micro=1 per GPU |
| `max_grad_norm` | `10.0` | infrastructure |
| `n_epochs` | `1` | infrastructure |
| `max_length` | `1024` | reduced from 2048 to fit A6000 |
| `max_prompt_length` | `512` | reduced from 1024 |
| `activation_checkpointing` | `true` | memory |
| `fsdp_cpu_offload` | `false` | host RAM constraint |
| `model.policy_dtype` | `float16` | overrides yaml default `float32` |
| `model.fsdp_policy_mp` | `float16` | FSDP mixed precision |
| `model.reference_dtype` | `float16` | matches yaml |
| `sample_during_eval` | `false` | skip generation in eval |
| `do_first_eval` | `false` | skip initial eval |
| `seed` | `42` | reproducibility |

## Bottom line for reproduction

Only the four values in Table B12 (α, β, γ, λ) are required to be set to the paper's choices. Anything else can be adjusted to fit available hardware without breaking paper fidelity.
