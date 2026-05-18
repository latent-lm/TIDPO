#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-tidpo}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RUN_TESTS=1
RUN_EXAMPLE=0
RUN_TRAIN=0

usage() {
  cat <<'EOF'
Usage: bash setup_tidpo_conda.sh [options]

Creates a conda environment for TIDPO, installs requirements, runs repository
environment setup, and verifies the installation.

Options:
  --env-name NAME        Conda environment name. Default: tidpo
  --python VERSION      Python version for the conda env. Default: 3.10
  --skip-tests          Do not run smoke tests after installation
  --run-example         Run run_tidpo_example.py after setup
  --run-train           Run the README manual TIDPO training command after setup
  -h, --help            Show this help

Environment overrides:
  ENV_NAME=tidpo PYTHON_VERSION=3.10 bash setup_tidpo_conda.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --skip-tests)
      RUN_TESTS=0
      shift
      ;;
    --run-example)
      RUN_EXAMPLE=1
      shift
      ;;
    --run-train)
      RUN_TRAIN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Install Miniconda/Anaconda or load conda first." >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Using existing conda environment: $ENV_NAME"
else
  echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python setup_environment.py

python -c "import gradient_attribution; print('Installation successful')"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  python test_gradient_attribution.py
  python test_tidpo.py
fi

if [[ "$RUN_EXAMPLE" -eq 1 ]]; then
  python run_tidpo_example.py
fi

if [[ "$RUN_TRAIN" -eq 1 ]]; then
  python -u train.py \
    model=gpt2_small \
    datasets=[hh] \
    loss=tidpo \
    exp_name=my_experiment \
    batch_size=4 \
    eval_batch_size=4 \
    n_epochs=1 \
    lr=1e-5 \
    max_length=256 \
    max_prompt_length=128 \
    gradient_accumulation_steps=1 \
    activation_checkpointing=true
fi
