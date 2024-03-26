#!/bin/bash
##Setting
ENV_NAME=decisionDiffuser
path_to_isaacgym=/home/kdyun/isaacgym
path_to_walktheseways=/home/kdyun/Desktop/walk-these-ways

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# setup conda
CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# deactivate the env, if it is active
conda deactivate

# Remove existing environment if exists
conda env remove -n "${ENV_NAME}" || { echo "Failed to remove existing environment"; exit 1; }

# Create a new conda environment
conda create -y -n "${ENV_NAME}" python=3.8 || { echo "Failed to create new environment"; exit 1; }

# Activate the newly created environment
conda activate "${ENV_NAME}" || { echo "Failed to activate new environment"; exit 1; }

conda info



##package install
# double check that the correct env is active
ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
if [ "${ENV_NAME}" != "${ACTIVE_ENV_NAME}" ]; then
	echo "*** Env is not active, aborting"
	exit 1
fi
pip install ml_logger==0.8.69
pip install params_proto==2.9.6 # walk these ways에 더 최신 버전이 있음
pip install jaynes==0.8.11
pip install gym
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install typed-argument-parser
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
pip install gitpython
pip install scikit-video==1.1.11
pip install scikit-image==0.17.2
pip install einops
pip install tensorboard
pip install adamp
pip install "cython<3"
pip install pandas
pip install wandb
pip install flax== 0.3.5
pip install jax<= 0.2.21
pip install ray==1.9.1
pip install crcmod
echo

# install external packages (isaac gym & walk these ways)
cd ${path_to_isaacgym}/python && pip install -e .
cd ${path_to_walktheseways} && pip install -e .

conda info|grep active
echo "========shell script Finished======="

