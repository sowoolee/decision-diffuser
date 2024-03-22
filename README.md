# Decision Diffuser Environment Setting

## Prerequisite
MuJoCo가 설치되어있어 ~/.mujoco 폴더가 있어야 빌드가 됨

## Conda Env Setting
### 1. create environment

envrionment.yml을 이용하여 가상환경을 만드나, 아래와 같이 
대부분의 pip dependencies들을 주석처리하고 manual하게 설치하는 것을 권장함

	    name: decidiff
	    channels:
		- defaults
		- conda-forge
	    dependencies:
		- python=3.8
		- pip
		- patchelf
		- pip:
		    - numpy
		    - matplotlib==3.3.4

위와 같이 수정한 이후 가상환경을 활성화하고,

    conda env create -f environment.yml
    conda activate decidiff
    export PYTHONPATH=/path/to/decision-diffuser/

train.py를 실행하면서 생기는 에러에 따라 pip dependencies를 설치하는 방법을 권장함

    cd analysis && python train.py

### 2. install pip dependencies
에러 발생 순서에 따라 설치하면 대략적으로 아래와 같음, 버전은 yml 참고

주의사항 : 
1) pip install gym을 실행할 때는 버전을 지정하지 말고, 이를 설치한 이후에 d4rl 데이터셋 패키지를 설치해야 함
2) c_error_callback이 발생할 경우 pip install “cython<3”을 실행해주어야 함

		pip install ml_logger
		pip install params_proto
		pip install jaynes
		pip install gym
		pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
		pip install typed-argument-parser
		pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
		pip install gitpython
		pip install scikit-video
		pip install scikit-image
		pip install einops
		pip install "cython<3"
		pip install pandas
		pip install wandb
		pip install flax
		pip install jax
		pip install ray
		pip install crcmod

### 3. external libraries
#### raisim
conda environment를 켜고 raisimpy 옵션을 켠 cmake를 실행한다

이후 train.py를 실행해보면서 없다고 나오는 raisim 관련 so 파일들을 conda env의 lib에 넣어준다
#### go1_gym
torch를 요구하는 버전에 맞춰 설치해준다

	pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

그 다음에 isaacgym pip, go1_gym pip를 설치해준다

	cd isaacgym/python && pip install -e .
 	cd wtw && pip install -e .
