# Decision Diffuser for Legged Robots


## Prerequisite
MuJoCo가 설치되어있어 ~/.mujoco 폴더가 있어야 빌드가 됨 ( 추후 mujoco dependency 제거할 예정 )

## Conda Env Setting
### Option 1) Shell script 실행하기 
start.sh 들어가서 변수 세팅 하기

	ENV_NAME=decisionDiffuser
	path_to_isaacgym=/home/kdyun/isaacgym
	path_to_walktheseways=/home/kdyun/Desktop/walk-these-ways
그 후 터미널에서 shell script 실행

	bash start.sh

### Option 2) 직접 설치 (에러가 날 경우)
#### 1. create environment

    conda env create -n ${ENV_NAME} python=3.8
    conda activate ${ENV_NAME}

#### 2. install pip dependencies
train.py를 실행하면서 생기는 에러에 따라 pip dependencies를 설치하는 방법을 권장함

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

주의사항 : 
1) pip install gym을 실행할 때는 버전을 지정하지 말고, 이를 설치한 이후에 d4rl 데이터셋 패키지를 설치해야 함
2) c_error_callback이 발생할 경우 pip install “cython<3”을 실행해주어야 함

#### 3. external libraries
#### raisim
conda environment를 켜고 raisimpy 옵션을 켠 cmake를 실행한다

이후 train.py를 실행해보면서 없다고 나오는 raisim 관련 so 파일들을 conda env의 lib에 넣어준다
#### go1_gym
isaacgym pip, go1_gym pip를 설치

	cd ${path_to_isaacgym}/python && pip install -e .
 	cd ${path_to_walktheseways} && pip install -e .

## Bashrc setting
	export PYTHONPATH=$PYTHONPATH:${path_to_decision_diffuser}


## Run
	    cd analysis && python train.py
## TroubleShooting
...