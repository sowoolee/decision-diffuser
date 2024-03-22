# Decision Diffuser Environment Setting


## Conda Env Setting

envrionment.yml pip dependencies

    conda env create -f environment.yml
    
then install pip dependencies manually

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

	pip install 
	
    conda activate decidiff
    export PYTHONPATH=/path/to/decision-diffuser/
    

