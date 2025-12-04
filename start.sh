#script for ugent HPC
module load vsc-venv
source vsc-venv --activate --requirements requirements.txt --modules modules.txt
HF_HOME=./hf_home python pacman.py