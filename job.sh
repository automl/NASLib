#!/bin/bash
conda deactivate
conda activate New-NASLib
export PYTHONPATH=.
source scripts/bash_scripts/run_nb201.sh
#source scripts/bash_scripts/run_darts.sh
#source scripts/bash_scripts/run_tnb101.sh