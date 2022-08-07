#!/bin/bash
#SBATCH --mem-per-cpu=1.5G
#SBATCH --time=2:00:00

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements-cc.txt
python benchmark.py