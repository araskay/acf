#!/bin/bash

#SBATCH -c 1
#SBATCH --mem=16g
#SBATCH -t 8:0:0
#SBATCH -o __run_acf.o
#SBATCH -e __run_acf.e

function run_acf {

module load anaconda/3.5.3
module load afni
module load anaconda/3.5.3

python ~/code/acf/acf.py -f $1 --ndiscard $2

}

run_acf $1 $2

