#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=150
step=5e-2
wvecDim=40
memDim=40
rho=1e-5

model="TreeLSTM"
label="author"

######################################################## 
# Probably a good idea to let items below here be
########################################################

outfile="models/${model}_${label}_wvecDim_${wvecDim}_memDim_${memDim}_step_${step}_epochs_${epochs}_rho_${rho}_droproot.bin"

echo $outfile

python runNNet.py --step $step --epochs $epochs --outFile $outfile --wvecDim $wvecDim --memDim $memDim --model $model --rho $rho --label $label --minibatch 20 --output_dim 5
read
