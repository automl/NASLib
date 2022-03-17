#!/bin/bash
num_nodes=21
num_configs=1000
#testOne=$(( ($num_configs + ($stepsize - 1)) / $step_size))
step_size=$(( ($num_configs + ($num_nodes - 1)) / $num_nodes))
echo Stepsize: $step_size
for ((i=0;i<=num_configs;i=i+step_size)); do
    echo ./submit_configs.sh $i $(( $i + $step_size))
    echo $i
done