#!/bin/bash

optimizers=("darts" "edge_popup" "drnas" "gdas")
#mkdir moz

for space in {1..3};
do
  space=2
  for opt in "${optimizers[@]}";
  do
    for i in {1..5};
    do
#      file=/home/shakiba/Documents/NASLib-fix/NASLib/naslib_results/nasbench"$space"01/cifar10/$opt/$i/log.log
      file=/home/shakiba/Documents/NASLib-fix/NASLib/naslib_results/nasbench"$space"01_cifar10-valid_"$opt"_"$i"
      log=$(grep -a -w '01' $file | grep -a -w 'Train accuracy (top1, top5):')
      (
      echo "month,day,hour,minute,second,epoch,1,5,train_top1,train_top5,val_top1,val_top5"
      while IFS= read -r line; do
        echo $line | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | tr '\n' ',' | sed 's/.$//'; echo ""
      done <<< "$log"
      ) > moz/nasbench"$space"01_cifar10-valid_"$opt"_"$i"
    done
  done
  break
done

