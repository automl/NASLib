# !/bin/bash

for i in file{1};do
  for x in {0..5};do
    echo "Copying $i to server $x"
    grep -a -w "Metric.TEST_ACCURACY" nas-fix/run/nasbench201/cifar10/drnas/$x/log.log
  done
done