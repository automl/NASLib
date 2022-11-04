# !/bin/bash

for i in file{1};do
  for x in web{0..3};do
    echo "Copying $i to server $x"
    grep -a -w "Metric.TEST_ACCURACY" nas-fix/run/nasbench101/cifar10/darts/$x/log.log
  done
done