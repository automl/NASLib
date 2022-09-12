#!/bin/bash

id=15026309
while [[ $id -lt 15026439 ]]
do
    scancel $id
    ((id=id+1))

done
