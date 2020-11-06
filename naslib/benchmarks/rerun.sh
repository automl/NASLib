#!/usr/bin/bash

RETRIES="$3"

if [[ $# -ne 3 ]]; then
	echo "Need to specify config file, optimizer and retires"
	exit 1
fi

for r in $(seq 1 $RETRIES); do
    eval "sbatch ../scheduler.sh $1 $2 $r"
done