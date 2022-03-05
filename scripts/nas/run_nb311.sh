export PYTHONPATH=$HOME/nasbench311/nasbench301:$HOME/nasbench311/NASLib:$PYTHONPATH
export OMP_NUM_THREADS=2
#optimizers=(rs)
optimizers=(re)
#optimizers=(rea_lce)
#optimizers=(rea_svr)
#optimizers=(ls)
#optimizers=(ls_lce)
#optimizers=(ls_svr)
#optimizers=(bananas)
#optimizers=(bananas_svr)
#optimizers=(bananas_lce)
#optimizers=(hb_simple)
#optimizers=(bohb_simple)
#optimizers=(dehb_simple)
predictor=bananas #optional: (gcn xgb)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

if [[ $optimizers == bananas* ]]
then
  acq_fn_optimization=mutation
else
  acq_fn_optimization=random_sampling
fi

# folders:
base_file=naslib
s3_folder=nas301
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=darts
dataset=cifar10
budgets=5000000
fidelity=97
single_fidelity=20

# trials / seeds:
trials=30
end_seed=$(($start_seed + $trials - 1))

# create config files
for i in $(seq 0 $((${#optimizers[@]}-1)) )
do
  optimizer=${optimizers[$i]}
  python $base_file/benchmarks/create_configs.py \
  --budgets $budgets --start_seed $start_seed --trials $trials \
  --out_dir $out_dir --dataset=$dataset --config_type nas \
  --search_space $search_space --optimizer $optimizer \
  --acq_fn_optimization $acq_fn_optimization --predictor $predictor \
  --fidelity $fidelity --single_fidelity $single_fidelity
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
  for optimizer in ${optimizers[@]}
    do
      if [[ $optimizer == bananas* ]]
      then
        config_file=$out_dir/$dataset/configs/nas/config\_$optimizer\_$predictor\_$t.yaml
      else
        config_file=$out_dir/$dataset/configs/nas/config\_$optimizer\_$t.yaml
      fi
      echo ================running $optimizer trial: $t =====================
      python $base_file/benchmarks/nas/runner.py --config-file $config_file
    done
done
