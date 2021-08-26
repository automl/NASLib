export OMP_NUM_THREADS=2
optimizers=(rs)
# optimizers=(re)
# optimizers=(ls)
# optimizers=(npenas)
#optimizers=(bananas)

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
out_dir=run

# search space / data:
search_space=nasbench101
dataset=cifar10

fidelity=108
epochs=15
predictor_type=bananas

# trials / seeds:
trials=10
end_seed=$(($start_seed + $trials - 1))

# create config files
for i in $(seq 0 $((${#optimizers[@]}-1)) )
do
  optimizer=${optimizers[$i]}
  python $base_file/benchmarks/create_configs.py \
 --start_seed $start_seed --trials $trials \
  --out_dir $out_dir --dataset=$dataset --config_type bbo \
  --search_space $search_space --optimizer $optimizer \
  --acq_fn_optimization $acq_fn_optimization --predictor_type $predictor_type \
  --fidelity $fidelity --epochs $epochs
done