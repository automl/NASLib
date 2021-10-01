search_space=$1

dataset_dir="/home/mehtay/research/NASLib/naslib/benchmarks/bbo/configs/$search_space/*"

for optimizer_dir in $dataset_dir/*
do 
	# echo $config_dir	
	for seed_dir in $optimizer_dir/*
	do
		for config_file in $seed_dir/*
		do
			echo submitted ${config_file}
			# sbatch submit_job.sh $config_file
		done
	done
done
