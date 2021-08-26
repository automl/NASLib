search_space=$1

config_dirs="/home/mehtay/research/NASLib/naslib/benchmarks/bbo/configs/$search_space/*"

for config_dir in $config_dirs
do 
	# echo $config_dir	
	for config_file in $config_dir/*
	do 
	echo submitted ${config_file}
	sbatch submit_job.sh $config_file
	done
done
