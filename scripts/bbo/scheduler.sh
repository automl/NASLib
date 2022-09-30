search_space=$1

dataset_dir="/home/robertsj/NASLib/naslib/configs/bbo/configs_cpu/$search_space/*"
 
for optimizer_dir in $dataset_dir/*
do 
	echo $config_dir	
	for config_dir in $optimizer_dir/*
	do
		echo starting to run ${config_dir} across 10 seeds ...
		# sbatch submit_folder.sh $config_dir
		bash submit_folder.sh $config_dir # for srun node testing
		
	done
done

# for running default config files separately
# for optimizer_dir in $dataset_dir/*
# do 
	# echo starting to run $optimizer_dir/config_0 across 10 seeds ...
	# sbatch submit_folder.sh $optimizer_dir/config_0 
	
# done
