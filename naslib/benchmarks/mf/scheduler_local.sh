search_space=$1

# dataset_dir="/Users/lars/Projects/NASLib/naslib/benchmarks/bbo/naslib/benchmarks/bbo/configs_cpu/$search_space/*"
dataset_dir="/Users/lars/Projects/NASLib/naslib/benchmarks/bbo/configs_m1/$search_space/*"
 
#for optimizer_dir in $dataset_dir/*
# do 
	# echo $config_dir	
	# for config_dir in $optimizer_dir/*
	# do
		# echo starting to run ${config_dir} across 10 seeds ...
		# sbatch submit_folder.sh $config_dir
		
	# done
# done

# for running default config files separately
for optimizer_dir in $dataset_dir/*
do 
	echo starting to run $optimizer_dir/config_0 across x seeds ...
	./submit_folder_local.sh $optimizer_dir/config_0 
done
