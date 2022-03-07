search_space=$1
bosch_partition=$2
dataset_dir="/home/mehtay/research/NASLib/naslib/configs/bbo/configs_cpu/$search_space/*"

for optimizer_dir in $dataset_dir/*
do 
	# echo $config_dir	
	for config_dir in $optimizer_dir/*
	do
		echo starting to run ${config_dir} across 10 seeds ...
		if [ $bosch_partition == 'gpu' ] 
		then
			sbatch --bosch submit_boschgpu_folder.sh $config_dir
		fi

		if [ $bosch_partition == 'cpu' ] 
		then
			sbatch --bosch submit_bosch_folder.sh $config_dir
		fi
	done
done