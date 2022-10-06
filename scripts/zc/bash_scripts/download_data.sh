cd naslib/data
search_space="$1"
dataset="$2"
echo dataset = $dataset
echo search_space = $search_space
tnb_datasets=(benevolence forkland merom)
if [ "$search_space" = "tnb101" ] || [ "$search_space" = "all" ]
then
   cd taskonomydata_mini	
   if [ "$dataset" = "jigsaw" ] || [ "$search_space" = "all" ] || [ "$dataset" = "all" ]
   then
      for dataset_base in ${tnb_datasets[@]}
      do
        file=$dataset_base\_rgb.tar
        filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/rgb/$file
        echo $filepath
	cd $dataset_base
        if [ -d "rgb" ]
        then
           echo rgb exists
        else
           echo rgb does not exist
           wget $filepath
           tar -xvf $file
           rm $file
        fi
        cd ..
      done
   fi
   if [ "$dataset" = "class_scene" ] || [ "$search_space" = "all" ] || [ "$dataset" = "all" ]
   then 
      for dataset_base in ${tnb_datasets[@]}
      do
       file=$dataset_base\_class_scene.tar
       filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/class_scene/$file
       echo $filepath
       cd $dataset_base
       if [ -d "class_scene" ]
       then
          echo class_scene exists
       else
          echo class_scene does not exist
          wget $filepath
          tar -xvf $file
          rm $file
       fi
       cd ..
      done
      for dataset_base in ${tnb_datasets[@]}
      do
       echo starting $dataset_base
       for j in $dataset_base/class_scene/*class_places.npy
       do
        #echo "$j"
        #echo "${j%class_places.npy}class_scene.npy"
        mv -- "$j" "${j%class_places.npy}class_scene.npy"
       done
       done
   fi
   if [ "$dataset" = "class_object" ] || [ "$search_space" = "all" ] || [ "$dataset" = "all" ]
   then
      for dataset_base in ${tnb_datasets[@]}
      do
       file=$dataset_base\_class_object.tar
       filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/class_object/$file
       echo $filepath
       cd $dataset_base
       if [ -d "class_object" ]
       then
          echo class_object exists
       else
          echo class_object does not exist
          wget $filepath
          tar -xvf $file
          rm $file
       fi
       cd ..
      done
   fi
   cd ..
fi
if [ "$search_space" = "nb301" ] || [ "$search_space" = "all" ]
then
  if [ -e nb_models ]
  then
    echo "nb301 models exist"
  else
   wget https://figshare.com/ndownloader/files/24992018
   unzip 24992018
   mv nb_models nb_models_1.0
  fi
  if [ -e nb301_full_training.pickle ]
  then
    echo "nb301 full training pickle exists"
  else
    gdown 1YJ80Twt9g8Gaf8mMgzK-f5hWaVFPlECF
  fi
fi
if [ "$search_space" = "nb201" ] || [ "$search_space" = "all" ]
then
  if [ "$dataset" = "cifar10" ] || [ "$search_space" = "all" ] || [ "$dataset" = "all" ]
  then
     if [ -e nb201_cifar10_full_training.pickle ] 
     then
        echo "cifar10 exists"
     else
        gdown 1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ
     fi
  fi
  if [ "$dataset" = "cifar100" ] || [ "$search_space" = "all" ] || [ "$dataset" = "all" ]
  then
     if [ -e nb201_cifar100_full_training.pickle ]
     then
        echo "cifar100 exists"
     else
        gdown 1hV6-mCUKInIK1iqZ0jfBkcKaFmftlBtp
     fi
  fi
  if [ "$dataset" = "Imagenet16-120" ] || [ "$search_space" = "all" ] || [ "$dataset" = "all" ]
  then
     if [ -e nb201_ImageNet16_full_training.pickle ]
     then
        echo "Imagenet16 exists"
     else
        gdown 1FVCn54aQwD6X6NazaIZ_yjhj47mOGdIH
     fi
     if [ -e ImageNet16-120 ]
     then
       echo "Imagenet16 data exits"
     else
       mkdir ImageNet16-120
       cd ImageNet16-120
       done="False"

       drive_ids=('1qd9Fkg7MdIe3MMbHtIJC8eZ8OsWcqPYA'
       '1pQBJ9exwpSG2E7m6aVvcOlJBGRhVhtg9'
       '175we9AOjnGam0j4sG5Vn0SHFyBvyv2Ia'
       '1FNBkOavsAP6Hvi7-41yLZwojdWfPub-R'
       '1HujB1GyiBjrSdAA0he5kZtkO9WEDAwCn'
       '1_vaYBQpbP6bx-G0_EiNysohqOJBJ_Ept'
       '1JwQk4TE21KqvrfvnOfVcdqnEB32ULWTr'
       '1T00JaN09RlNZPod8dQnF_Xdz0BWtjCWr'
       '1fB2JYSZRfd8uKfKLBO9P3mn9HWIWtMOH'
       '19Qvrqt-wi0UOCZBwI_Jw6-bLIXCYcPyl'
       )

       for idx in ${!drive_ids[@]}
       do
         done="False"
         while [ $done = "False" ]
         do
           echo $idx ${drive_ids[idx]}
           gdown ${drive_ids[idx]}
           idx_plus_1=`expr $idx + 1`
           echo $idx_plus_1 train_data_batch_${idx_plus_1}
           if [ -e train_data_batch_${idx_plus_1} ]
           then
             done="True"
           else
	     echo "Waiting 30 seconds to try again..."
	     sleep 30
	   fi
         done
       done

       done="False"
       while [ $done = "False" ]
       do
         gdown 1LQNICeSrwwE2KdDxc9Z9FXmi7N4HKUsA
         if [ -e val_data ]
         then
           done="True"
         else
	   echo "Waiting 30 seconds to try again..."
	   sleep 30
	 fi
       done
       cd ..
     fi
  fi
fi

cd ..
cd ..
