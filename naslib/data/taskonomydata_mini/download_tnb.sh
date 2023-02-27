# this downloads and renames all the taskonomydata_mini data for TransNAS-Bench-101

datasets=(wainscott tolstoy klickitat pinesdale stockman beechwood coffeen corozal \
benevolence eagan forkland hanson hiteman ihlen lakeville lindenwood \
marstons merom newfields pomaria shelbyville uvalda)

# download all rgb files
for dataset in ${datasets[@]}
do
    file=$dataset\_rgb.tar
    filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/rgb/$file
    echo $filepath
    cd $dataset
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

# download all class_object files
for dataset in ${datasets[@]}
do
    file=$dataset\_class_object.tar
    filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/class_object/$file
    echo $filepath
    cd $dataset
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

# download all class_scene files
for dataset in ${datasets[@]}
do
    file=$dataset\_class_scene.tar
    filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/class_scene/$file
    echo $filepath
    cd $dataset
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

# rename all class_places.npy to class_scene.npy
for dataset in ${datasets[@]}
do
    echo starting $dataset
    for j in $dataset/class_scene/*class_places.npy
    do
        #echo "$j"
        #echo "${j%class_places.npy}class_scene.npy"
        mv -- "$j" "${j%class_places.npy}class_scene.npy"
    done
done

# download all normal files
for dataset in ${datasets[@]}
do
    file=$dataset\_normal.tar
    filepath=http://downloads.cs.stanford.edu/downloads/taskonomy_data/normal/$file
    echo $filepath
    cd $dataset
    if [ -d "normal" ]
    then
        echo normal exists
    else
        echo normal does not exist
        wget $filepath
        tar -xvf $file
        rm $file
    fi
    cd ..
done