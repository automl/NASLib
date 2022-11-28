cd naslib/data
search_space="$1"
echo search_space = $search_space
if [ "$search_space" = "tnb101" ] || [ "$search_space" = "all" ]
then
   filepath1=zc_transbench101_macro.json
   filepath2=zc_transbench101_micro.json
   fileid1=1teH8JcQsamZngUD_DMQyNkCoUYYSTM0M
   fileid2=1SBOVAyhLCBTAJiU_fo7hLRknNrGNqFk7
   if [ -f $filepath1 ] && [ -f $filepath2]
   then
      echo "tnb101 files exist"
   else
      echo "tnb101 files no exist"
      gdown $fileid1
      gdown $fileid2
   fi
fi
if [ "$search_space" = "nb301" ] || [ "$search_space" = "all" ]
then
   filepath=zc_nasbench301.json
  if [ -f $filepath ]
  then
    echo "nb301 file exist"
  else
   gdown 1RddgmwqjWJ1czGT8gEPB8qqhUHazp92G
  fi
fi
if [ "$search_space" = "nb201" ] || [ "$search_space" = "all" ]
then
   filepath=zc_nasbench201.json
  if [ -f $filepath ]
  then
    echo "nb201 file exist"
  else
   gdown 1R7n7GpFHAjUZpPISzbhxH0QjubnvZM5H
  fi
fi
if [ "$search_space" = "nb101" ] || [ "$search_space" = "all" ]
then
   filepath=zc_nasbench101.json
  if [ -f $filepath ]
  then
    echo "nb101 file exist"
  else
   gdown 1Rkse44EWgYdBS34iyhjSs9Y2l0fxPCpU
  fi
fi
cd ..
cd ..
