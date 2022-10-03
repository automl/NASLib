cd naslib/data
search_space="$1"
echo search_space = $search_space
if [ "$search_space" = "tnb101" ] || [ "$search_space" = "all" ]
then
   filepath1=zc_transbench101_macro.json
   filepath2=zc_transbench101_micro.json
   fileid1=1nMaa3LjlP1d_umgudX7abKSdDMeoBNb5
   fileid2=1i8N2n7yflN33xAuQVzlTYM4E7W1Cwn1S
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
   gdown 11kIyLr7LwFB_fXGDk3ic5yjiYMY5JLXG
  fi
fi
if [ "$search_space" = "nb201" ] || [ "$search_space" = "all" ]
then
   filepath=zc_nasbench201.json
  if [ -f $filepath ]
  then
    echo "nb201 file exist"
  else
   gdown 1k2EUtVJ4JqoJCnuyJEVgZs6vAmbg6XVB
  fi
fi
if [ "$search_space" = "nb101" ] || [ "$search_space" = "all" ]
then
   filepath=zc_nasbench101.json
  if [ -f $filepath ]
  then
    echo "nb101 file exist"
  else
   gdown 1uT3tuIDMVaB4U1N8l9imEYHWPOLls3FD
  fi
fi
cd ..
cd ..
