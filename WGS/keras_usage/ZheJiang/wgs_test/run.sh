#!/bin/sh
rm -f MSG
  epoch=10000
  while [ $epoch -lt 300000 ]
  do
  size=2
  while [ $size -lt 24 ]
  do
      sed 's/wgs_batch_size/'$size'/g ' single.py > 1wgs_single.py 
      sed 's/wgs_nb_epoch/'$epoch'/g '  1wgs_single.py > wgs_single.py 
      echo '======= 'batch_size'  '${size}'======= ' >> MSG
      echo '======= 'nb_epoch'  '${epoch} '======= ' >> MSG
      python wgs_single.py >> MSG
      echo '======================' >> MSG
  size=`expr $size + 2`
  done 
  epoch=`expr $epoch + 10000`
  done 
