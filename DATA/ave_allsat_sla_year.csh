#!/bin/csh
# ===
# === Step-1: Check  variables, echo history and usage
# === sla anomaly for 2010 - ave(1993---2010) 
# === sla anomaly for 2010 - 2009 

  if (${#} == 0) then
    echo 'Usage '
    echo ' run.csh year month'
    echo ' For example '
    echo ' run.csh 2010 07 '
  exit
  endif

  set year = $1
  set mete_root = /public/wind_flow/flow/WGS/Bulletin/Monthly_Bulletin/download/ssh/ssh2/new_metedate

  mkdir temp
  rm -f temp/*
  rm -f year_ave_nrt_global_allsat_msla_h_*.nc cha_year_ave_nrt_global_allsat_msla_h_y${year}-*.nc
  
  #ln -s $mete_root/nrt_global_allsat_msla_h_${year}${month}*.nc ./temp
  #cd temp
  set num = 1
  foreach file ( `ls *global_allsat_msla_h_${year}*nc` )
   echo $file' '$num
   ncks -v sla $file ww_$num.nc 
   set num = `expr $num + 1`
  end
  foreach file ( `ls *global_allsat_msla_h_y${year}*nc` )
   echo $file' '$num
   ncks -v sla $file ww_$num.nc 
   set num = `expr $num + 1`
  end
  ncea ww_*nc year_ave_nrt_global_allsat_msla_h_${year}.nc

  # ====== cha ====  
  set year2 = `expr $year - 1`
  set dt_root = /public/wind_flow/flow/WGS/Bulletin/Monthly_Bulletin/download/ssh/dt_monthly_mean

  ln -s ${dt_root}/dt_global_allsat_msla_h_y${year2}_m*.nc ./

  set num = 1
  foreach file ( `ls dt_global_allsat_msla_h_y${year2}_m*nc` )
   echo $file' '$num
   ncks -v sla $file ss_$num.nc 
   set num = `expr $num + 1`
  end
  ncea ss_*nc year_ave_nrt_global_allsat_msla_h_${year2}.nc

  ncbo --op_typ=- year_ave_nrt_global_allsat_msla_h_${year}.nc year_ave_nrt_global_allsat_msla_h_${year2}.nc  cha_year_ave_nrt_global_allsat_msla_h_y${year}-${year2}.nc
  
  rm -f ww_*.nc ss_*.nc
  rm -f dt_global_allsat_msla_h_y${year2}_m*.nc
