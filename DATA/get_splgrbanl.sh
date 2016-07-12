#!/bin/sh
  . /etc/profile
  . ~/.bashrc
  unalias -a
  current_root=`pwd`
  cd $current_root
# ---
# ---  Purpose:
# ---    Extract a small domain from a GRIB2 file
# ---    Interpolate onto latlon grids
# ---
# ---    wgrib2 with V1.9 above is needed!
# ---    
# ------------------------------------------------
# ---  History:
# ---    Dr. GAO Shanhong, 16 Feb 2011.
# ---
# ------------------------------------------------

  # --- set domian and resolution
  #lon_beg=120
  #lon_end=125
  #lat_beg=38
  #lat_end=39
  lon_beg=110
  lon_end=135
  lat_beg=25
  lat_end=50
  res=0.25

  max_spd=0      
  int_hour=1
  out_all=1
  if_plot=0 # --- need out_all=1
  plot_nice=0 # --- need if_plot=1
  ifort=pgf90
  # ======= End of YOur Mofifications =============

  if [ $# -eq 0  ] ; then
     echo ' '
     echo ' Usage:'
     echo '   get_cfs_uv.sh wgrib2_file '
     echo ' '
     exit
  fi
  # --- get variables
  wgrib2_file=$1
  input_var=$2
  echo $wgrib2_file
  # ============
  
  beg_day=`echo $input_file | cut -c1-8`
  beg_hour=`echo $input_file | cut -c9-10`
  output_file=` wgrib2 $wgrib2_file | sed -n 1p | sed 's/:/ /g' | sed 's/=/ /g' |awk '{print $4}' `
  fyear=`echo $output_file | cut -c1-4`
  fmonth=`echo $output_file | cut -c5-6`
  fdate=`echo $output_file | cut -c1-8`
  ftime=`echo $output_file | cut -c9-10`
  
  LAT1=` echo $lat_beg | awk '{ print $1 * 1000 }' `
  LAT2=` echo $lat_end | awk '{ print $1 * 1000 }' `
  LON1=` echo $lon_beg | awk '{ print $1 * 1000 }' `
  LON2=` echo $lon_end | awk '{ print $1 * 1000 }' `
  DLAT=` echo $res | awk '{ print $1 * 1000 }' `
  DLON=` echo $res | awk '{ print $1 * 1000 }' `
  DIS_LAT=` expr $LAT2 - $LAT1 `
  DIS_LON=` expr $LON2 - $LON1 `
  NX=` expr $DIS_LON \/ $DLON + 1`
  NY=` expr $DIS_LAT \/ $DLAT + 1`
  
  echo $output_file $NX $NY 
  echo "=================="
  # ------
  #wgrib2 $wgrib2_file | grep $input_var | wgrib2 -i $wgrib2_file -grib_out temp_file.grib2 > msg
  # --- do interpolation
  #wgrib2 temp_file.grib2 -new_grid_winds earth -new_grid_interpolation bilinear \
  wgrib2 $wgrib2_file -new_grid_winds earth -new_grid_interpolation bilinear \
          -new_grid latlon ${lon_beg}:${NX}:${res} ${lat_beg}:${NY}:${res} \
           ${output_file}.grib2 >> msg
  wgrib2 ${output_file}.grib2 -csv msg.csv
  #cat msg.csv | sed -n 1p | sed 's/,/ /g' | awk '{print $5" "$9" "$10" " $11}' > ${output_file}.csv
   grep TMP msg.csv | sed 's/,/ /g' | awk '{print $9" "$10" " $11}' > 11
   grep HGT msg.csv | sed 's/,/ /g' | awk '{print $11}' > 22
  grep UGRD msg.csv | sed 's/,/ /g' | awk '{print $11}' > 33
  grep VGRD msg.csv | sed 's/,/ /g' | awk '{print $11}' > 44
  #grep PRES msg.csv | sed 's/,/ /g' | awk '{print $11}' > 55
  paste 11 22 33 44 > ${output_file}.csv
exit
  # --- save the dimension
  if test -e dimension.dat ; then
     rm -f dimension.dat
  fi
  echo $NX $NY $lon_beg $lon_end $lat_beg $lat_end $res > dimension.dat
  # -------------------------------------
  # ----- output all data   -------------
  if [ ${out_all} -ge 1 ] ; then
    wgrib2 ${output_file}.grib2 -text sfc_wind >> msg
    if test ! -e functions/convert_winds.exe ; then
       $ifort functions/convert_winds.f90 -o functions/convert_winds.exe
    fi
    functions/convert_winds.exe
    mv sfc_wind_ok ${output_file}.sfc
    rm -f sfc_wind temp_file.grib2
    # ---
    YY=`echo $fdate | cut -c1-4`
    MM=`echo $fdate | cut -c5-6`
    DD=`echo $fdate | cut -c7-8`
    HH=`echo $ftime`
    #sed 's/YYYY-MM-DD:HH/'$YY-$MM-${DD}\_$HH'/g' ${output_file}.sfc > tmp_file
    mv -f ${output_file}.sfc ${YY}${MM}${DD}_${HH}.sfc
    #rm -f ${output_file}.sfc
    # --- do plot
    if [ $if_plot -ge 1 ] ; then
      if test ! -e functions/plot_winds.exe ; then
         pgf90 functions/plot_winds.f90 -o functions/plot_winds.exe
      fi
      if [ $plot_nice -ge 1 ] ; then
        pgf90 functions/plot_winds_nice.f90 -o functions/plot_winds_nice.exe
        functions/plot_winds_nice.exe ${YY}${MM}${DD}_${HH}.sfc
        sed -e 's/wgs_date/'${YY}'-'${MM}'-'${DD}'_'${HH}'/'  \
        2uv.gs > 2uv_OK.gs
        grads -bpc 2uv_OK.gs 
        mv 2new_year_1.jpg ${YY}${MM}${DD}_${HH}.jpg
      fi
      if [ $plot_nice -eq 0 ] ; then
        functions/plot_winds.exe ${YY}${MM}${DD}_${HH}.sfc
      fi
    fi
    # --- output 
    
    #if [ $if_plot -ge 1 ] ; then
    #  cd ${YY}_sfc
    #  if test ! -d jpg_${MM} ; then
    #     mkdir jpg_${MM}
    #  fi
    #  cd jpg_${MM}
    #  mv ../../*.gif ./
    #  cd ../..
    #fi
  fi

  # =================================
  # --- do delete
  rm -f msg temp*.grib2 dimension.dat *.grib2
  if test ! -s ${YY}.msg ; then
    rm -f ${YY}.msg
  fi
 # ==================== End of File ========================

