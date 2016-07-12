#!/bin/csh
# === cdas1.splgrbanl.tar
# --- untar ---
  foreach file ( `ls cdas1*.splgrbanl.tar` )
   echo $file
   #echo $file:r  
   tar -xvf $file
   set yymm = `wgrib2 cdas1.t00z.splgrbanl.grib2 | sed -n 1p | sed 's/:/ /g' | sed 's/=/ /g' |awk '{print $4}' `
   mv cdas1.t00z.splgrbanl.grib2 splgrbanl.${yymm}.grb2
   set yymm = `wgrib2 cdas1.t06z.splgrbanl.grib2 | sed -n 1p | sed 's/:/ /g' | sed 's/=/ /g' |awk '{print $4}' `
   mv cdas1.t06z.splgrbanl.grib2 splgrbanl.${yymm}.grb2
   set yymm = `wgrib2 cdas1.t12z.splgrbanl.grib2 | sed -n 1p | sed 's/:/ /g' | sed 's/=/ /g' |awk '{print $4}' `
   mv cdas1.t12z.splgrbanl.grib2 splgrbanl.${yymm}.grb2
   set yymm = `wgrib2 cdas1.t18z.splgrbanl.grib2 | sed -n 1p | sed 's/:/ /g' | sed 's/=/ /g' |awk '{print $4}' `
   mv cdas1.t18z.splgrbanl.grib2 splgrbanl.${yymm}.grb2
  end
exit  
  
# === splanl.gdas
# --- untar ---
  foreach file ( `ls splanl*tar` )
   echo $file
   #echo $file:r  
   tar -xvf $file
  end
# --- rename ---
  foreach file ( `ls splanl*grb2` )
   echo $file
   set yymm = ` echo $file | sed 's/\./ /g' | awk '{print $3}' `
   echo $yymm
   mv $file splgrbanl.${yymm}.grb2
   #echo $file:r  
  end

