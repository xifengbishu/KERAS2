#!/bin/sh
  if [ $# -eq 0  ] ; then
     echo ' '
     echo ' Usage:'
     echo '   get_cfs_uv.sh wgrib2_file '
     echo ' '
     exit
  fi
  # --- get variables
  #git pull origin master
  rm_file=$1
  git rm --cached $1
  git commit --amend -CHEAD
  git push
