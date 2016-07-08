#!/bin/bash
awk -F , '{

  for (shiftx=-1;shiftx<=1;shiftx++) {
  for (shifty=-1;shifty<=1;shifty++) {

    for (y=0;y<28;y++) {
      for (x=1;x<=28;x++) {
        x_shift = x + shiftx;
        y_shift = y + shifty;
        if ((x_shift<1) || (x_shift>28) || (y_shift<0) || (y_shift>=28)) {
          printf "0,";
        } else {
          i=y_shift*28+x_shift;
          printf $i",";
        }
      }
    }

		printf"\n"

	}}
}' |sed 's///g' | sed 's/,$//g'
