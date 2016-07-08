#!/bin/bash
echo "ImageId,Label" > $1
cat ./predict_res | awk '{print NR","$0}' >> $1
gzip $1
