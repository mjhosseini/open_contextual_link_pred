#!/bin/bash
fileid=1VdeY62BjWS4bERKU-v-nnEu96IzjedQL
filename="pretrained_models.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|perl -nle'print $& while m{(confirm=[a-zA-Z0-9\-_]+)}g'`&id=${fileid}" -o ${filename}
tar -xvzf pretrained_models.tar.gz
rm pretrained_models.tar.gz
rm cookie
