#!/bin/bash
fileid=1d93CTsJ9OG0I68fHHWvX8iVkSefCmkyq
filename="data.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|perl -nle'print $& while m{(confirm=[a-zA-Z0-9\-_]+)}g'`&id=${fileid}" -o ${filename}
tar -xvzf data.tar.gz
rm data.tar.gz
rm cookie