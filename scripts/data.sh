#!/bin/bash
fileid=1lGqeBUL2JKlAclvsqNfwwR9NB6wl5uEH
filename="data.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
tar -xvzf data.tar.gz
rm cookie
