#!/bin/bash
fileid=1lGqeBUL2JKlAclvsqNfwwR9NB6wl5uEH
filename="data.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
tar -xvzf data.tar.gz
rm cookie
