#!bin/bash
BASE_DIR=./data/2
SRC_PATH=target.jpg
DST_PATH=background.jpg
MASK_PATH=mask.jpg
TYPE=plane  # just for the name of the output file, define by yourself
left=0
up=0  # choose where to clone

python test.py -base-dir ${BASE_DIR} -src-path ${SRC_PATH} \
    -mask-path ${MASK_PATH} -dst-path ${DST_PATH} \
    -pos $left $up -type ${TYPE}