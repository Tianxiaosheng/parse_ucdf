#! /bin/bash
# Program: plot part of data from specific ucdf.
# History:
#         2022/09/02    TianXiaosheng   First release

set -e #exit on the frist error

while getopts :f: option
do
case "${option}"
    in
    f) ucdf_file=${OPTARG};;
    ?) echo "wrong input opt!";;
esac
done
echo "ucdf_file" ${ucdf_file}


. set_env.sh

bin/uos_replay-dump 1.txt ${ucdf_file}

python par_gui.py --log 1.txt
