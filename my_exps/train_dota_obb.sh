#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
env=yolox_dect
conda activate ${env}
echo -e "\033[34m*******************************\033[0m"
echo -e "\033[31mactivate env ${env}\033[0m"
echo -e "\033[34m*******************************\033[0m"
echo -e "\033[34mCurrent dir is ${PWD}\033[0m"

expn=$1
config=$2
cuda=$3

work_dir=./YOLOX_outputs/${expn}

echo -e "\033[33mconfig is ${config}\033[0m"
echo -e "\033[33mwork_dir is ${work_dir}\033[0m"
echo -e "\033[33mdevice is cuda: ${cuda}\033[0m"

sleep 2s
if [ -d ${work_dir} ]; then
    read -n1 -p "find ${work_dir}, do you want to del(y or n):"
    echo 
    if [ ${REPLY}x = yx ]; then  
	rm -rf ${work_dir}
	echo -e "\033[31mAlready del ${work_dir}\033[0m"
    else
	ls -a | grep *log*
	read -n1 -p "do you want to del log(y or n):"
	echo
	if [ ${REPLY}x = yx ]; then
	   rm -rf *log*
	   echo -e "\033]31mAlready del log files\033[0m"
	fi
    
    fi
fi
sleep 2s

CUDA_VISIBLE_DEVICES=${cuda} python tools/train.py -expn ${expn} -f ${config} -d 1 -b 8 --cache --fp16
cd -

