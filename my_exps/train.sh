#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
env=yolox_dect
conda activate ${env}
echo -e "\033[34m*******************************\033[0m"
echo -e "\033[31mactivate env ${env}\033[0m"
echo -e "\033[34m*******************************\033[0m"
echo -e "\033[34mCurrent dir is ${PWD}\033[0m"

expn=$1
exp=$2
cuda_device=${3:-0}
num_device=${4:-1}
batch_size=${5:-8}
py_args=${@:6}

work_dir=YOLOX_outputs/${expn}

echo -e "\033[33mexp is ${exp}\033[0m"
echo -e "\033[33mcuda_device is cuda: ${cuda_device}\033[0m"
echo -e "\033[33mnum_device is ${num_device}\033[0m"
echo -e "\033[33mbatch_size is ${batch_size}\033[0m"
echo -e "\033[33mother args: ${py_args}\033[0m"

sleep 2s
deal_array=("events.out.tfevents.*")
if [ -d ${work_dir} ]; then
    read -n1 -p "find ${work_dir}, do you want to del(y or n):"
    echo 
    if [ ${REPLY}x = yx ]; then  
		rm -rf ${work_dir}
		echo -e "\033[31mAlready del ${work_dir}\033[0m"
    else
		cd $work_dir
		for ((i=0;i<${#deal_array[@]};i++)) 
		do
			ls -a | grep ${deal_array[$i]}
			read -n1 -p "do you want to del ${deal_array[$i]} (y or n):"
			echo
			if [ ${REPLY}x = yx ]; then
				rm -rf ${deal_array[$i]}
				echo -e "\033]31mAlready del log ${deal_array[$i]}\033[0m"
			fi
		done
		cd -
    fi
fi
sleep 2s

CUDA_VISIBLE_DEVICES=${cuda_device} python tools/train.py -expn ${expn} -f ${exp} -d ${num_device} -b ${batch_size} ${py_args}

cd -

