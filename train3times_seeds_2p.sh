#!/bin/bash


while getopts 'm:e:c:t:l:w:' OPT; do
    case $OPT in
        m) method=$OPTARG;;
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    t) task=$OPTARG;;
		    l) lr=$OPTARG;;
		    w) cps_w=$OPTARG;;
    esac
done
echo $method
echo $cuda

epoch=200
echo $epoch

labeled_data="labeled_2p"
unlabeled_data="unlabeled_2p"
folder="Task_"${task}"_2p/"
cps="AB"

echo $folder

python3 code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r
python3 code/test.py --task ${task} --exp ${folder}${method}${exp}/fold1 -g ${cuda} --cps ${cps}
python3 code/evaluate_Ntimes.py --task ${task} --exp ${folder}${method}${exp} --folds 1 --cps ${cps}
#python3 code/train_${method}.py --exp ${folder}${method}${exp}/fold2 --seed 1 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r
#python3 code/test.py --exp ${folder}${method}${exp}/fold2 -g ${cuda} --cps ${cps}
#python3 code/evaluate_Ntimes.py --exp ${folder}${method}${exp} --folds 2 --cps ${cps}
#python3 code/train_${method}.py --exp ${folder}${method}${exp}/fold3 --seed 666 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r
#python3 code/test.py --exp ${folder}${method}${exp}/fold3 -g ${cuda} --cps ${cps}

#python3 code/evaluate_Ntimes.py --exp ${folder}${method}${exp} --cps ${cps}
