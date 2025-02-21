bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name FMINIST

bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR


bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5" --gradient_manipulation_mode "single" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name FMINIST

bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5" --gradient_manipulation_mode "single" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR

bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5,1.0" --gradient_manipulation_mode "cmd" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name FMINIST

bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5,1.0" --gradient_manipulation_mode "cmd" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR


rsync -av --exclude='*.pt' zzs5287@E5-cse-cbsjm01.eecscl.psu.edu:/scratch/zzs5287/poison_data_valuation/results/backdoor/FMINIST/backdoor_mode_single_strength_0.1/  /c/Users/zeyu song/Desktop/codes/poison_data_valuation/result