bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name FMINIST

bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR


bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5" --gradient_manipulation_mode "single" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name FMINIST

bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5" --gradient_manipulation_mode "single" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR

bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5,1.0" --gradient_manipulation_mode "cmd" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name FMINIST

bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5,1.0" --gradient_manipulation_mode "cmd" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR


