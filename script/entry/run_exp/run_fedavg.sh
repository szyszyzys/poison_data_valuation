agg_name="fedavg"

bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name FMNIST --aggregation_method "$agg_name"

bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR --aggregation_method "$agg_name"

bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name FMNIST --aggregation_method "$agg_name" --poison_strength "0.5" --gradient_manipulation_mode cmd --trigger_rate 0.5

bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name FMNIST --aggregation_method "$agg_name" --poison_strength "1" --gradient_manipulation_mode cmd --trigger_rate 0.5

bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name FMNIST --aggregation_method "$agg_name" --gradient_manipulation_mode single --trigger_rate 0.5

bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name FMNIST --aggregation_method "$agg_name" --gradient_manipulation_mode single --trigger_rate 0.3

bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name FMNIST --aggregation_method "$agg_name" --gradient_manipulation_mode single --trigger_rate 0.1






agg_name="martfl"

bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name FMNIST --aggregation_method "$agg_name"

bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR --aggregation_method "$agg_name"

bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name FMNIST --aggregation_method "$agg_name" --poison_strength "0.5" --gradient_manipulation_mode cmd --trigger_rate 0.5

bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name FMNIST --aggregation_method "$agg_name" --poison_strength "1" --gradient_manipulation_mode cmd --trigger_rate 0.5

bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name FMNIST --aggregation_method "$agg_name" --gradient_manipulation_mode single --trigger_rate 0.5

bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name FMNIST --aggregation_method "$agg_name" --gradient_manipulation_mode single --trigger_rate 0.3

bash script/gradient_script/backdoor.sh --n_adversaries "1,2,3,4,5" --gpu_ids 6 --dataset_name FMNIST --aggregation_method "$agg_name" --gradient_manipulation_mode single --trigger_rate 0.1
