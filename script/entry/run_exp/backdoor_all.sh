bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name FMINIST

bash script/gradient_script/backdoor.sh --poison_strength "0" --gradient_manipulation_mode "None" --n_adversaries "0" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR


bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5" --gradient_manipulation_mode "single" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name FMINIST

bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5" --gradient_manipulation_mode "single" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR

bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5,1.0" --gradient_manipulation_mode "cmd" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name FMINIST

bash script/gradient_script/backdoor.sh --poison_strength "0.1,0.5,1.0" --gradient_manipulation_mode "cmd" --n_adversaries "1,2,3,4,5" --local_epoch 2 --local_lr 1e-2 --dataset_name CIFAR

results/exp_36/backdoor_trigger_static/is_sybil_mimic/is_iid_discovery/buyer_data_biased/martfl_True/FMNIST/discovery_quality_10.0/backdoor_mode_single_trigger_rate_0.5_trigger_type_blended_patch/n_seller_30_adv_rate_0.4_local_epoch_2_local_lr_0.01/run_9/market_log.json