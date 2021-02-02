
# CIFAR10 - 5 Tasks 
# For  training --
CUDA_VISIBLE_DEVICES=0 python cifar10_incremental_five_tasks.py --split 0 --train 1 --retrain 1 --call_test 0  --rand_init 0 --train_epoch 40  --epoch_list 60 --save-model 1 --full_task 0 --lr 0.01 --momentum 0.9 --var_kept 0.995 0.95 0.95 0.95 0.95 --num_out 2 --file_suffix 10

# For Testing --
CUDA_VISIBLE_DEVICES=0 python cifar10_incremental_five_tasks.py --split 0 --train 0 --retrain 0 --call_test 1  --rand_init 0 --train_epoch 40  --epoch_list 60 --save-model 0 --full_task 0 --lr 0.01 --momentum 0.9 --var_kept 0.995 0.95 0.95 0.95 0.95 --num_out 2 --file_suffix 10



# CIFAR100 - 10 Tasks 
# For  training --
CUDA_VISIBLE_DEVICES=0 python cifar100_incremental_ten_tasks.py --split 0 --train 1 --retrain 1 --call_test 0  --rand_init 0 --train_epoch 80  --epoch_list 120 --save-model 1 --full_task 0 --lr 0.01 --momentum 0.9 --var_kept 0.999 0.99 0.99 0.99 0.99 --num_out 10 --file_suffix 10

# For Testing --
CUDA_VISIBLE_DEVICES=0 python cifar100_incremental_ten_tasks.py --split 0 --train 0 --retrain 0 --call_test 1  --rand_init 0 --train_epoch 80  --epoch_list 120 --save-model 0 --full_task 0 --lr 0.01 --momentum 0.9 --var_kept 0.999 0.99 0.99 0.99 0.99 --num_out 10 --file_suffix 10



# Mixture of Dataset - 8 Tasks 
# For Training 
CUDA_VISIBLE_DEVICES=0 python resnet_8_mixture.py --var_kept 0.995 0.95 0.95 0.95 0.95 0.93 0.93 0.93 0.93 0.93 0.92 0.92 0.92 0.92 0.92 0.91 0.91 0.91 0.91 0.91 --lr 0.01 --file_suffix 10 --train --retrain --save-model

# For Testing 
CUDA_VISIBLE_DEVICES=0 python resnet_8_mixture.py --var_kept 0.995 0.95 0.95 0.95 0.95 0.93 0.93 0.93 0.93 0.93 0.92 0.92 0.92 0.92 0.92 0.91 0.91 0.91 0.91 0.91 --lr 0.01 --file_suffix 10 --call-test
