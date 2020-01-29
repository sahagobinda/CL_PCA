#CUDA_VISIBLE_DEVICES=3 python cifar10_multi_task_twodiv_CT.py --split 0 --train 1 --retrain 1  --train_epoch 40 --epoch_list 25 30 30 35 40  --save-model 1 --full_task 0 --lr 0.01 --momentum 0.9 --var_kept 0.985 --num_out 2 --file_suffix 99 | tee ./PARAM/CIFAR10_task/cifar_10_test_upload.log

#CUDA_VISIBLE_DEVICES=3 python cifar10_multi_task_twodiv_CT.py --split 0 --train 0 --retrain 0  --call_test 1 --train_epoch 40 --epoch_list 25 30 30 35 40  --save-model 0 --full_task 0 --lr 0.01 --momentum 0.9 --var_kept 0.985 --num_out 2 --file_suffix 99

#CUDA_VISIBLE_DEVICES=3 python cifar10_incremental_five_tasks.py --split 0 --train 1 --retrain 1  --train_epoch 40 --epoch_list 25 30 30 35 40  --save-model 1 --full_task 0 --lr 0.01 --momentum 0.9 --var_kept 0.985 --num_out 2 --file_suffix 50| tee ./PARAM/CIFAR10_task/cifar_10_test_upload_final.log

CUDA_VISIBLE_DEVICES=3 python cifar10_incremental_five_tasks.py --split 0 --train 0 --retrain 0  --call_test 1 --train_epoch 40 --epoch_list 25 30 30 35 40  --save-model 0 --full_task 0 --lr 0.01 --momentum 0.9 --var_kept 0.985 --num_out 2 --file_suffix 50
