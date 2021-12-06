# python train.py --name v2_Resnet_18 --model_name Resnet_18\
#     --output ./output_Resnet_18 \
#     --log_path ./log/v2_Resnet_18.txt    --gpu_ids "1" \
#     --batch_size 10   --epoch 200   --learning_rate 0.0001 \
#     --lr_scheduler const   --lr_warmup 1e-4   --warmup_epochs 5 \
#     --decay_epochs 10   --lr_decay_rate 0.6 

# python train.py --name v1_Resnet_50 --model_name Resnet_50\
#     --output ./output_Resnet_50 \
#     --log_path ./log/v2_Resnet_50.txt    --gpu_ids "1" \
#     --batch_size 10   --epoch 200   --learning_rate 0.0001 \
#     --lr_scheduler const   --lr_warmup 1e-4   --warmup_epochs 5 \
#     --decay_epochs 10   --lr_decay_rate 0.6 

# python train.py --name v1_Resnet_34 --model_name Resnet_34\
#     --output ./output_Resnet_34 \
#     --log_path ./log/v2_Resnet_34.txt    --gpu_ids "1" \
#     --batch_size 10   --epoch 200   --learning_rate 0.0001 \
#     --lr_scheduler const   --lr_warmup 1e-4   --warmup_epochs 5 \
#     --decay_epochs 10   --lr_decay_rate 0.6 

python train.py --name v1_GoogleNet --model_name GoogleNet\
    --output ./output_v1_GoogleNet \
    --log_path ./log/v1_GoogleNet.txt    --gpu_ids "2" \
    --batch_size 10  --epoch 200   --learning_rate 0.0001 \
    --lr_scheduler const   --lr_warmup 1e-4   --warmup_epochs 5 \
    --decay_epochs 10   --lr_decay_rate 0.6 