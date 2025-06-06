CUDA_VISIBLE_DEVICES=0 python train.py \
   --expname train_hamlyn\
   --use_viewdirs True\
   --dataset_name hamlyn \
   --datadir <your_hamlyn_data_folder> \
   --view_num 7 \
   --num_epochs 30 \
   --patch_size 4 \
   --patch_num 50 \
   --lrate 2e-4 \
   --ckpt ./pretrained_weights/ucnerf.tar 
