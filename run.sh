CUDA_VISIBLE_DEVICES=1 python main.py \
--cfg Configs/nnUnetConfigs/UNet.json \
--data_list_file Configs/nnUNetDatalist/imagesTest.json \
--model_path "PATH_TO_CHECKPOINT"