datapath=./PolanecAD
datasets=('tablet')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

./simplenet_env/Scripts/python.exe main.py \
--gpu -1 \
--seed 0 \
--log_group simplenet_polanecad \
--log_project PolanecAD_Results \
--results_path results \
--run_name run_40ep \
--save_segmentation_images \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1024 \
--target_embed_dimension 1024 \
--patchsize 3 \
--meta_epochs 40 \
--embedding_size 256 \
--gan_epochs 4 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 8 \
--resize 200 \
--imagesize 200 "${dataset_flags[@]}" polanecad $datapath
