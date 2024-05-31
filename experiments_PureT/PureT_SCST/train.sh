# CUDA_VISIBLE_DEVICES=1 python main.py --folder ./experiments_PureT/PureT_SCST/ --resume 27

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=3142 --nproc_per_node=4 \
    main_multi_gpu.py \
    --folder ./experiments_PureT/PureT_SCST/ \
    --resume 27
    
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=3142 --nproc_per_node=4 \
#     main_multi_gpu.py \
#     --folder ./experiments_PureT/PureT_SCST/ \
#     --resume 27