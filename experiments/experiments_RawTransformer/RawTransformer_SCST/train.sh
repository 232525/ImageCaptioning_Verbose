# CUDA_VISIBLE_DEVICES=3,2,1,0 python3 -m torch.distributed.launch --nproc_per_node=4 main.py --folder ./experiments/xtransformer

CUDA_VISIBLE_DEVICES=1 python main.py --folder ./experiments_ablation_swin/raw_transformer_rl --resume 13