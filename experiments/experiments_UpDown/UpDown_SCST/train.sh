# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=3140 --nproc_per_node=4 main_multi_gpu.py --folder ./experiments/experiments_UpDown/UpDown_SCST/ --resume *

CUDA_VISIBLE_DEVICES=1 python main.py --folder ./experiments/experiments_UpDown/UpDown_SCST/ --resume 14