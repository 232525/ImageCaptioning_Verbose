# Single model inference
# UpDown, updown region feats
CUDA_VISIBLE_DEVICES=0 python main_test.py \
    --folder experiments/experiments_UpDown/UpDown_SCST/ \
    --resume **

# XLAN, updown region feats
CUDA_VISIBLE_DEVICES=0 python main_test.py \
    --folder experiments/experiments_XLAN/xlan_rl/ \
    --resume 36

# XTransformer, updown region feats
CUDA_VISIBLE_DEVICES=0 python main_test.py \
    --folder experiments/experiments_XLAN/xtransformer_rl/ \
    --resume 62

# Raw Transformer, updown region feats
CUDA_VISIBLE_DEVICES=0 python main_test.py \
    --folder experiments/experiments_RawTransformer/RawTransformer_SCST/ \
    --resume 18

# PureT, swinL grid feats
CUDA_VISIBLE_DEVICES=0 python main_test.py \
    --folder experiments/experiments_PureT/PureT_SCST/ \
    --resume 27

##################################

# Ensemble model inference
CUDA_VISIBLE_DEVICES=0 python main_ensemble_test.py \
    --folder experiments/experiments_UpDown/UpDown_SCST/ \
    --model_folders experiments/experiments_UpDown/UpDown_SCST/ experiments/experiments_UpDown/UpDown_SCST/ \
    --model_resumes ** ** \
    --model_types UpDown UpDown

CUDA_VISIBLE_DEVICES=0 python main_ensemble_test.py \
    --folder experiments/experiments_XLAN/xlan_rl/ \
    --model_folders experiments/experiments_XLAN/xlan_rl/ experiments/experiments_XLAN/xlan_rl/ \
    --model_resumes 36 36 \
    --model_types XLAN XLAN

CUDA_VISIBLE_DEVICES=0 python main_ensemble_test.py \
    --folder experiments/experiments_XLAN/xtransformer_rl/ \
    --model_folders experiments/experiments_XLAN/xtransformer_rl/ experiments/experiments_XLAN/xtransformer_rl/ \
    --model_resumes 62 62 \
    --model_types XTransformer XTransformer

CUDA_VISIBLE_DEVICES=0 python main_ensemble_test.py \
    --folder experiments/experiments_RawTransformer/RawTransformer_SCST/ \
    --model_folders experiments/experiments_RawTransformer/RawTransformer_SCST/ experiments/experiments_RawTransformer/RawTransformer_SCST/ \
    --model_resumes 18 18 \
    --model_types RawTransformer RawTransformer

CUDA_VISIBLE_DEVICES=0 python main_ensemble_test.py \
    --folder experiments/experiments_PureT/PureT_SCST/ \
    --model_folders experiments/experiments_PureT/PureT_SCST/ experiments/experiments_PureT/PureT_SCST/ \
    --model_resumes 27 27 \
    --model_types PureT PureT