# ImageCaptioning-Verbose
PyTorch implementation for Image Captioning, supporting single or multi-gpus training, and inference with a single model or ensemble of multiple models.

__This repo is a version that uses pre-extracted features for training and testing.__
- [x] Raw Transformer
- [x] PureT: [[Paper]](https://arxiv.org/abs/2203.15350), [[Source Code]](https://github.com/232525/PureT)
- [x] XLAN and XTransformer: [[Paper]](https://arxiv.org/abs/2003.14080), [[Source Code]](https://github.com/JDAI-CV/image-captioning)
- [ ] UpDown
- [ ] M2 Transformer

## Requirements (Our Main Enviroment)
+ Python 3.10.11
+ PyTorch 1.13.1
+ TorchVision 0.14.1 
+ [coco-caption](https://github.com/tylin/coco-caption)
+ numpy
+ tqdm

Note: Also supports earlier PyTorch versions, such as 1.5.1. For newer version (>2.0), we have not verified!

## Preparation
### 1. coco-caption preparation
Refer coco-caption [README.md](./coco_caption/README.md), you will first need to download the [Stanford CoreNLP 3.6.0](http://stanfordnlp.github.io/CoreNLP/index.html) code and models for use by SPICE. To do this, run:
```bash
cd coco_caption
bash get_stanford_models.sh
```
### 2. Data preparation
The necessary files in training and evaluation are saved in __`mscoco`__ folder, which is organized as follows:
```
mscoco/
|--feature/
    |--COCO_SwinL_Feats/ # Grid Features, for PureT
       |--*.npz
    |--COCO_UpDown_10_100_Feats/ # Region Features, for UpDown, XLAN, XTransformer, ...
       |--*.npz
|--misc/
|--sent/
|--txt/
```
where the `mscoco/feature/COCO_SwinL_Feats` folder contains the pre-extracted features of [MSCOCO 2014](https://cocodataset.org/#download) dataset. You can download other files from [GoogleDrive](https://drive.google.com/drive/folders/1HBw5NGGw8DjkyNurksCP5v8a5f0FG7zU?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1tyXGJx50sllS-zylN62ZAw)(提取码: hryh). 

## Training
*Note: our repository is mainly based on [JDAI-CV/image-captioning](https://github.com/JDAI-CV/image-captioning), and we directly reused their config.yml files, so there are many useless parameter in our model. （__waiting for further sorting__）*

### 1. Training under XE loss
Download pre-trained Backbone model (Swin-Transformer) from [GoogleDrive](https://drive.google.com/drive/folders/1HBw5NGGw8DjkyNurksCP5v8a5f0FG7zU?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1tyXGJx50sllS-zylN62ZAw)(提取码: hryh) and save it in the root directory.

Before training, you may need check and modify the parameters in `config.yml` and `train.sh` files. Then run the script:

```
# for XE training
bash experiments_PureT/PureT_XE/train.sh
```
### 2. Training using SCST (self-critical sequence training)
Copy the pre-trained model under XE loss into folder of `experiments_PureT/PureT_SCST/snapshot/` and modify `config.yml` and `train.sh` files. Then run the script:

```bash
# for SCST training
bash experiments_PureT/PureT_SCST/train.sh
```

## Evaluation
You can download the pre-trained model from [GoogleDrive](https://drive.google.com/drive/folders/1HBw5NGGw8DjkyNurksCP5v8a5f0FG7zU?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1tyXGJx50sllS-zylN62ZAw)(提取码: hryh). 

```bash
CUDA_VISIBLE_DEVICES=0 python main_test.py --folder experiments_PureT/PureT_SCST/ --resume 27
```

|BLEU-1|BLEU-2|BLEU-3|BLEU-4|METEOR|ROUGE-L| CIDEr |SPICE |
| ---: | ---: | ---: | ---: | ---: | ---:  | ---:  | ---: |
| 82.1 | 67.3 | 52.0 | 40.9 | 30.2 | 60.1  | 138.2 | 24.2 |


## Reference
If you find this repo useful, please consider citing (no obligation at all):
```
@inproceedings{wangyiyu2022PureT,
  author       = {Yiyu Wang and
                  Jungang Xu and
                  Yingfei Sun},
  title        = {End-to-End Transformer Based Model for Image Captioning},
  booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages        = {2585--2594},
  publisher    = {{AAAI} Press},
  year         = {2022},
  url          = {https://ojs.aaai.org/index.php/AAAI/article/view/20160}, 
  doi          = {10.1609/aaai.v36i3.20160},
}

```

## Acknowledgements
This repository is based on [JDAI-CV/image-captioning](https://github.com/JDAI-CV/image-captioning), [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) and [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

TODO:
- [ ] Details of data preparation
- [ ] More datasets supported
- [ ] More approachs supported