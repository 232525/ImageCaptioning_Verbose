# Data Preparation
For this repo, we need to: 
1. generate necessary files (e.g. `mscoco` folder) for model training and evaluation. 
2. extract the image features for corresponding datasets.

## 1. Datasets Download
Download raw datasets from corresponding official website:
- MSCOCO: https://cocodataset.org/#download
- Flickr30K: https://www.kaggle.com/hsankesara/flickr-image-dataset
- NoCaps: https://nocaps.org/download

Download 'Karpathy split' json files for MSCOCO and Flickr30K from [kaggle](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits).
```
|-dataset_coco.json
|-dataset_flickr30k.json
```

## 2. MSCOCO
``` bash
python data_preparation.py \
    --anno_file ./karpathy_split/dataset_coco.json \
    --save_path ./preparation_output/mscoco \
    --dataset_name coco
```
Run the above command and generate the following files (or you can also download the necessary files from [here](https://drive.google.com/open?id=1i5YJRSZtpov0nOtRyfM0OS1n0tPCGiCS), it was released by [JDAI-CV/image-captioning](https://github.com/JDAI-CV/image-captioning)):

If you generate these files by yourself, maybe you need change the corresponding `config.yml` file to use them correctly.
```
preparation_output/mscoco/
├── feature
├── misc
│   ├── coco_captions_test5k.json
│   ├── coco_captions_val5k.json
│   ├── coco_train_cider.pkl
│   └── coco_train_gts.pkl
├── sent
│   ├── coco_train_input.pkl
│   └── coco_train_target.pkl
└── txt
    ├── coco_bad_token.txt
    ├── coco_test_image_id.txt
    ├── coco_train_image_id.txt
    ├── coco_val_image_id.txt
    └── coco_vocabulary.txt
```
### 2.1 Feature Extraction
#### 2.1.1 UpDown Region Features
Download from [peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention), and convert these features to `.npz` files:
```bash
python create_feats.py \
    --infeats {DOWNLOAD_FILES} \
    --outfolder preparation_output/mscoco/feature/COCO_UpDown_10_100_Feats
```

### 2.1.2 SwinTransformer Grid Features
......

## 3. Flickr30K
``` bash
python data_preparation.py \
    --anno_file ./karpathy_split/dataset_flickr30k.json \
    --save_path ./preparation_output/flickr30k \
    --dataset_name flickr30k
```
Run the above command and generate the following files:
```
preparation_output/flickr30k/
├── feature
├── misc
│   ├── flickr30k_captions_test1k.json
│   ├── flickr30k_captions_val1k.json
│   ├── flickr30k_train_cider.pkl
│   └── flickr30k_train_gts.pkl
├── sent
│   ├── flickr30k_train_input.pkl
│   └── flickr30k_train_target.pkl
└── txt
    ├── coco_bad_token.txt
    ├── flickr30k_test_image_id.txt
    ├── flickr30k_train_image_id.txt
    ├── flickr30k_val_image_id.txt
    └── flickr30k_vocabulary.txt
```
### 3.1 Feature Extraction
similar to MSCOCO

UpDown Region Features for Flickr30K datasets can be download from [kuanghuei/SCAN](https://github.com/kuanghuei/SCAN).

## 4. NoCaps: a evaluation benchmark

UpDown Region Features for NoCaps datasets can be download from [here](https://nocaps.org/updown-baseline/setup_dependencies.html).