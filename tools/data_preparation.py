# encoding=utf-8
import json
import pickle
import os
import sys
import argparse
from tqdm import tqdm
from nltk import word_tokenize
import numpy as np
import pickle
from collections import defaultdict

def tokenize(sent):
    sent = sent.lower().strip()
    words = word_tokenize(sent)
    return words

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Data Preparation for COCO-style Datasets')
    parser.add_argument('--anno_file', help='annotation file that contains train / val / test splits', type=str, default=None)

    parser.add_argument('--save_path', help='save path', type=str, default=None)
    parser.add_argument('--dataset_name', help='dataset name', type=str, default='coco')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

class DataPreparation:
    def __init__(self, args):
        self.args = args
        print(self.args)
        
        # 创建文件夹存储生成的文件
        for _name in ["feature", "misc", "sent", "txt"]:
            path_name = os.path.join(self.args.save_path, _name)
            if not os.path.exists(path_name):
                os.makedirs(path_name)

    def run(self,):
        print(f"read and split {self.args.dataset_name} dataset")
        self.read_and_split_data()
        self.name_output_files()
        print("Down!\n")
        print(f"generate txt files:")
        print(f"\t{self.txt_vocabulary_file}")
        print(f"\t{self.txt_train_id_file}")
        print(f"\t{self.txt_val_id_file}")
        print(f"\t{self.txt_test_id_file}")
        self.save_txt_files()
        print("Down!\n")
        print(f"generate json files for evaluation:")
        print(f"\t{self.misc_test5k_ann_file}")
        print(f"\t{self.misc_val5k_ann_file}")
        self.save_eval_files()
        print("Down!\n")
        print(f"generate some pkl files (gt, input, target) for training:")
        print(f"\t{self.misc_gts_file}")
        print(f"\t{self.sent_input_file}")
        print(f"\t{self.sent_target_file}")
        self.save_pkl_files()
        print("Down!\n")
        print(f"generate CIDEr cache files:")
        print(f"\t{self.misc_cider_file}")
        self.save_cider_cache()
        print("All Down!\n")

    def read_and_split_data(self,):
        """
            划分数据集、统计词频
        """
        # 读取karpathy split anno_file
        with open(self.args.anno_file, "r") as f:
            all_data = json.load(f)['images']

        self.train_split = []
        self.val_split = []
        self.test_split = []
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []

        # 统计词频，
        # 原则上统计train set就可以，但是https://github.com/JDAI-CV/image-captioning中统计的是所有，这里为了保持一致也统计了train / val / test所有词汇词频
        # 但是统计的词汇表在顺序上不一致，不清楚他们是怎么处理的
        token_counter = {} 
        # 按照karpathy split进行划分
        for _data in tqdm(all_data):
            _file_name = _data['filename'] # ***.jpg
            if _data.get('cocoid', None) is not None:
                _id = _data['cocoid']
            else:
                _id = int(_file_name.split('.')[0])
            # 统计词频
            for _sent in _data['sentences']:
                for _token in _sent['tokens']:
                    token_counter[_token] = token_counter.get(_token, 0) + 1
            if _data['split'] in ['train', 'restval']:
                self.train_split.append(_data)
                self.train_ids.append(_id)
                # # 统计train set词频
                # for _sent in _data['sentences']:
                #     for _token in _sent['tokens']:
                #         token_counter[_token] = token_counter.get(_token, 0) + 1
            elif _data['split'] == 'val':
                self.val_split.append(_data)
                self.val_ids.append(_id)
            elif _data['split'] == 'test':
                self.test_split.append(_data)
                self.test_ids.append(_id)
            else:
                pass
        
        print("Train split: ", len(self.train_split))
        print("Val split: ", len(self.val_split))
        print("Test split: ", len(self.test_split))

        # 按照token出现次数排序
        self.ct = sorted([(count, token) for token, count in token_counter.items()], reverse=True)
        # self.ct = [(count, token) for token, count in token_counter.items()]

    def name_output_files(self,):
        # misc/
        cnt_val, cnt_test = len(self.val_split), len(self.test_split)
        if cnt_val > 1000:
            cnt_val_ = str(cnt_val // 1000) + "k"
        else:
            cnt_val_ = str(cnt_val)
        if cnt_test > 1000:
            cnt_test_ = str(cnt_test // 1000) + "k"
        else:
            cnt_test_ = str(cnt_test)
        self.misc_gts_file = f'{self.args.save_path}/misc/{self.args.dataset_name}_train_gts.pkl'
        self.misc_cider_file = f'{self.args.save_path}/misc/{self.args.dataset_name}_train_cider.pkl'
        self.misc_val5k_ann_file = f'{self.args.save_path}/misc/{self.args.dataset_name}_captions_val{cnt_val_}.json'  # 必须包含'type':'captions'字段
        self.misc_test5k_ann_file = f'{self.args.save_path}/misc/{self.args.dataset_name}_captions_test{cnt_test_}.json' # 必须包含'type':'captions'字段
        # sent/
        self.sent_input_file = f'{self.args.save_path}/sent/{self.args.dataset_name}_train_input.pkl'
        self.sent_target_file = f'{self.args.save_path}/sent/{self.args.dataset_name}_train_target.pkl'
        # txt/
        self.txt_train_id_file = f'{self.args.save_path}/txt/{self.args.dataset_name}_train_image_id.txt'
        self.txt_val_id_file = f'{self.args.save_path}/txt/{self.args.dataset_name}_val_image_id.txt'
        self.txt_test_id_file = f'{self.args.save_path}/txt/{self.args.dataset_name}_test_image_id.txt'
        self.txt_vocabulary_file = f'{self.args.save_path}/txt/{self.args.dataset_name}_vocabulary.txt'

        # others
        self.bad_tokens_file = f'{self.args.save_path}/txt/coco_bad_token.txt'

    def save_txt_files(self, ):
        index = 1
        self.token2index = {}  # 分词词汇 -> index序号 映射
        final_token = []  # 最终收集到的分词词汇集合
        bad_tokens = []
        with open(self.txt_vocabulary_file, 'w') as f:
            for count, token in self.ct:
                if count >= 6:
                    f.write(str(token) + '\n')
                    self.token2index[str(token)] = index
                    final_token.append((index, str(token)))
                    index += 1
                else:
                    bad_tokens.append(token)
            f.write(str('UNK') + '\n')
            self.token2index[str('UNK')] = index
            final_token.append((index, str('UNK')))
            
        with open(self.bad_tokens_file, 'w') as f:
            for token in bad_tokens:
                f.write(str(token) + '\n')
                
        with open(self.txt_train_id_file, 'w') as f:
            for _id in self.train_ids:
                f.write(str(_id) + '\n')

        with open(self.txt_val_id_file, 'w') as f:
            for _id in self.val_ids:
                f.write(str(_id) + '\n')

        with open(self.txt_test_id_file, 'w') as f:
            for _id in self.test_ids:
                f.write(str(_id) + '\n')

    def save_eval_files(self,):
        # 保存val / test annotation 到json文件中
        def info_pre():
            data = {}
            data['type'] = "captions"
            data['info'] = {
                "contributor": "232525", 
                "description": "MSCOCO",
                "version": "1",
                "year": 2024
            }
            data['licenses'] = [
                {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}, 
                {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2, "name": "Attribution-NonCommercial License"}, 
                {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3, "name": "Attribution-NonCommercial-NoDerivs License"}, 
                {"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4, "name": "Attribution License"}, 
                {"url": "http://creativecommons.org/licenses/by-sa/2.0/", "id": 5, "name": "Attribution-ShareAlike License"}, 
                {"url": "http://creativecommons.org/licenses/by-nd/2.0/", "id": 6, "name": "Attribution-NoDerivs License"}, 
                {"url": "http://flickr.com/commons/usage/", "id": 7, "name": "No known copyright restrictions"}, 
                {"url": "http://www.usa.gov/copyright.shtml", "id": 8, "name": "United States Government Work"}]
            data['images'] = []
            data['annotations'] = []
            return data

        # 共用字段生成
        val_annotation_data = info_pre()
        test_annotation_data = info_pre()
        caption_id = 1
        for _data in tqdm(self.test_split):
            _file_name = _data['filename'] # ***.jpg
            if _data.get('cocoid', None) is not None:
                _id = _data['cocoid']
            else:
                _id = int(_file_name.split('.')[0])
            _captions = [_['raw'] for _ in _data['sentences']]
            test_annotation_data['images'].append({'file_name': _file_name, 'id': _id})
            for _caption in _captions:
                test_annotation_data['annotations'].append(
                    {
                        'caption': _caption,
                        'id': caption_id,
                        'image_id': _id
                    }
                )
                caption_id += 1
                
        caption_id = 1
        for _data in tqdm(self.val_split):
            _file_name = _data['filename'] # ***.jpg
            if _data.get('cocoid', None) is not None:
                _id = _data['cocoid']
            else:
                _id = int(_file_name.split('.')[0])
            _captions = [_['raw'] for _ in _data['sentences']]
            val_annotation_data['images'].append({'file_name': _file_name, 'id': _id})
            for _caption in _captions:
                val_annotation_data['annotations'].append(
                    {
                        'caption': _caption,
                        'id': caption_id,
                        'image_id': _id
                    }
                )
                caption_id += 1
                
        with open(self.misc_test5k_ann_file, 'w') as f:
            json.dump(test_annotation_data, f)
            
        with open(self.misc_val5k_ann_file, 'w') as f:
            json.dump(val_annotation_data, f)

    def save_pkl_files(self,):
        # 需要将caption从str序列转为index序列
        def str2index(sent, token2index):
            result = [token2index[str(token)] if str(token) in token2index else token2index[str('UNK')] for token in sent.split(' ')] # 逐token转换为index
            result.append(0)  # 在末尾添加 <eos>
            return result

        def fill_list(_list, fill, length):
            _len = len(_list)
            if _len > length:
                return _list[:length]
            else:
                _fill = [fill for i in range(length-_len)]
                return _list + _fill

        train_gts = []     # list
        train_input = {}
        train_target = {}

        for _data in tqdm(self.train_split):
            _file_name = _data['filename']
            if _data.get('cocoid', None) is not None:
                _image_id = _data['cocoid']
            else:
                _image_id = int(_file_name.split('.')[0])
            _captions = [' '.join(_['tokens']) for _ in _data['sentences']]
            _index_list = []
            _input = []
            _target = []
            for i, _caption in enumerate(_captions):
                # 将_caption从str序列转换为对应的index序列
                # NOTE：里面存在几个图像只有4个caption，即存在caption为 ""（空），使用同图像另4个中的一个替代
                if len(_caption) == 0:
                    if i != 0:
                        _caption = _captions[0]
                    else:
                        _caption = _captions[i-1]
                _caption_index = str2index(_caption, self.token2index)
                _index_list.append(_caption_index[:20]) # 只保留前20个分词，多余的直接舍弃
                # 需要将其扩展到长度为20
                _input_index = [0] + _caption_index  # 首部添加 <bos>
                _target_index = _caption_index       
                _input.append(fill_list(_input_index, 0, 20))
                _target.append(fill_list(_target_index, -1, 20))
                
            train_gts.append(_index_list)
            _id = _image_id
            train_input[_id] = np.array(_input)
            train_target[_id] = np.array(_target)
        
        # 保存到文件中
        with open(self.misc_gts_file, 'wb') as f:
            pickle.dump(train_gts, f)
            
        with open(self.sent_input_file, 'wb') as f:
            pickle.dump(train_input, f)
            
        with open(self.sent_target_file, 'wb') as f:
            pickle.dump(train_target, f)

    def save_cider_cache(self,):
        # CIDEr score cache file，用于训练过程中快速计算CIDEr得分，避免训练速度瓶颈
        def precook(words, n=4, out=False):
            """
            Takes a string as input and returns an object that can be given to
            either cook_refs or cook_test. This is optional: cook_refs and cook_test
            can take string arguments as well.
            :param s: string : sentence to be converted into ngrams
            :param n: int    : number of ngrams for which representation is calculated
            :return: term frequency vector for occuring ngrams
            """
            counts = defaultdict(int)
            for k in range(1,n+1):
                for i in range(len(words)-k+1):
                    ngram = tuple(words[i:i+k])
                    counts[ngram] += 1
            return counts

        def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
            '''Takes a list of reference sentences for a single segment
            and returns an object that encapsulates everything that BLEU
            needs to know about them.
            :param refs: list of string : reference sentences for some image
            :param n: int : number of ngrams for which (ngram) representation is calculated
            :return: result (list of dict)
            '''
            return [precook(ref, n) for ref in refs]

        def cook_test(test, n=4):
            '''Takes a test sentence and returns an object that
            encapsulates everything that BLEU needs to know about it.
            :param test: list of string : hypothesis sentence for some image
            :param n: int : number of ngrams for which (ngram) representation is calculated
            :return: result (dict)
            '''
            return precook(test, n, True)

        # 读取 sent_target_file.pkl，生成保存 Flickr_train_cider.pkl
        target_seqs = pickle.load(open(self.sent_target_file, 'rb'), encoding='bytes')

        # 读取 Flickr_train_gts.pkl 文件
        gts = pickle.load(open(self.misc_gts_file, 'rb'), encoding='bytes')

        # 核心操作，统计词频（分词词汇的index，非实际str信息）
        crefs = []
        for gt in gts:
            crefs.append(cook_refs(gt))

        document_frequency = defaultdict(float)
        for refs in crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                document_frequency[ngram] += 1
        ref_len = np.log(float(len(crefs)))
        pickle.dump(
            {'document_frequency': document_frequency, 'ref_len': ref_len }, 
            open(self.misc_cider_file, 'wb')
        )


if __name__ == "__main__":
    args = parse_args()
    data_preparation = DataPreparation(args)
    data_preparation.run()

# python data_preparation.py \
#     --anno_file ./karpathy_split/dataset_coco.json \
#     --save_path ./preparation_output/mscoco \
#     --dataset_name coco
