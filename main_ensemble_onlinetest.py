"""
集成模型在MSCOCO Online Test数据集测试
"""
import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
from models.att_ensemble_model import AttEnsembleModel, AttEnsembleTransformer
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from evaluation.online_tester import OnlineTester
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file

class EnsembleTester(object):
    def __init__(self, args):
        super(EnsembleTester, self).__init__()
        self.args = args
        self.device = torch.device("cuda")

        self.setup_logging()
        self.setup_network()
        self.evaler = OnlineTester(
            eval_ids = cfg.DATA_LOADER.TEST_4W_ID,   # MSCOCO 在线测试图片ID文件
            gv_feat = cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS
        )
        """
        self.evaler = Evaler(
            eval_ids=cfg.DATA_LOADER.TEST_ID,
            gv_feat=cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats=cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile=cfg.INFERENCE.TEST_ANNFILE
        )
        """

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, 'Ensemble_OnlineTest_' + cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def setup_network(self):
        # 创建集成模型的每一个子模型，并导入模型参数
        _models = []
        _model_folders = self.args.model_folders
        _model_resumes = self.args.model_resumes
        assert len(_model_folders) == len(_model_resumes)
        for i in range(len(_model_folders)):
            if self.args.model_types is None:
                tmp_type = cfg.MODEL.TYPE
            else:
                # VPNet VP_AoA VP_UPDOWN
                tmp_type = self.args.model_types[i]
            # tmp = models.create(cfg.MODEL.TYPE)
            tmp = models.create(tmp_type)
            tmp = torch.nn.DataParallel(tmp).cuda()
            tmp_snapshot_file = os.path.join(_model_folders[i],
                                               "snapshot",
                                               "caption_model_"+str(_model_resumes[i])+".pth")
            # print('sub model loaded from %s' % tmp_snapshot_file)
            self.logger.info('sub model loaded from %s' % tmp_snapshot_file)
            tmp.load_state_dict(torch.load(tmp_snapshot_file,
                                           map_location=lambda storage, loc: storage))
            _models.append(tmp)
        weights = None
        if self.args.weights is not None:
            weights = [float(_) for _ in self.args.weights]
            
        """
        if 'Transformer' in self.args.model_types[0]:
            model = AttEnsembleTransformer(_models, weights=weights)
        else:
            model = AttEnsembleModel(_models, weights=weights)
        """
        model = AttEnsembleTransformer(_models, weights=weights)
        self.model = torch.nn.DataParallel(model).cuda()
        # self.model.eval()

    def eval(self, epoch):
        self.evaler(self.model, 'online_test_ensemble')
        self.logger.info('######## Ensemble Online Test ########')
        self.logger.info('Inference result saved to result_online_test_ensemble.json')

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', default=None, type=str)
    parser.add_argument('--model_folders', nargs='+', required=True, default=None)
    parser.add_argument("--model_resumes", nargs='+', required=True, default=None)
    parser.add_argument('--model_types', nargs='+', required=False, default=None)
    parser.add_argument("--weights", nargs='+', required=False, default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    tester = EnsembleTester(args)
    tester.eval("Ensemble Models")
