from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.config import cfg

from .basic_model import BasicModel
from .att_basic_model import AttBasicModel
from models.pure_transformer import subsequent_mask
import lib.utils as utils

class AttEnsembleModel(AttBasicModel):
    def __init__(self, models, weights=None):
        BasicModel.__init__(self)

        self.models = nn.ModuleList(models)           # Ensemble的每个子模型
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1    # include <BOS>/<EOS>
        self.seq_length = cfg.MODEL.SEQ_LEN

        weights = weights or [1.0] * len(self.models)
        self.register_buffer('weights', torch.tensor(weights))

    def init_hidden(self, batch_size):
        state = [m.module.init_hidden(batch_size) for m in self.models]
        return self.pack_state(state)

    # 把state展成单层的list
    def pack_state(self, state):
        return sum([list(_) for _ in state], [])

    def expand_tensor(self, x, beam_size):
        x = [utils.expand_tensor(_, beam_size) for _ in x]
        return x

    def get_logprobs_state(self, **kwargs):
        # 集成模型的前向预测函数，需要分别调用每一个子模型的get_logprobs_state函数
        wt = kwargs[cfg.PARAM.WT]                    # 没有为每个子模型创建分项，共用
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        state = kwargs[cfg.PARAM.STATE]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # 分别为每个子模型构造输入，
        # 然后调用get_logprobs_state函数或者Forward函数（需要接softmax和log函数）
        # 得到每个子模型的logprobs和state
        _output = []
        _state = []
        use_log_softmax = False
        for i, m in enumerate(self.models):
            kwargs = self.make_kwargs(wt, gv_feat[i], att_feats[i], att_mask[i], p_att_feats[i], state[i*2:(i+1)*2], **kwargs)
            if use_log_softmax:
                # 方式一：直接调用get_logprobs_state函数
                __output, __state = m.module.get_logprobs_state(**kwargs)
                _output.append(__output)                   # log(softmax())
                _state.append(__state)
            else:
                # 方式二：调用Forward函数 + softmax(logit())
                # 这个逻辑应该才是对的
                __output, __state = m.module.Forward(**kwargs)
                _output.append(F.softmax(m.module.logit(__output), dim=1))  # softmax()，缺少log()
                _state.append(__state)

        # 按照权重，融合每个子模型的logprobs
        if use_log_softmax:
            # 调用get_logprobs_state函数
            logprobs = torch.stack(_output, 2).mul(self.weights).div(self.weights.sum()).sum(-1)
        else:
            # 调用Forward函数
            logprobs = torch.stack(_output, 2).mul(self.weights).div(self.weights.sum()).sum(-1).log()
        state = _state
        return logprobs, self.pack_state(state)

    def preprocess(self, **kwargs):
        # 集成模型的特征预处理，分别调用每一个子模型的特征预处理函数
        # 返回 gv_feat, att_feats, att_mask, p_att_feats
        # 每个分量都是一个tuple，长度为self.models的个数
        # 如：gv_feat[i] 表示self.models中第i个子模型的gv_feat输出
        return tuple(zip(*[m.module.preprocess(**kwargs) for m in self.models]))
    
    def decode_beam(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        batch_size = att_feats.size(0)

        # 输入数据预处理
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        
        seq_logprob = torch.zeros(batch_size, 1, 1, device='cuda')
        log_probs = []
        selected_words = None
        seq_mask = torch.ones(batch_size, beam_size, 1, device='cuda')
        state = self.init_hidden(batch_size)
        wt = torch.zeros(batch_size, dtype=torch.long, device='cuda')

        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

        outputs = []
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]
            # print('selected words:', t, selected_words)

            for s in range(len(state)):
                state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                att_feats = self.expand_tensor(att_feats, beam_size)
                gv_feat = self.expand_tensor(gv_feat, beam_size)
                att_mask = self.expand_tensor(att_mask, beam_size)
                p_att_feats = self.expand_tensor(p_att_feats, beam_size)

                kwargs[cfg.PARAM.ATT_FEATS] = att_feats
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        # 输出语句条数
        out_size = 1
        if 'OUT_SIZE' in kwargs:
            out_size = kwargs['OUT_SIZE']
            
        if out_size > 1:
            outputs = outputs.contiguous()[:, :out_size]
            log_probs = log_probs.contiguous()[:, :out_size]
        else:
            outputs = outputs.contiguous()[:, 0]
            log_probs = log_probs.contiguous()[:, 0]
            
        return outputs, log_probs

    def _decode_beam(self, **kwargs):
        # 集成模型beam search入口函数
        beam_size = kwargs['BEAM_SIZE']
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        batch_size = gv_feat[0].size(0)  # gv_feat为所有子模型gv_feat的Tuple，因此需要随意指定一个indx获取batch_size

        sents = Variable(torch.zeros((cfg.MODEL.SEQ_LEN, batch_size), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(cfg.MODEL.SEQ_LEN, batch_size).cuda())
        self.done_beams = [[] for _ in range(batch_size)]

        import time
        for n in range(batch_size):
            # 输入都是list
            # [value_1, value_2, ..., value_l]分别用于子模型1、子模型2、...、子模型l
            state = self.init_hidden(beam_size)
            gv_feat_beam = [gv_feat[i][n:n+1].expand(beam_size, gv_feat[i].size(1)).contiguous() for i, m in enumerate(self.models)]
            att_feats_beam = [att_feats[i][n:n+1].expand(*((beam_size,) + att_feats[i].size()[1:])).contiguous() for i, m in enumerate(self.models)]
            att_mask_beam = [att_mask[i][n:n+1].expand(*((beam_size,) + att_mask[i].size()[1:])).contiguous() for i, m in enumerate(self.models)]
            p_att_feats_beam = [p_att_feats[i][n:n+1].expand(*((beam_size,) + p_att_feats[i].size()[1:])).contiguous() for i, m in enumerate(self.models)]

            # wt输入没有为每一个模型创建分项
            wt = Variable(torch.zeros(beam_size, dtype=torch.long).cuda())
            kwargs = self.make_kwargs(wt, gv_feat_beam, att_feats_beam, att_mask_beam, p_att_feats_beam, state, **kwargs)
            logprobs_t, state = self.get_logprobs_state(**kwargs)

            self.done_beams[n] = self.beam_search(state, logprobs_t, **kwargs)
            sents[:, n] = self.done_beams[n][0]['seq']
            logprobs[:, n] = self.done_beams[n][0]['logps']
        return sents.transpose(0, 1), logprobs.transpose(0, 1)


class AttEnsembleTransformer(BasicModel):
    def __init__(self, models, weights=None):
        BasicModel.__init__(self)

        self.models = nn.ModuleList(models)           # Ensemble的每个子模型
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1    # include <BOS>/<EOS>
        self.seq_length = cfg.MODEL.SEQ_LEN

        weights = weights or [1.0] * len(self.models)
        self.register_buffer('weights', torch.tensor(weights))

    def init_hidden(self, batch_size):
        state = [None for m in self.models]            
        return state
    
    def init_att_mask(self, att_mask):
        return [att_mask for m in self.models]
    
    def init_buffer(self, batch_size):
        for m in self.models:
            m.module.decoder.init_buffer(batch_size)
            
    def clear_buffer(self):
        for m in self.models:
            m.module.decoder.clear_buffer()
            
    def apply_to_states(self, batch_size, beam_size, cur_beam_size, selected_beam):
        for m in self.models:
            fn = m.module._expand_state(batch_size, beam_size, cur_beam_size, selected_beam)
            m.module.decoder.apply_to_states(fn)

    # 把state展成单层的list
    def pack_state(self, state):
        return sum([list(_) for _ in state], [])
    
    def get_logprobs_state(self, **kwargs):
        # 集成模型的前向预测函数，需要分别调用每一个子模型的get_logprobs_state函数
        # wt即上一时间步，选择的前beam size个单词
        wt = kwargs[cfg.PARAM.WT]                    # 没有为每个子模型创建分项，共用
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gx = kwargs[cfg.PARAM.GLOBAL_FEAT]
        # p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # 分别为每个子模型构造输入，
        # 然后调用get_logprobs_state函数或者Forward函数（需要接softmax和log函数）
        # 得到每个子模型的logprobs和state
        _output = []
        _state = []
        use_log_softmax = False
        for i, m in enumerate(self.models):
            if state[i] is None:
                _ys = wt.unsqueeze(1)
            else:
                _ys = torch.cat([state[i][0][0], wt.unsqueeze(1)], dim=1)
                
            _seq_mask = subsequent_mask(_ys.size(1)).to(encoder_out[i].device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(1)
            
            # 方式二：调用 decoder 前向计算 + softmax(logit())
            # 这个逻辑应该才是对的
            # print(gx[i].size(), encoder_out[i].size(), att_mask[i].size())
            __output = m.module.decoder(gx[i], _ys[:, -1].unsqueeze(-1), encoder_out[i], _seq_mask, att_mask[i]).squeeze(1)
            __state = [_ys.unsqueeze(0)]
            if not use_log_softmax:
                # _output.append(F.softmax(__output, dim=1))  # softmax()，缺少log()
                _output.append(torch.exp(F.log_softmax(__output, dim=-1)))
            else:
                _output.append(F.log_softmax(__output, dim=-1))
            _state.append(__state)

        # 按照权重，融合每个子模型的logprobs
        if not use_log_softmax:
            logprobs = torch.stack(_output, 2).mul(self.weights).div(self.weights.sum()).sum(-1).log() # [beam_size, |V|]
        else:
            logprobs = torch.stack(_output, 2).mul(self.weights).div(self.weights.sum()).sum(-1) # [beam_size, |V|]
        state = _state
        # print(logprobs.size(), len(state), state[0][0].size())
        return logprobs, state

    def preprocess(self, att_feats, att_mask):
        # 集成模型的特征预处理，分别调用每一个子模型的特征预处理函数
        # 返回 gv_feat, att_feats, att_mask, p_att_feats
        # 每个分量都是一个tuple，长度为self.models的个数
        # 如：gv_feat[i] 表示self.models中第i个子模型的gv_feat输出
        def _encode(m, x, mask):
            # 针对end-to-end模型，包含backbone encoder部分
            if hasattr(m.module, "backbone"):
                x = m.module.backbone(x)
            x = m.module.encoder(m.module.att_embed(x), mask)
            return x
        return tuple(zip(*[_encode(m, att_feats, att_mask) for m in self.models]))
    
    def expand_tensor(self, x, beam_size):
        x = [utils.expand_tensor(_, beam_size) for _ in x]
        return x
    
    def expand_state(self, state, beam_size):
        _state = []
        for _ in state:
            _[0] = _[0].squeeze(0)
            _[0] = utils.expand_tensor(_[0], beam_size)
            _[0] = _[0].unsqueeze(0)
            _state.append(_)
        return _state

    def _decode_beam(self, **kwargs):
        return self.models[0].module.decode_beam(**kwargs)
    
    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        # 集成模型beam search入口函数
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()
        
        # Transformer 各个子模型 Encoder 部分前向计算
        # 即：为每一个子模型准备输入数据
        gx, encoder_out = self.preprocess(att_feats, att_mask)
        # batch_size = gx[0].size(0)  # gv_feat为所有子模型gv_feat的Tuple，因此需要随意指定一个indx获取batch_size

        state = self.init_hidden(batch_size)
        att_mask = self.init_att_mask(kwargs[cfg.PARAM.ATT_FEATS_MASK])
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        
        outputs = []
        self.init_buffer(batch_size)
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            # [B*cur_beam_size, Vocab_size] --> [B, cur_beam_size, Vocab_size]
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            # 候选log概率，即已生成单词log概率之和，用于判断该步选择哪个单词
            # [B, cur_beam_size, Vocab_size]
            candidate_logprob = seq_logprob + word_logprob
            
            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # 基于candidate_logprob选择出前beam_size大的序列index及log概率（句子）
            # [B, beam_size], [B, beam_size]
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            # selected_beam为选择的单词在哪个beam里面，[B, 3]
            # selected_words为选择的单词在词汇表中的index，[B, 3]
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # 对decoder中的buffer进行更新
            self.apply_to_states(batch_size, beam_size, cur_beam_size, selected_beam)
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                # 相关输入复制扩展，用于下一步beam_search
                encoder_out = self.expand_tensor(encoder_out, beam_size)
                gx = self.expand_tensor(gx, beam_size)
                att_mask = self.expand_tensor(att_mask, beam_size)
                # state单独处理
                state = self.expand_state(state, beam_size)

                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats_tmp
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.clear_buffer()
        return outputs, log_probs