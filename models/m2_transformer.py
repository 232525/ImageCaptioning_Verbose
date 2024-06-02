import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

import lib.utils as utils
from models.basic_model import BasicModel

from models.m2.m2_encoder import M2Encoder
from models.m2.m2_decoder import M2Decoder

# 构造三角矩阵，用于生成训练时单词序列的mask
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class M2Transformer(BasicModel):
    def __init__(self):
        super(M2Transformer, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        
        # att_feats encoder
        # """
        self.att_embed = nn.Identity()
        
        # m2 encoder
        # """
        d_in = cfg.MODEL.ATT_FEATS_DIM
        d_model = cfg.MODEL.ATT_FEATS_EMBED_DIM
        h = cfg.MODEL.BILINEAR.HEAD
        d_kv = d_model // h
        N = cfg.MODEL.BILINEAR.ENCODE_LAYERS
        self.encoder = M2Encoder(
            N=N, 
            d_in=d_in, 
            d_model=d_model, 
            d_k=d_kv, 
            d_v=d_kv, 
            h=h, 
            d_ff=4*d_model
        )
        
        # m2 decoder
        w_pf = False    # 是否使用Pre Fusion操作
        self.decoder = M2Decoder(
            vocab_size=self.vocab_size, 
            max_len=54, 
            N_dec=N, 
            dim=d_model, 
            num_heads=h,
            mlp_ratio=4.0,
            dropout=.1,
            w_pf=w_pf
        )
        
    def forward(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        # grid特征，无需att_mask
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        # 构造单词序列掩码
        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:,0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask=att_mask)
        decoder_out = self.decoder(gx, seq, encoder_out, seq_mask)
        return F.log_softmax(decoder_out, dim=-1)

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
        
        # grid特征本质上无需att_mask及图像全局特征gx
        # 预计算p_att_feats可以考虑保留，减少Linear层的调用
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gx = kwargs[cfg.PARAM.GLOBAL_FEAT]
        # p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # state[0][0]: [B, seq_len-1]，用于保存截止到当前步，生成的单词序列
        # ys: [B, seq_len]，seq_len为当前步数，即单词个数
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
            
        # 依靠ys，计算seq_mask
        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(1)
        
        # [B, 1, Vocab_Size] --> [B, Vocab_Size]
        decoder_out = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, seq_mask).squeeze(1)
        
        logprobs = F.log_softmax(decoder_out, dim=-1)
        
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s
        return fn

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask=att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        self.decoder.init_buffer(batch_size)
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
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
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
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                gx = utils.expand_tensor(gx, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                # p_att_feats_tmp = []
                # for p_feat in p_att_feats:
                #     p_key, p_value2 = p_feat
                #     p_key = utils.expand_tensor(p_key, beam_size)
                #     p_value2 = utils.expand_tensor(p_value2, beam_size)
                #     p_att_feats_tmp.append((p_key, p_value2))

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

        self.decoder.clear_buffer()
        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask=att_mask)
        # 预计算可考虑保留，减少Linear层调用次数
        # p_att_feats = self.decoder.precompute(encoder_out)
        self.decoder.init_buffer(batch_size)
        
        state = None
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # 预计算可考虑保留，减少Linear层调用次数
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        
        # 按时间步迭代进行推理计算
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            logprobs_t, state = self.get_logprobs_state(**kwargs)
            
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        self.decoder.clear_buffer()
        return sents, logprobs