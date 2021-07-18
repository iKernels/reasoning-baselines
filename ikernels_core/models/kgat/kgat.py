# THIS IS A WRAPPER FOR THE KGAT MODEL FROM http://github.com/thunlp/KernelGAT/tree/masterc/kgat
# The core code was copied from  https://github.com/thunlp/KernelGAT/blob/master/kgat/models.py, we only fit it into
# the AllenNLP framework interface
# KGAT code is distributed under the MIT license:
# MIT License
#
# Copyright (c) 2019 THUNLP
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, Union
from ikernels_core.models.kgat.kgat_utils import kernal_mus, kernel_sigmas
from ikernels_core.evaluators.evaluator import Evaluator, MultiClasswithCELOSSEvaluator
from transformers import AutoModel
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
import torch.nn.functional as F
TextFieldTensors = Dict[str, Dict[str, torch.Tensor]]

@Model.register("kgat_model")
class KGAT(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 transformer_model: Union[str, Model],
                 kernel: int = 21,
                 layer: int = 1,
                 bert_hidden_dim: int = 768,
                 dropout: float = 0.6,
                 num_labels: int = None,
                 index: str = "transfomer_tokens",
                 label_namespace: str = "labels",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 evaluator: Evaluator = MultiClasswithCELOSSEvaluator(num_labels=3, use_nll=True),
                 store_pooled_output: bool = False) -> None:
        super().__init__(vocab)

        self.bert_hidden_dim = bert_hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels

        if isinstance(transformer_model, str):
            self.transformer_model = AutoModel.from_pretrained(transformer_model)
        else:
            self.transformer_model = transformer_model

        self.nlayer = layer
        self.kernel = kernel
        self.proj_inference_de = nn.Linear(self.bert_hidden_dim * 2, self.num_labels)
        self.proj_att = nn.Linear(self.kernel, 1)
        self.proj_input_de = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)
        self.proj_gat = nn.Sequential(
            Linear(self.bert_hidden_dim * 2, 128),
            ReLU(True),
            Linear(128, 1)
        )
        self.proj_select = nn.Linear(self.kernel, 1)
        self.mu = torch.nn.Parameter(torch.FloatTensor(kernal_mus(self.kernel)).view(1, 1, 1, 21),
                                     requires_grad=False)
        self.sigma = torch.nn.Parameter(torch.FloatTensor(kernel_sigmas(self.kernel)).view(1, 1, 1, 21),
                                        requires_grad=False)
        self._label_namespace = label_namespace
        self._evaluator = evaluator
        self._index = index
        initializer(self)
        self._store_pooled_output = store_pooled_output

    def forward(self,
                input_tokens: TextFieldTensors,
                token_segment_ids: Dict[str, torch.LongTensor] = None,
                qid: Dict[str, torch.Tensor] = None,
                aid: Dict[str, torch.Tensor] = None,
                labels: torch.Tensor = None,
                ) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = input_tokens[self._index]["token_ids"], input_tokens[self._index]["mask"]
        type_ids = input_tokens[self._index]['type_ids']


        batch_size, text_num, text_sze = input_ids.shape

        inp_tensor = input_ids.view((batch_size * text_num), text_sze)
        msk_tensor = attention_mask.view((batch_size * text_num), text_sze)
        seg_tensor = token_segment_ids.view((batch_size * text_num), text_sze)
        # in seg_tensor we always assign 0 to claim and 1 to evidence tokens, regardless of the transformer used
        #
        # r_token_type_ids is the actual segment encodings used by a given transformer
        # should be the same as seg_tensor for BERT, but will be simply all zeroes for RoBERTa, for example
        r_token_type_ids = type_ids.view((batch_size * text_num), text_sze) if type_ids is not None else None

        transformer_output = self.transformer_model(input_ids=inp_tensor,
                                                    token_type_ids=r_token_type_ids,
                                                    attention_mask=msk_tensor)
        inputs_hiddens = transformer_output.last_hidden_state
        inputs = transformer_output.pooler_output

        mask_text = msk_tensor.view(-1, text_sze).float()
        mask_text[:, 0] = 0.0
        mask_claim = (1 - seg_tensor.float()) * mask_text
        mask_evidence = seg_tensor.float() * mask_text
        inputs_hiddens = inputs_hiddens.view(-1, text_sze, self.bert_hidden_dim)
        inputs_hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=2)
        log_pooling_sum = self.get_intersect_matrix(inputs_hiddens_norm, inputs_hiddens_norm, mask_claim, mask_evidence)
        log_pooling_sum = log_pooling_sum.view([-1, text_num, 1])
        select_prob = F.softmax(log_pooling_sum, dim=1)
        # print(select_prob)
        inputs = inputs.view([-1, text_num, self.bert_hidden_dim])
        inputs_hiddens = inputs_hiddens.view([-1, text_num, text_sze, self.bert_hidden_dim])
        inputs_att_de = []
        for i in range(text_num):
            outputs, outputs_de = self.self_attention(inputs, inputs_hiddens, mask_text, mask_text, i, text_sze, text_num)
            inputs_att_de.append(outputs_de)
        inputs_att = inputs.view([-1, text_num, self.bert_hidden_dim])
        inputs_att_de = torch.cat(inputs_att_de, dim=1)
        inputs_att_de = inputs_att_de.view([-1, text_num, self.bert_hidden_dim])
        inputs_att = torch.cat([inputs_att, inputs_att_de], -1)
        inference_feature = self.proj_inference_de(inputs_att)
        class_prob = F.softmax(inference_feature, dim=2)
        prob = torch.sum(select_prob * class_prob, 1)
        prob = torch.log(prob)

        output_dict = {"prob": prob, "qid": qid, "aid": aid, "label_logits": prob, 'wgt': select_prob}

        if self._store_pooled_output:
            output_dict["pooled_output"] = inputs

        if labels is not None:
            loss = self._evaluator.evaluate(prob, labels, qid, aid)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._evaluator.get_metrics(reset)


    def self_attention(self, inputs, inputs_hiddens, mask, mask_evidence, index, max_len, evi_num):
        if self.mu.is_cuda:
            # idx = torch.LongTensor([index]).cuda()
            idx = torch.LongTensor([index])
            idx = idx.to(self.mu.device)
        else:
            idx = torch.LongTensor([index])

        mask = mask.view([-1, evi_num, max_len])
        mask_evidence = mask_evidence.view([-1, evi_num, max_len])
        own_hidden = torch.index_select(inputs_hiddens, 1, idx)
        own_mask = torch.index_select(mask, 1, idx)
        own_input = torch.index_select(inputs, 1, idx)
        own_hidden = own_hidden.repeat(1, evi_num, 1, 1)
        own_mask = own_mask.repeat(1, evi_num, 1)
        own_input = own_input.repeat(1, evi_num, 1)

        hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=-1)
        own_norm = F.normalize(own_hidden, p=2, dim=-1)

        att_score = self.get_intersect_matrix_att(hiddens_norm.view(-1, max_len, self.bert_hidden_dim), own_norm.view(-1, max_len, self.bert_hidden_dim),
                                                  mask_evidence.view(-1, max_len), own_mask.view(-1, max_len))
        att_score = att_score.view(-1, evi_num, max_len, 1)
        #if index == 1:
        #    for i in range(self.evi_num):
        #print (att_score.view(-1, self.evi_num, self.max_len)[0, 1, :])
        denoise_inputs = torch.sum(att_score * inputs_hiddens, 2)
        weight_inp = torch.cat([own_input, inputs], -1)
        weight_inp = self.proj_gat(weight_inp)
        weight_inp = F.softmax(weight_inp, dim=1)
        outputs = (inputs * weight_inp).sum(dim=1)
        weight_de = torch.cat([own_input, denoise_inputs], -1)
        weight_de = self.proj_gat(weight_de)
        weight_de = F.softmax(weight_de, dim=1)
        outputs_de = (denoise_inputs * weight_de).sum(dim=1)
        return outputs, outputs_de

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1], 1)
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1) / (torch.sum(attn_q, 1) + 1e-10)
        log_pooling_sum = self.proj_select(log_pooling_sum).view([-1, 1])
        return log_pooling_sum

    def get_intersect_matrix_att(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1])
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        log_pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(log_pooling_sum, min=1e-10))
        log_pooling_sum = self.proj_att(log_pooling_sum).squeeze(-1)
        log_pooling_sum = log_pooling_sum.masked_fill_((1 - attn_q).bool(), -1e4)
        log_pooling_sum = F.softmax(log_pooling_sum, dim=1)
        return log_pooling_sum
