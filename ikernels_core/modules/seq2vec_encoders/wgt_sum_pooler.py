import torch
import torch.nn
from overrides import overrides
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import masked_softmax


@Seq2VecEncoder.register("wgt_sum_pooler")
class WgtSumPooler(Seq2VecEncoder):
    def __init__(self, embedding_dim: int = None, verbose=False, wgt_activation = 'softmax', att_threshold : float =None):
        '''
        :param embedding_dim:
        :param verbose:
        :param wgt_activation: can be softmax or sigmoid
        '''
        super().__init__()
        self._embedding_dim = embedding_dim
        self._wgt_layer = torch.nn.Linear(embedding_dim, 1)
        self._verbose = verbose
        self._softmax_activation = wgt_activation == "softmax"
        self._att_threshold = att_threshold

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def set_verbose(self, verbose=True):
        self._verbose = verbose

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        raw_wgt = self._wgt_layer(tokens)
        if self._softmax_activation:
            wgt = masked_softmax(raw_wgt.squeeze(-1), dim=-1, mask=mask)
        else:
            wgt = torch.sigmoid(raw_wgt).squeeze(-1)
        if self._verbose:
            print(raw_wgt, wgt)
        if self._att_threshold is not None:
            wgt = torch.where(wgt > self._att_threshold, wgt, torch.zeros_like(wgt))
        wgt_tokens = tokens * wgt.unsqueeze(-1)

        return wgt_tokens.sum(dim=-2), wgt
