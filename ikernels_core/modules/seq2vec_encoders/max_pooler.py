import torch.nn
from overrides import overrides

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("max_pooler")
class MaxPooler(Seq2VecEncoder):
    """

    """
    def __init__(self, embedding_dim: int = 768, dim=-2):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._dim = dim


    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        # tokens is assumed to have shape (batch_size, sequence_length, embedding_dim).  We just
        # want the first token for each instance in the batch.

        # print(tokens.shape, output.shape, tokens.is_cuda, output.is_cuda)
        return tokens.max(dim=self._dim)[0]
