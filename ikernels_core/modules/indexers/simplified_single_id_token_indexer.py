from typing import Dict, List
import itertools

from overrides import overrides
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer

# Note: this is a minor modification of the original AllenNLP  SingleIdTokenIndexer


@TokenIndexer.register("simplified_single_id")
class SimplifiedSingleIdTokenIndexer(SingleIdTokenIndexer):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(kwargs)


    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            if getattr(token, 'text_id', None) is not None:
                indices.append(token.text_id)
            else:
                text = self._get_feature_value(token)
                if self.namespace is None:
                    # We could have a check here that `text` is an int; not sure it's worth it.
                    indices.append(text)  # type: ignore
                else:
                    if self.lowercase_tokens:
                        text = text.lower()
                    indices.append(vocabulary.get_token_index(text, self.namespace))

        return {"tokens": indices}
