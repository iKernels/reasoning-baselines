from allennlp.common.registrable import Registrable
import logging
from typing import Dict
import pathlib
from transformers import AutoTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

class InstanceGenerator(Registrable):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer_model_name_or_path: str = None,
                 claim_first=True,
                 max_sequence_length: int = 512,
                 force_segment_ids=False) -> None:


        self._token_indexers = token_indexers or {"transfomer_tokens": SingleIdTokenIndexer()}

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_or_path, use_fast=False)

        self._max_sequence_length = max_sequence_length

        self._force_segment_ids = force_segment_ids

        self._claim_first = claim_first

    def generate_instance(self, aid, claim_text, evidence_text, label, qid, verbose=False, **kwargs):
        pass