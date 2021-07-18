from allennlp.data import Instance
from allennlp.data.fields import LabelField, MetadataField
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField, ListField
from ikernels_core.readers.base.instance_generator import InstanceGenerator

@InstanceGenerator.register("sep_instance_generator")
class SepInstanceGenerator(InstanceGenerator):
    '''
    Reader for the corpus serialized as a pandas dataframe
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def apply_token_indexers(self, instance: Instance) -> None:
        for f in instance.fields["input_tokens"]:
            f.token_indexers = self._token_indexers

    def generate_instance(self, aid, claim_text, evidence_text, label, qid, verbose=False):
        input_text_fields = []
        token_type_id_fields = []
        if self._force_segment_ids:
            len_q = len(self._tokenizer.encode_plus(claim_text, add_special_tokens=True,
                                                    max_length=self._max_sequence_length, truncation=True)["input_ids"])
        else:
            len_q = 0

        for e in evidence_text:
            if self._claim_first:
                inputs = self._tokenizer.encode_plus(claim_text, e,
                                                 add_special_tokens=True, max_length=self._max_sequence_length,
                                                 truncation_strategy='only_second', truncation=True)
            else:
                inputs = self._tokenizer.encode_plus(e, claim_text,
                                                     add_special_tokens=True, max_length=self._max_sequence_length,
                                                     truncation_strategy='only_first', truncation=True)
            input_ids = inputs["input_ids"]
            word_pieces = self._tokenizer.convert_ids_to_tokens(input_ids)
            tokens = [Token(t, text_id=token_id) for t, token_id in zip(word_pieces, input_ids)]
            text_field = TextField(tokens)
            # , token_indexers=self._token_indexers)

            input_text_fields.append(text_field)
            # this is needed for running KGAT models
            if self._force_segment_ids:
                token_type_ids = [0] * len_q + [1] * (len(input_ids) - len_q)
                token_type_id_fields.append(SequenceLabelField(token_type_ids, text_field))

        question_id_field = MetadataField(str(qid))
        answer_id_field = MetadataField(aid)
        fields = {"input_tokens": ListField(input_text_fields),
                  "qid": question_id_field,
                  "aid": answer_id_field}

        if self._force_segment_ids:
            fields["token_segment_ids"] = ListField(token_type_id_fields)

        if label is not None:
            label_field = LabelField(label=label, skip_indexing=True)
            fields["labels"] = label_field

        inst = Instance(fields)
        if verbose:
            print(qid, aid, claim_text, evidence_text, label)
            print(inst)
        return inst
