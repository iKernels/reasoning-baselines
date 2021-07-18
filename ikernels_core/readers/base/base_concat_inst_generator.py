from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.tokenizers import Token
from ikernels_core.readers.base.instance_generator import InstanceGenerator
import numpy as np
from itertools import chain

@InstanceGenerator.register("concat_instance_generator")
class ConcatInstanceGenerator(InstanceGenerator):
    '''
    Reader for the corpus serialized as a pandas dataframe
    '''
    def __init__(self,
                 truncate_after_concat=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._truncate_after_concat = truncate_after_concat
        self._sequence_delimiter = self._tokenizer.sep_token_id
        if hasattr(self._tokenizer, "num_added_tokens"):
            # how many "service tokens" (e.g. [CLS], [SEP] or else) will be added when encoding a pair of texts?
            self._num_added_tokens_pair = self._tokenizer.num_added_tokens(True)

            # number_of_tokens_added_when_encoding_a_pair -  number_of_tokens_added_when_encoding_a_single_text
            self._offset = self._num_added_tokens_pair - self._tokenizer.num_added_tokens(False)
        else:
            self._num_added_tokens_pair = self.num_added_tokens(True)
            self._offset = self._num_added_tokens_pair - self.num_added_tokens(False)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["input_tokens"].token_indexers = self._token_indexers

    def get_number_added_tokens(self):
        return len(self._tokenizer.encode_plus("")['input_ids'])

    def num_added_tokens(self, pair=False):
        """
        copy-pasted from hugging-face
        Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self._tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def get_input_ids(self, text, **kwargs):
        if isinstance(text, str):
            return self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text, **kwargs))
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
            return self._tokenizer.convert_tokens_to_ids(text)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
            return text
        else:
            print("TEXT: ", text)
            raise ValueError(
                "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.")

    def generate_concatenated_ids(self, question_text, answer_text, max_length=False, ev_first=False):
        text_ids = [self.get_input_ids(question_text)] + [self.get_input_ids(a) for a in answer_text]

        total_added_tokens = (self._num_added_tokens_pair - self._offset) * len(text_ids) + self._offset

        total_len = sum([len(t) for t in text_ids]) + total_added_tokens

        if max_length and total_len > max_length:
            num_tokens_to_remove = total_len - max_length
            for _ in range(num_tokens_to_remove):
                lengths = [len(t) for t in text_ids]
                id_to_truncate = np.argmax(lengths)
                text_ids[id_to_truncate] = text_ids[id_to_truncate][:-1]
        if ev_first:
            t1 = text_ids[-1]
            t2 = list(chain(*[t + [self._sequence_delimiter] for t in text_ids[:-1]]))[:-1]
        else:
            t1 = text_ids[0]
            t2 = list(chain(*[t + [self._sequence_delimiter] for t in text_ids[1:]]))[:-1]
        return t1, t2



    def generate_instance(self, aid, claim_text, evidence_text, label, qid, verbose=False):

        if self._truncate_after_concat:
            sep = " %s " % self._tokenizer.sep_token
            joint_ev_text = sep.join(evidence_text)
        else:
            claim_text, joint_ev_text = self.generate_concatenated_ids(claim_text, evidence_text,
                                                                       max_length=self._max_sequence_length)

        if self._claim_first:
            inputs = self._tokenizer.encode_plus(claim_text, joint_ev_text,
                                             add_special_tokens=True, max_length=self._max_sequence_length,
                                             truncation_strategy='longest_first', truncation=True)
        else:
            inputs = self._tokenizer.encode_plus(joint_ev_text, claim_text,
                                                 add_special_tokens=True, max_length=self._max_sequence_length,
                                                 truncation_strategy='longest_first', truncation=True)

        input_ids = inputs["input_ids"]
        word_pieces = self._tokenizer.convert_ids_to_tokens(input_ids)
        tokens = [Token(t, text_id=token_id) for t, token_id in zip(word_pieces, input_ids)]
        text_field = TextField(tokens)
        question_id_field = MetadataField(str(qid))
        answer_id_field = MetadataField(aid)
        fields = {"input_tokens": text_field,
                  "qid": question_id_field,
                  "aid": answer_id_field}
        if label is not None:
            label_field = LabelField(label=label, skip_indexing=True)
            fields["labels"] = label_field

        inst = Instance(fields)
        if verbose:
            print(qid, aid, self._tokenizer.decode(claim_text), self._tokenizer.decode(joint_ev_text), label)
            print(inst)
        return inst
