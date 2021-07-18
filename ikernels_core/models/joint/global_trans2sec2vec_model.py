from typing import Dict, Union

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.initializers import InitializerApplicator
from ikernels_core.evaluators.evaluator import MultiClasswithCELOSSEvaluator, Evaluator
from transformers import AutoModel
from allennlp.modules import FeedForward
TextFieldTensors = Dict[str, Dict[str, torch.Tensor]]


@Model.register("global_trans2sec2vec_model")
class GlobalTrans2Sec2VecModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 transformer_model: Union[str, Model],
                 claim_encoder: Seq2VecEncoder,
                 classification_layer: FeedForward,
                 dropout: float = 0.0,
                 index: str = "transfomer_tokens",
                 trainable: bool = True,
                 freeze_word_embeddings=False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 evaluator: Evaluator = MultiClasswithCELOSSEvaluator(num_labels=3),
                 store_pooled_output: bool = False) -> None:
        super().__init__(vocab)

        if isinstance(transformer_model, str):
            self.transformer_model = AutoModel.from_pretrained(transformer_model)
        else:
            self.transformer_model = transformer_model

        self._store_pooled_output = store_pooled_output

        for param in self.transformer_model.parameters():
            param.requires_grad = trainable

        if freeze_word_embeddings:
            for name, param in self.transformer_model.named_parameters():
                if name.startswith("embeddings"):
                    param.requires_grad = False

        self._dropout = torch.nn.Dropout(p=dropout)
        self._claim_encoder = claim_encoder
        self._classification_layer = classification_layer

        self._keep_token_ids = transformer_model in ["bert", "xlnet", "albert"]
        self._evaluator = evaluator
        self._index = index
        initializer(self)

    def get_cls_representation(self, transfomer_output):
        return transfomer_output.pooler_output

    def forward(self,
                input_tokens: TextFieldTensors,
                qid: Dict[str, torch.Tensor] = None,
                aid: Dict[str, torch.Tensor] = None,
                labels: torch.Tensor = None,
                ) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = input_tokens[self._index]["token_ids"], input_tokens[self._index]["mask"]
        type_ids = input_tokens[self._index]['type_ids']
        batch_size, text_num, text_sze = input_ids.shape

        r_input_ids = input_ids.view((batch_size * text_num), text_sze)
        r_attention_mask = attention_mask.view((batch_size * text_num), text_sze)
        r_token_type_ids = type_ids.view((batch_size * text_num), text_sze) if type_ids is not None else None

        transformer_output = self.transformer_model(input_ids=r_input_ids,
                                                    token_type_ids=r_token_type_ids,
                                                    attention_mask=r_attention_mask)

        pooled = self.get_cls_representation(transformer_output)
        pooled = pooled.view(batch_size, text_num, pooled.shape[-1])
        pooled_att_mask = attention_mask[:, :, :1].squeeze(-1)
        pooled = self._dropout(pooled)
        out_claim_encoder = self._claim_encoder(pooled, pooled_att_mask)

        if type(out_claim_encoder) is tuple:
            encoded_seq, wgt = out_claim_encoder
        else:
            encoded_seq = out_claim_encoder
            wgt = None

        logits = self._classification_layer(encoded_seq)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"label_logits": logits, "probs": probs, "qid": qid, "aid": aid}

        if wgt is not None:
            output_dict["wgt"] = wgt

        if self._store_pooled_output:
            output_dict["pooled_output"] = pooled

        if labels is not None:
            loss = self._evaluator.evaluate(logits, labels, qid, aid)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._evaluator.get_metrics(reset)
