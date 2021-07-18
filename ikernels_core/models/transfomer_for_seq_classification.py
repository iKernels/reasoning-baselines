from typing import Dict, Union
import re
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from ikernels_core.evaluators.evaluator import Evaluator
TextFieldTensors = Dict[str, Dict[str, torch.Tensor]]
from transformers import AutoModel
from allennlp.modules import FeedForward


@Model.register("transfomer_for_seq_classification")
class TransformerForSeqClassification(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 transformer_model: Union[str, Model],
                 classification_layer: FeedForward,
                 dropout: float = 0.0,
                 index: str = "transfomer_tokens",
                 trainable: bool = True,
                 num_layers_to_freeze=None,
                 freeze_word_embeddings=False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 evaluator: Evaluator = None,
                 store_pooled_output: bool = False) -> None:
        super().__init__(vocab)

        if isinstance(transformer_model, str):
            self.transformer_model = AutoModel.from_pretrained(transformer_model)
        else:
            self.transformer_model = transformer_model

        self._store_pooled_output = store_pooled_output

        for param in self.transformer_model.parameters():
            param.requires_grad = trainable

        if num_layers_to_freeze is not None:
            r = re.compile("[^A-Za-z\_]*encoder\.layer\.([0-9]+)\..*")
            for name, param in self.transformer_model.named_parameters():
                m = r.match(name)
                if m is not None:
                    n = int(m.group(1))
                    if n < num_layers_to_freeze:
                        param.requires_grad = False

        if freeze_word_embeddings:
            for name, param in self.transformer_model.named_parameters():
                if name.startswith("embeddings"):
                    param.requires_grad = False

        self._dropout = torch.nn.Dropout(p=dropout)

        self._classification_layer = classification_layer
        self._evaluator = evaluator
        self._index = index

        initializer(self)

    def get_cls_representation(self, transfomer_output):
        return transfomer_output.pooler_output

    def forward(self,  # type: ignore
                input_tokens: TextFieldTensors,
                qid: Dict[str, torch.Tensor] = None,
                aid: Dict[str, torch.Tensor] = None,
                labels: torch.Tensor = None,
                ) -> Dict[str, torch.Tensor]:

        input_ids, attention_mask = input_tokens[self._index]["token_ids"], input_tokens[self._index]["mask"]
        type_ids = input_tokens[self._index]['type_ids']

        transformer_output = self.transformer_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=type_ids,
        )

        pooled = self.get_cls_representation(transformer_output)

        pooled = self._dropout(pooled)

        logits = self._classification_layer(pooled)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"label_logits": logits, "probs": probs, "qid": qid, "aid": aid}

        if self._store_pooled_output:
            output_dict["pooled_output"] = pooled

        if labels is not None and self._evaluator is not None:
            loss = self._evaluator.evaluate(logits, labels, qid, aid)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._evaluator.get_metrics(reset)
