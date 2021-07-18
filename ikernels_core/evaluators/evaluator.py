from typing import Dict

import torch

from allennlp.common.registrable import Registrable
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics.f1_measure import F1Measure


class Evaluator(Registrable):
    def __init__(self):
        raise NotImplementedError

    def evaluate(self, label_logits, labels, qid, aid):
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        raise NotImplementedError

@Evaluator.register("classification_multiclass_metrics_ce_loss")
class MultiClasswithCELOSSEvaluator(Evaluator):
    def __init__(self, num_labels, class_name_list=None, class_weights=None, is_cuda=True,
                 brief_report=False, negative_label="n", only_accuracy=False, use_nll=False):
        self._only_accuracy = only_accuracy
        if class_weights is not None:
            if is_cuda:
                weights = torch.FloatTensor(class_weights).cuda()
            else:
                weights = torch.FloatTensor(class_weights)
        else:
            weights = None

        self._negative_label = negative_label

        self._loss = torch.nn.CrossEntropyLoss(weight=weights) if not use_nll else torch.nn.NLLLoss()

        self._accuracy = CategoricalAccuracy()
        self._brief_report = brief_report
        if class_name_list is None:
            class_name_list = [str(i) for i in range(num_labels)]

        if only_accuracy:
            self._prf_metrics =None
        else:
            self._prf_metrics = dict()
            for label_int in range(num_labels):
                self._prf_metrics[class_name_list[label_int]] = F1Measure(positive_label=label_int)

    def evaluate(self, label_logits, labels, qid=None, aid=None, mask=None):
        if self._prf_metrics is not None:
            for key in self._prf_metrics:
                self._prf_metrics[key](label_logits, labels, mask=mask)
        self._accuracy(label_logits, labels, mask=mask)
        return self._loss(label_logits, labels.long().view(-1))

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        rez = dict()

        if self._prf_metrics is not None:
            macro_f1 = []

            for key in self._prf_metrics:
                prf_metrics = self._prf_metrics[key].get_metric(reset)
                if not self._brief_report:
                    rez['p_%s' % key] = prf_metrics["precision"]
                    rez['r_%s' % key] = prf_metrics["recall"]
                rez['f1_%s' % key] = prf_metrics["f1"]
                if self._negative_label is not None and key != self._negative_label:
                    macro_f1.append(prf_metrics["f1"])

            rez['m_f1'] = float(sum(macro_f1) / float(len(macro_f1)))
        rez['accuracy'] = self._accuracy.get_metric(reset)
        return rez