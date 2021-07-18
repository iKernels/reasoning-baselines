from typing import Dict
from ikernels_core.evaluators.evaluator import Evaluator, MultiClasswithCELOSSEvaluator


@Evaluator.register("named_classification_multiclass_metrics_ce_loss")
class NamedMultiClasswithCELOSSEvaluator(MultiClasswithCELOSSEvaluator):
    def __init__(self,
                 evaluator_name="",
                 **kwargs):
        super().__init__(**kwargs)
        self._evaluator_name = evaluator_name

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        rez = dict()

        if self._prf_metrics is not None:
            macro_f1 = []

            for key in self._prf_metrics:
                prf_metrics = self._prf_metrics[key].get_metric(reset)
                if not self._brief_report:
                    rez[f"{self._evaluator_name}_p_{key}"] = prf_metrics["precision"]
                    rez[f"{self._evaluator_name}_r_{key}"] = prf_metrics["recall"]
                rez[f"{self._evaluator_name}_f1_{key}"] = prf_metrics["f1"]
                if self._negative_label is not None and key != self._negative_label:
                    macro_f1.append(prf_metrics["f1"])


            rez[f"{self._evaluator_name}_mf1"] = float(sum(macro_f1) / float(len(macro_f1)))
        rez[f"{self._evaluator_name}_a"] = self._accuracy.get_metric(reset)
        return rez