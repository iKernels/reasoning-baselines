from typing import Iterator
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
import pandas as pd
from ikernels_core.readers.base.instance_generator import InstanceGenerator
import logging

logger = logging.getLogger(__name__)
from ikernels_core.readers.fever.kgat_reading_utils import process_sent, process_wiki_title
from collections import defaultdict

@DatasetReader.register("fever_kgat_fmt_json_reader")
class FeverIkrnJsonReader(DatasetReader):
    '''
    Reader for the corpus serialized as a pandas dataframe
    '''
    def __init__(self,
                 instance_generator: InstanceGenerator,
                 max_sentences_to_keep: int = None,
                 label_mappings=(("SUPPORTS", 0), ("REFUTES", 1), ("NOT ENOUGH INFO", 2)),
                 threshold: float = None,
                 prepend_page_name=False,
                 page_name_delimiter=". ",
                 num_examples_to_print: int = 3,
                 lines_to_read: int = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._threshold = threshold
        self._num_examples_to_print = num_examples_to_print
        self._max_sentences_to_keep = max_sentences_to_keep
        self._prepend_page_name = prepend_page_name
        self._instance_generator = instance_generator
        self._label_mappings = dict(label_mappings)
        self._lines_to_read = lines_to_read
        self._page_name_delimiter = page_name_delimiter

    def apply_token_indexers(self, instance: Instance) -> None:
        self._instance_generator.apply_token_indexers(instance)

    def _read(self, file_path: str) -> Iterator[Instance]:

        # {
        #     "id": 75397,
        #     "evidence": [
        #         ["Fox_Broadcasting_Company", 0, "The Fox Broadcasting Company LRB often shortened to Fox and stylized "
        #                                         "as FOX RRB is an American English language commercial broadcast"
        #                                         " television network that is owned by the Fox Entertainment Group "
        #                                         "subsidiary of 21st Century Fox .", 1.0],
        #         ["Nikolaj_Coster-Waldau", 7,  "He then played Detective John Amsterdam in the short lived Fox "
        #                                       "television series New Amsterdam LRB 2008 RRB , as well as appearing "
        #                                       "as Frank Pike in the 2009 Fox television film Virtuality ,"
        #                                       " originally intended as a pilot .", 1.0],
        #         ["Nikolaj_Coster-Waldau", 8, "He became widely known to a broad audience for his current role as "
        #                                      "Ser Jaime Lannister , in the HBO series Game of Thrones .",
        #          0.1474965512752533],
        #         ["Nikolaj_Coster-Waldau", 9, "In 2017 , he became one of the highest paid actors on television and "
        #                                      "earned 2 million per episode of Game of Thrones .",
        #          -0.23199528455734253],
        #         ["Nikolaj_Coster-Waldau", 3, "Since then he has appeared in numerous films in his native Scandinavia"
        #                                      " and Europe in general , including Headhunters LRB 2011 RRB and "
        #                                      "A Thousand Times Good Night LRB 2013 RRB .",
        #          -0.7567344307899475]],
        #     "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.", "label": "SUPPORTS"
        # }
        cou = 0

        logger.info(f"Reading from {file_path}")

        df_check = pd.read_json(file_path, lines=True, nrows=self._lines_to_read)

        cols = ["id", "label", "claim", "evidence"]

        if "label" not in df_check.columns:
            df_check["label"] = "NOT ENOUGH INFO"
        labels = defaultdict(int)
        for claim_id, label, claim_text, evidence in df_check[cols].values:
            label = self._label_mappings.get(label, label) if self._label_mappings is not None else label

            if self._max_sentences_to_keep is not None:
                evidence = evidence[:self._max_sentences_to_keep]


            aids = ["%s_%d" % (e[0], e[1]) for e in evidence]

            if self._prepend_page_name:
                for e in evidence:
                    e[2] = "%s%s%s" % (process_wiki_title(e[0], keep_underscore=False), self._page_name_delimiter, e[2])

            evidence_text = [process_sent(e[2]) for e in evidence]

            if len(evidence_text) == 0:
                evidence_text = ["none"]

            yield self._instance_generator.generate_instance(
                    aids, claim_text, evidence_text, label, claim_id, cou < self._num_examples_to_print
                )

            labels[label] += 1
            cou += 1

        print("Labels: ", labels)
