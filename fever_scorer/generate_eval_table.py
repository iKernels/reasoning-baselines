import argparse
import logging
import pandas as pd
import codecs
import json
from fever_scorer.fever_one_system_eval import evaluate_claim_level_pred
import os
import re
CLASS_ID_TO_LABEL_MAP = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
LABEL_TO_CLASS_ID_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}


def get_evaluation_table(of_dev_fever,
                         allennlp_prediction_folder,
                         allennlp_prediction_file_pattern=None,
                         logits_field="label_logits"):
    # read the json file with the predictions
    fname_pattern = re.compile(allennlp_prediction_file_pattern) if allennlp_prediction_file_pattern is not None else\
        None
    fnames = [fname for fname in os.listdir(allennlp_prediction_folder) if fname_pattern is None or
              fname_pattern.search(fname)]

    logging.info(f"Evaluating on the following files: {fnames}")
    headers = ['title', 'FEVER', 'LA', 'Ev P', 'Ev R', 'Ev F1']
    rez_tuples = []
    for fname in fnames:
        prediction_file_path = os.path.join(allennlp_prediction_folder, fname)
        df_predictions = pd.read_json(prediction_file_path, lines=True)
        logging.info(f"Read predictions from {prediction_file_path}")

        # strict_score, label_accuracy, precision, recall, f1
        fever_scores = evaluate_claim_level_pred(of_dev_fever,
                                                 df_predictions,
                                                 predictions_col=logits_field,
                                                 qid_to_predicted_evidence_dict=None)

        rez_tuples.append([fname] + [x * 100 for x in fever_scores])
    return pd.DataFrame(rez_tuples, columns=headers)


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Generates a fever evaluation table on a number of outputs')

    parser.add_argument('--gold_standard_fever', help='original gold-standard FEVER reference data file.',
                        required=False)

    parser.add_argument('--allennlp_prediction_folder', help='folder with json file with allennlp predictions',
                        required=False,
                        default=None)

    parser.add_argument('--allennlp_prediction_file_pattern', help='we will run evaluation on the files in the'
                                                                   ' allennlp_prediction_folder which match the '
                                                                   'pattern',
                        required=False,
                        default=None)
    parser.add_argument('--logits_field', help='name of the field with the predicted logits for an example. Use '
                                               '"prob" for kgat models and "label_logits" for all others',
                        required=False,
                        default="label_logits")
    return parser


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    parser = get_arg_parser()
    args = parser.parse_args()

    #read the gold data
    of_dev_fever = None
    if args.gold_standard_fever:
        with codecs.open(args.gold_standard_fever, "rb", "utf-8") as f:
            of_dev_fever = [json.loads(x) for x in f]

    df_eval = get_evaluation_table(of_dev_fever, args.allennlp_prediction_folder, args.allennlp_prediction_file_pattern,
                                   args.logits_field)

    print(df_eval.to_csv(sep="\t", float_format="%.2f", index=False))