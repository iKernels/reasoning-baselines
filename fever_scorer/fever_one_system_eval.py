import argparse
import logging
import pandas as pd
import codecs
import json
from fever_scorer.eval_utils import evaluate_claim_level_pred
import os

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Rewrites the data in a processable way')

    parser.add_argument('--gold_standard_fever', help='official gold standard file. If not provided, '
                                                      'then the predictions will be simply stored in a file '
                                                      'specified by the --output_file parameter',
                        required=False)
    parser.add_argument('--allennlp_prediction_folder', help='folder with json file with allennlp predictions',
                        required=False,
                        default=None)
    parser.add_argument('--allennlp_prediction_file', help='file with json file with allennlp predictions',
                        required=False,
                        default=None)
    parser.add_argument('--logits_field', help='name of the field with the predicted logits for an example. Use '
                                               '"prob" for kgat models and "label_logits" for all others',
                        required=False,
                        default="label_logits")
    parser.add_argument('--only_convert', help='for generating the test submission: only convert data to the scorer '
                                               'format but do not actually evaluate',
                        required=False,
                        default=False, action='store_true', dest='only_convert')
    parser.add_argument('--output_file', help='file where to store the predictions in the format employed by the '
                                              'official fever scorer',
                        required=False,
                        default=None)
    return parser

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    parser = get_arg_parser()
    args = parser.parse_args()

    #read the gold data
    of_dev_fever = None
    logging.info(f"Reading gold standard from {args.gold_standard_fever}")
    if args.gold_standard_fever:
        with codecs.open(args.gold_standard_fever, "rb", "utf-8") as f:
            of_dev_fever = [json.loads(x) for x in f]

    #read the json file with the predictions
    logging.info(f"Reading prediction from {args.allennlp_prediction_file}")
    df_predictions = pd.read_json(os.path.join(args.allennlp_prediction_folder, args.allennlp_prediction_file),
                                  lines=True)

    strict_score, label_accuracy, precision, recall, f1 = evaluate_claim_level_pred(of_dev_fever,
                                                                                    df_predictions,
                                                                                    output_file=args.output_file,
                                                                                    predictions_col=args.logits_field,
                                                                                    qid_to_predicted_evidence_dict=None,
                                                                                    only_convert=args.only_convert)
    logging.info(f"FEVER score = {(strict_score*100):.2f}")
    logging.info(f"Label accuracy = {(label_accuracy*100):.2f}")
    logging.info(f"Evidence precision = {(precision * 100):.2f}")
    logging.info(f"Evidence recall = {(recall * 100):.2f}")
    logging.info(f"Evidence F1 = {(f1 * 100):.2f}")
