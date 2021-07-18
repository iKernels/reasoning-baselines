import argparse
import logging
import pandas as pd
from fever_scorer.fever_one_system_eval import evaluate_claim_level_pred
import os
import re
import numpy as np
from itertools import chain

def get_most_confident_non_nei(x, non_nei_threshold=0):
    x = np.vstack(x)
    labels = x.argmax(axis=1)
    vals = np.take_along_axis(x, labels[:, None], axis=-1).flatten()
    non_nei_scores = [(label, val) for label, val in zip(labels, vals) if label < 2 and val > non_nei_threshold]
    predictions = [0, 0, 0]
    pred_label = 2 if len(non_nei_scores) == 0 else non_nei_scores[0][0]
    predictions[pred_label] = 1
    return predictions


def get_top_ranking_non_nei(x):
    '''
    here, we assume that the predictions are alread sorted according to the scores assigned to them during retrieval
    '''
    return list(x)[0]

def aggregate_local(input_file, output_file, aggregator=get_most_confident_non_nei, logits_field="label_logits"):
    logging.info(f"Reading from {input_file}")
    df_local = pd.read_json(input_file, lines=True)
    df_agg = df_local[["qid", "aid", logits_field]].groupby("qid").aggregate({
        logits_field: aggregator,
        'aid': lambda x: list(chain(*x))
    }).reset_index()
    logging.info(f"Writing to {output_file}")
    df_agg.to_json(output_file, lines=True, orient="records")

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

    parser.add_argument('--input_local_file', help='Input file with local predictions', required=False)
    parser.add_argument('--output_global_file', help='Output file with local predictions merged to global',
                        required=False)
    parser.add_argument('--merge_heuristic', help='Do we use merge heuristic 1 or 2 from the paper?',
                        type=int, default=1,
                        required=False)


    parser.add_argument('--logits_field', help='name of the field with the predicted logits for an example. Use '
                                               '"prob" for kgat models and "label_logits" for all others',
                        required=False,
                        default="label_logits")
    return parser


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    parser = get_arg_parser()
    args = parser.parse_args()
    aggregator = get_top_ranking_non_nei if not args.merge_heuristic == 1 else get_most_confident_non_nei
    aggregate_local(args.input_local_file,
                    args.output_global_file,
                    aggregator=aggregator)