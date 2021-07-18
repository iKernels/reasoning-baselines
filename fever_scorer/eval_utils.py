import logging
import numpy as np
from fever_scorer.scorer import fever_score
import codecs
import json
import re

CLASS_ID_TO_LABEL_MAP = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
LABEL_TO_CLASS_ID_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}


def evaluate_claim_level_pred(fever_data,
                              predictions_dataframe,
                              predictions_col="label_logits",
                              qid_to_predicted_evidence_dict=None,
                              output_file=None,
                              only_convert=False):

    qid_to_predicted_labels_dict = {qid : CLASS_ID_TO_LABEL_MAP[np.argmax(predicted_logits)]
                                    for qid, predicted_logits in predictions_dataframe[["qid", predictions_col]].values}

    if qid_to_predicted_evidence_dict is None:
        qid_to_predicted_evidence_dict = dict([(qid, [split_into_page_and_sent_id(aid_i) for aid_i in aid]) for
                                               qid, aid in predictions_dataframe[["qid", "aid"]].values])

    predicted_fever_data = generate_predicted_fever_data(fever_data,
                                  qid_to_predicted_evidence_dict,
                                  qid_to_predicted_labels_dict,
                                  verbose=True)
    if output_file is not None:
        logging.info(f"Writing the fever-format data into {output_file}")
        with codecs.open(output_file, "wb", "utf-8") as f:
            for prediction in predicted_fever_data:
                f.write(json.dumps({
                    'id': prediction['id'],
                    'predicted_label': prediction['predicted_label'],
                    'predicted_evidence': prediction['predicted_evidence']
                }))
                f.write("\n")
    if only_convert:
        strict_score, label_accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        strict_score, label_accuracy, precision, recall, f1 = fever_score(predicted_fever_data)

    return strict_score, label_accuracy, precision, recall, f1

def split_into_page_and_sent_id(x, separator="_"):
    """
    converts evidence id to the official fever scorer format
    :param x: evidence id in format pagename_linenum
    :return: evidence id in format [pagename, linenum'
    """
    p, sid = x.rsplit(separator, 1)
    return [p, int(sid.strip("\""))]


def generate_predicted_fever_data(fever_data, qid_to_retrieved_sent_sel, qid_to_label, verbose = False):
    '''
    Generate the input for the official FEVER evaluator

    :param fever_data: array of fever data
    :param qid_to_retrieved_sent_sel: dict where qid is mapped to retrieved sentences
    :param qid_to_label: dict where qid is mapped to the final labels
    :return:
    '''
    pred_fever_data = []
    not_found = set()
    for df in fever_data:
        f = df.copy()
        f['predicted_evidence'] = []
        if df['id'] in qid_to_retrieved_sent_sel:
            predicted_ev = qid_to_retrieved_sent_sel[df['id']]
            f['predicted_evidence'] = [[x[0], x[1]] for x in predicted_ev]

        if df['id'] in qid_to_label:
            f['predicted_label'] = qid_to_label[df['id']]
        else:
            if verbose:
                logging.warning("No evidence found for '%d', predicting NOT ENOUGH INFO" % (df['id']))
            not_found.add(df['id'])
            f['predicted_label'] = "NOT ENOUGH INFO"

        pred_fever_data.append(f)
    if len(not_found) > 0 and verbose:
        logging.warning("No evidence found for '%d' claims" % (len(not_found)))
        logging.warning(not_found)
    return pred_fever_data