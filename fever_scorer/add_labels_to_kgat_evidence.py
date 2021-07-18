## add labels to kgat evidence files in order to run experiments in the local setting
from itertools import chain
import codecs
import json
import argparse
from collections import defaultdict
import logging
import pandas as pd

def read_cid2evidence_label(standard_file):
    with codecs.open(standard_file, "rb", "utf-8") as f:
        of_fever = [json.loads(x) for x in f]

    cid2evidence = {
        x['id']: set(chain(*[[(e_i[2], e_i[3]) for e_i in e if e_i[2] is not None] for e in x['evidence']])) for x in
        of_fever
    }

    cid2label = {x['id']: x['label'] for x in of_fever}
    return cid2evidence, cid2label

def add_labels_to_evidence(cid, e, cid2evidence, label, nei_label="NOT ENOUGH INFO"):
    enriched_e = []
    for page, sent_id, sent_text, wgt in e:
        non_nei = (page, sent_id) in cid2evidence[cid]
        if non_nei:
            enriched_e.append([page, sent_id, sent_text, wgt, label])
        else:
            enriched_e.append([page, sent_id, sent_text, wgt, nei_label])
    return enriched_e

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Generates a fever evaluation table on a number of outputs')

    parser.add_argument('--input_file', help='input kgat data file', required=True,)
    parser.add_argument('--output_file', help='output kgat data file', required=True)
    parser.add_argument('--gold_standard_reference', help='Official FEVER file with gold standard reference if '
                                                          'available. Otherwise all evidence labels with be set to '
                                                          'NOT ENOUGH INFO',
                        required=False, default=None)
    return parser

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    parser = get_arg_parser()
    args = parser.parse_args()

    infile = args.input_file
    outfile = args.output_file

    logging.info(f"Reading from {infile}")

    if args.gold_standard_reference is not None:
        logging.info(f"Reading gold standard from {args.gold_standard_reference}")
        cid2evidence, cid2label = read_cid2evidence_label(args.gold_standard_reference)
    else:
        cid2evidence = defaultdict(set)

    df = pd.read_json(infile, lines=True)
    if not "label" in df.columns:
        df["label"] = 'NOT ENOUGH INFO'
    df["evidence"] = [add_labels_to_evidence(cid, evidence, cid2evidence, label) for cid, evidence, label in
                          df[["id", "evidence", "label"]].values]
    logging.info(f"Writing to {outfile}")
    df.to_json(outfile, orient="records", lines=True)
