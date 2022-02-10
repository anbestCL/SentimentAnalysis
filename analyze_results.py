from typing import List, Tuple, Dict

import pickle
import argparse
import os

import matplotlib.pyplot as plt
from sklearn.metrics import (
                             confusion_matrix,
                             precision_recall_fscore_support,
                             accuracy_score,
                             ConfusionMatrixDisplay)


RESULTS_DIR = "processed_data/"
IMDB_DIR = "aclImdb/"
TWITTER_DIR = "twitter/"


def imdb_label2index(label: str) -> int:
    return 1 if label == "pos" else 0


def twitter_index_remap(index: int) -> int:
    return 1 if index == 4 else index


def pred_label2index(label: str) -> int:
    if label == "POSITIVE":
        return 1
    elif label == "NEGATIVE":
        return 0
    elif label == "NEUTRAL":
        return 2
    else:
        return None


def get_labels(complete_dict: Dict, datatype: str) -> Tuple[List[int]]:
    true_labels = []
    pred_labels = []
    for _, (_, true_label, result) in complete_dict.items():
        pred_label = pred_label2index(result["Sentiment"])

        if pred_label is None:
            continue

        if datatype == "imdb":
            # imdb has only pos and neg reviews
            if pred_label == 2:
                continue
            else:
                pred_labels.append(pred_label)
                true_labels.append(imdb_label2index(true_label))

        elif datatype == "twitter":
            pred_labels.append(pred_label)
            true_labels.append(twitter_index_remap(true_label))

    return (true_labels, pred_labels)


def compute_metrics(labels: Tuple[List[int]]) -> None:
    true_labels, pred_labels = labels
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Accuracy: {accuracy}")
    scores = precision_recall_fscore_support(true_labels,
                                             pred_labels,
                                             average=None)
    print(f"Precision:{scores[0]} \nRecall:{scores[1]} \nFscore: {scores[2]}")
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    if len(set(pred_labels)) == 2:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                      display_labels=["negative", "positive"])

    if len(set(pred_labels)) == 3:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                      display_labels=["negative",
                                                      "positive",
                                                      "neutral"])

    disp.plot()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        help="specifying data set, imdb or twitter",
    )
    args = parser.parse_args()

    if args.data_type == "imdb":
        results = [filename for filename in
                   os.listdir(f"{RESULTS_DIR}{IMDB_DIR}")
                   if filename.startswith("orig_w")]
        path = f"{RESULTS_DIR}{IMDB_DIR}{results[0]}"
    elif args.data_type == "twitter":
        results = [filename for filename in
                   os.listdir(f"{RESULTS_DIR}{TWITTER_DIR}")
                   if filename.startswith("orig_w")]
        path = f"{RESULTS_DIR}{TWITTER_DIR}{results[0]}"
    complete_dict = pickle.load(open(path, "rb"))

    labels = get_labels(complete_dict, args.data_type)
    print(f"Results for {args.data_type}")
    compute_metrics(labels)
