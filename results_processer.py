from typing import Dict
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
                             confusion_matrix,
                             precision_recall_fscore_support,
                             accuracy_score,
                             ConfusionMatrixDisplay)


class Results:
    def __init__(self, path: str):
        self._data = pickle.load(open(path, "rb"))

        if "twitter" in path:
            self.text_type = "tweets"
        elif "Imdb" in path:
            self.text_type = "reviews"

    def metrics(self):
        metrics = self._compute_metrics()
        self._show_metrics(metrics)

    def _shortlabel2index(self, label: str) -> int:
        return 1 if label == "pos" else 0

    def _longlabel2index(self, label: str) -> int or None:
        if label == "Positive":
            return 1
        elif label == "Negative":
            return 0
        elif label == "Neutral":
            return 2
        else:
            return None

    def _index2index(self, index: int) -> int:
        return 1 if index == 4 else index

    def _rearrange(self, data_dict):
        return list(zip(*data_dict.values()))

    def _filter(self, zipped_list):
        filtered_texts, filtered_labels, filtered_sents = [], [], []
        for (text, true_label, sent) in zipped_list:
            if sent == "Mixed":
                continue
            if sent == "Neutral":
                if self.text_type == "reviews":
                    continue
                # TODO: SIMPLIFY
                else:
                    filtered_texts.append(text)
                    filtered_labels.append(true_label)
                    filtered_sents.append(sent)
            else:
                filtered_texts.append(text)
                filtered_labels.append(true_label)
                filtered_sents.append(sent)
        return (filtered_texts, filtered_labels, filtered_sents)

    def _get_numeric_labels(self, data_dict):
        text, true_labels, sentiments = self._rearrange(data_dict)

        sents, _ = self._extract_sentiments(sentiments)
        _, true_labels, sents = self._filter(zip(text, true_labels, sents))

        if self.text_type == "tweets":
            true_labels = list(map(self._index2index, true_labels))
        elif self.text_type == "reviews":
            true_labels = list(map(self._shortlabel2index, true_labels))

        sents = list(map(self._longlabel2index, sents))

        return (true_labels, sents)

    def _compute_metrics(self):
        true_labels, pred_labels = self._get_numeric_labels(self._data)
        accuracy = accuracy_score(true_labels, pred_labels)

        precision, recall, fscore, _ = precision_recall_fscore_support(
                                        true_labels, pred_labels, average=None)
        conf_matrix = confusion_matrix(true_labels, pred_labels)

        return {"Accuracy": accuracy, "Precision": precision,
                "Recall": recall, "Fscore": fscore,
                "ConfusionMatrix": conf_matrix}

    def _show_metrics(self, metrics: Dict):
        print(f'Accuracy: {metrics["Accuracy"]}')
        print(f'Precision:{metrics["Precision"]} \n'
              f'Recall:{metrics["Recall"]} \n'
              f'Fscore: {metrics["Fscore"]}')

        conf_matrix = metrics["ConfusionMatrix"]
        if len(conf_matrix) == 2:
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                          display_labels=["negative",
                                                          "positive"])

        if len(conf_matrix) == 3:
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                          display_labels=["negative",
                                                          "positive",
                                                          "neutral"])

        disp.plot()
        plt.show()

    def save2csv(self):
        df = self._transform2df(self._data)
        df.to_csv(f"results/{self.text_type}_results.csv")

    def _extract_sentiments(self, sentiments):
        sents = []
        sent_scores = []
        for entry in sentiments:
            sent = entry["Sentiment"].title()
            sents.append(sent)
            sent_scores.append(entry["SentimentScore"][sent])
        return (sents, sent_scores)

    def _index2label(self, index):
        if index == 0:
            return "Negative"
        elif index == 2:
            return "Neutral"
        elif index == 4:
            return "Positive"

    def _label2long(self, label):
        return "Positive" if label == "pos" else "Negative"

    def _transform2df(self, data_dict):
        content = list(zip(*data_dict.values()))
        texts, true_labels, sentiments = content

        if self.text_type == "tweets":
            true_labels = list(map(self._index2label, true_labels))
        elif self.text_type == "reviews":
            true_labels = list(map(self._label2long, true_labels))

        sents, sent_scores = self._extract_sentiments(sentiments)

        df = pd.DataFrame(list(zip(texts, true_labels, sents, sent_scores)),
                          columns=["text", "true_label", "pred_label",
                                   "pred_confidence"])
        return df


if __name__ == "__main__":
    path = "processed_data/twitter/orig_w_results_1000"
    path2 = "processed_data/aclImdb/orig_w_results_992"
    results = Results(path)
    results.metrics()
