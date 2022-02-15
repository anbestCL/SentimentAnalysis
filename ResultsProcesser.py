from typing import Dict, List, Tuple
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
                             confusion_matrix,
                             precision_recall_fscore_support,
                             accuracy_score,
                             ConfusionMatrixDisplay)


class ResultsProcesser:
    def __init__(self, data, data_name: str) -> None:
        self._data = data
        self._data_name = data_name

    @classmethod
    def fromfilename(cls, filename: str, data_name: str):
        data = pickle.load(open(filename, "rb"))
        return cls(data, data_name)

    @classmethod
    def fromdict(cls, datadict: Dict[int, Tuple[str, int or str, Dict]],
                 data_name: str):
        return cls(datadict, data_name)

    def metrics(self) -> None:
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

    def _rearrange(self,
                   data_dict: Dict[int, Tuple[str, str or int, Dict]]
                   ) -> List[List]:
        return list(zip(*data_dict.values()))

    def _filter(self, zipped_list: List[List]) -> Tuple[List]:
        filtered_texts, filtered_labels, filtered_sents = [], [], []
        for (text, true_label, sent) in zipped_list:
            if sent == "Mixed":
                continue
            if sent == "Neutral":
                if self._data_name == "imdb":
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

    def _get_numeric_labels(self,
                            data_dict: Dict[int, Tuple[str, str or int, Dict]],
                            label_type: str
                            ) -> Tuple[List]:
        text, true_labels, sentiments = self._rearrange(data_dict)

        sents, _ = self._extract_sentiments(sentiments)
        _, true_labels, sents = self._filter(zip(text, true_labels, sents))

        if label_type == "index":
            true_labels = list(map(self._index2index, true_labels))
        elif label_type == "shortlabel":
            true_labels = list(map(self._shortlabel2index, true_labels))
        elif label_type == "longlabel":
            true_labels = list(map(self._longlabel2index, true_labels))

        sents = list(map(self._longlabel2index, sents))

        return (true_labels, sents)

    def _compute_metrics(self) -> Dict[str, float]:
        if self._data[0][1].isdigit():
            true_labels, pred_labels = self._get_numeric_labels(self._data,
                                                                "index")
        elif self._data[0][1] in ("pos", "neg"):
            true_labels, pred_labels = self._get_numeric_labels(self._data,
                                                                "shortlabel")
        else:
            true_labels, pred_labels = self._get_numeric_labels(self._data,
                                                                "longlabel")

        accuracy = accuracy_score(true_labels, pred_labels)

        precision, recall, fscore, _ = precision_recall_fscore_support(
                                        true_labels, pred_labels, average=None)
        conf_matrix = confusion_matrix(true_labels, pred_labels)

        return {"Accuracy": accuracy, "Precision": precision,
                "Recall": recall, "Fscore": fscore,
                "ConfusionMatrix": conf_matrix}

    def _show_metrics(self, metrics: Dict[str, float]) -> None:
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

    def save2csv(self) -> None:
        df = self._transform2df(self._data)
        df.to_csv(f"results/{self._data_name}_results.csv")

    def _extract_sentiments(self, sentiments: Dict[str, str or Dict]
                            ) -> Tuple[List]:
        sents = []
        sent_scores = []
        for entry in sentiments:
            sent = entry["Sentiment"].title()
            sents.append(sent)
            sent_scores.append(entry["SentimentScore"][sent])
        return (sents, sent_scores)

    def _index2label(self, index: int) -> str:
        if index == 0:
            return "Negative"
        elif index == 2:
            return "Neutral"
        elif index == 4:
            return "Positive"

    def _label2long(self, label: str) -> str:
        return "Positive" if label == "pos" else "Negative"

    def _transform2df(self, data_dict: Dict[int, Tuple[str, int or str, Dict]]
                      ) -> pd.DataFrame:
        content = self._rearrange(data_dict)
        texts, true_labels, sentiments = content

        if self._data_name == "twitter":
            true_labels = list(map(self._index2label, true_labels))
        elif self._data_name == "imdb":
            true_labels = list(map(self._label2long, true_labels))

        sents, sent_scores = self._extract_sentiments(sentiments)

        df = pd.DataFrame(list(zip(texts, true_labels, sents, sent_scores)),
                          columns=["text", "true_label", "pred_label",
                                   "pred_confidence"])
        return df


if __name__ == "__main__":
    # path = "processed_data/twitter/orig_w_results_1000"
    # path2 = "processed_data/aclImdb/orig_w_results_992"
    # results = ResultsProcesser.fromfilename(path, "twitter")

    text = ('@switchfoot http://twitpic.com/2y1zl'
            '- Awww, that\'s a bummer.  You shoulda got'
            'David Carr of Third Day to do it. ;D')
    sentiment = {
        'Sentiment': 'NEGATIVE',
        'SentimentScore': {
            'Positive': 0.10567719489336014,
            'Negative': 0.6624420881271362,
            'Neutral': 0.22724494338035583,
            'Mixed': 0.004635817836970091
        }
    }
    test_dict = {0: [text, "Positive", sentiment],
                 1: [text, "Negative", sentiment]}
    results = ResultsProcesser.fromdict(test_dict, "twitter")
    results.metrics()
