import pickle
import pandas as pd


class Results:
    def __init__(self, path: str):
        self.data = pickle.load(open(path, "rb"))

        if "twitter" in path:
            self.text_type = "tweets"
        elif "Imdb" in path:
            self.text_type = "reviews"

    def save2csv(self):
        df = self._transform2df(self.data)
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
    results = Results(path2)
    results.save2csv()
