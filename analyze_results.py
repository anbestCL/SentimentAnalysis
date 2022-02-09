import pickle
from sklearn.metrics import precision_recall_fscore_support

def convert_label2index(label):
    return 1 if label == "pos" else 0

def get_labels(complete_dict):
    true_labels = []
    pred_labels = []
    for _, (_, true_label, result) in complete_dict.items():
        if result["Sentiment"] == "NEUTRAL":
            continue

        elif result["Sentiment"] == "POSITIVE":
            true_labels.append(convert_label2index(true_label))
            pred_labels.append(convert_label2index("pos"))

        elif result["Sentiment"] == "NEGATIVE":
            true_labels.append(convert_label2index(true_label))
            pred_labels.append(convert_label2index("neg"))
    return (true_labels, pred_labels)

def compute_metrics(labels):
    true_labels, pred_labels = labels
    score = precision_recall_fscore_support(true_labels, pred_labels, average='macro')         
    print(f"Precision:{score[0]} \nRecall:{score[1]} \nFscore: {score[2]}")

if __name__ == "__main__":
    complete_dict = pickle.load(open("orig_w_results.p", "rb"))

    labels = get_labels(complete_dict)
    compute_metrics(labels)
