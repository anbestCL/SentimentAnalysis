import os
import random
import pickle
import pandas as pd
import argparse
from typing import Iterator, List, Tuple, Dict

import boto3

from comprehend_request import send_batch_request

random.seed(1000)

DATA_DIR = "orig_data/"
RESULTS_DIR = "processed_data/"

IMDB_DIR = "aclImdb/train/"
IMDB_DIR_RESULTS = "aclImdb/"

TWITTER_DIR = "twitter/"
TWITTER_FILE = "tweets.csv"
TWITTER_LEN = 1600000

SAMPLE_SIZE = 1000
BATCH_SIZE = 25
MAX_TEXT_LEN = 5000


def read_twitter_sample(file_path: str, sample_size: int) -> pd.DataFrame:
    sample = random.sample(range(1, TWITTER_LEN), TWITTER_LEN - sample_size)
    tweets = pd.read_csv(
        file_path,
        skiprows=sample,
        encoding="ISO-8859-1",
        header=None,
        names=["sentiment", "id", "date", "flag", "user", "tweet"],
    )

    return tweets


def create_twitter_dataset(file_path: str, sample_size: int) -> List[Tuple]:
    tweets = read_twitter_sample(file_path, sample_size)
    tweets.tweet.drop_duplicates()
    tweet_examples = list(tweets[["tweet", "sentiment"]].
                          to_records(index=False))
    tweet_examples_filtered = [
        (tweet, sentiment)
        for (tweet, sentiment) in tweet_examples
        if len(tweet) <= MAX_TEXT_LEN
    ]
    return tweet_examples_filtered


def read_imdb_sample(dir_name: str, polar: str, num: int) -> List[Tuple]:
    ex_list = []
    subdir_name = os.path.join(dir_name, polar)
    top_num = os.listdir(subdir_name)[:num]
    for file in top_num:
        ex_file = open(os.path.join(subdir_name, file), "r")
        ex_list.append((ex_file.read(), polar))
        ex_file.close()
    return ex_list


def create_imdb_dataset(dir_name: str, ex_count: int) -> List[Tuple]:
    examples_labeled = []
    for polar in ("pos", "neg"):
        ex_list = read_imdb_sample(dir_name, polar, int(ex_count / 2))
        ex_list_filtered = [
            (text, sentiment)
            for (text, sentiment) in ex_list
            if len(text) <= MAX_TEXT_LEN
        ]
        examples_labeled.extend(ex_list_filtered)
    random.shuffle(examples_labeled)
    return examples_labeled


def save_results(dataset: List[Tuple], results: List[Dict], path: str) -> None:
    complete_dict = {}
    for index, ((text, label), result_dict) in enumerate(
        zip(dataset, results)
    ):
        complete_dict[index] = [text, label, result_dict]

    path_name = os.path.join(
        path,
        f"orig_w_results_{len(complete_dict.keys())}",
    )
    pickle.dump(complete_dict, open(path_name, "wb"))


def batch_examples(examples_labeled: List[Tuple],
                   batch_size: int) -> Iterator[List[str]]:
    examples, _ = list(zip(*examples_labeled))
    for batch_start in range(0, len(examples), batch_size):
        batch_end = min(batch_start+batch_size, len(examples))
        yield examples[batch_start:batch_end]


def classify(path2dir: str, path2results: str, args: Dict) -> None:
    comprehend = boto3.client(service_name="comprehend")

    if args.data_type == "imdb":
        examples_labeled = create_imdb_dataset(path2dir, args.sample_size)

    elif args.data_type == "twitter":
        examples_labeled = create_twitter_dataset(path2dir, args.sample_size)

    classification_results = []
    for batch in batch_examples(examples_labeled, BATCH_SIZE):
        batch_results = send_batch_request(comprehend, batch)
        classification_results.extend(batch_results)

    save_results(examples_labeled, classification_results, path2results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        help="specifying data set, imdb or twitter",
    )
    parser.add_argument(
        "--sample_size", type=int, required=True, help="size of desired sample"
    )
    args = parser.parse_args()

    if args.data_type == "imdb":
        path2dir = os.path.join(DATA_DIR, IMDB_DIR)
        path2results = os.path.join(RESULTS_DIR, IMDB_DIR_RESULTS)

    elif args.data_type == "twitter":
        path2dir = os.path.join(DATA_DIR, TWITTER_DIR, TWITTER_FILE)
        path2results = os.path.join(RESULTS_DIR, TWITTER_DIR)

    classify(path2dir, path2results, args)
