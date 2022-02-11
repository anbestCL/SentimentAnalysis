from typing import List, Tuple

import random
import os

import pandas as pd

SEED = 1000

MAX_TEXT_LEN = 5000

DATA_DIR = "data/"

IMDB_DIR = "aclImdb/train"
TWITTER_DIR = "twitter/"
AIRLINE_FILE = "Airline-Sentiment-2-w-AA.csv"
TWITTER_FILE = "tweets.csv"

TWITTER_LEN = 1600000


class DataLoader:
    def __init__(self, data_name: str, file_path: str,
                 sample_size: int = 0, preprocess: bool = True):

        self._data_name = data_name
        self._sample_size = sample_size

        self._dataset = self._create_dataset(file_path, preprocess)
        self.data = self._df2list(self._dataset)

    def _create_dataset(self, file_path: str, preprocess: bool):
        if self._data_name == "airline":
            self._dataset = self._create_airline_dataset(file_path)

        if self._data_name == "twitter":
            self._dataset = self._create_twitter_dataset(file_path,
                                                         self._sample_size)
        if self._data_name == "imdb":
            self._dataset = self._create_imdb_dataset(file_path,
                                                      self._sample_size)
        if preprocess:
            return self._preprocess_text(self._dataset)
        else:
            return self._dataset

    def _read_airline_dataset(self, file_path: str) -> pd.DataFrame:
        tweets = pd.read_csv(
            file_path,
            encoding="ISO-8859-1",
            header=0,
        )
        return tweets

    def _create_airline_dataset(self, file_path: str) -> pd.DataFrame:
        tweets = self._read_airline_dataset(file_path)
        tweets.text = tweets.text.drop_duplicates()
        tweets.airline_sentiment = tweets.airline_sentiment.str.title()
        tweets = tweets.rename(columns={"airline_sentiment": "sentiment"})
        return tweets[["text", "airline", "sentiment"]]

    def _read_twitter_sample(self, file_path: str,
                             sample_size: int) -> pd.DataFrame:
        sample = random.sample(range(1, TWITTER_LEN),
                               TWITTER_LEN - sample_size)
        tweets = pd.read_csv(file_path, skiprows=sample, encoding="ISO-8859-1",
                             header=None,
                             names=["sentiment", "id", "date", "flag", "user",
                                    "text"])
        return tweets

    def _index2label(self, index: int) -> str:
        if index == 0:
            return "Negative"
        elif index == 4:
            return "Positive"
        else:
            return "Neutral"

    def _create_twitter_dataset(self, file_path: str,
                                sample_size: int) -> pd.DataFrame:
        tweets = self._read_twitter_sample(file_path, sample_size)
        tweets.text = tweets.text.drop_duplicates()
        tweets.sentiment = tweets.sentiment.map({0: "Negative",
                                                 2: "Neutral", 4: "Positive"})

        return tweets[["text", "sentiment"]]

    def _read_imdb_sample(self, dir_name: str, polar: str,
                          num: int) -> List[Tuple]:
        reviews = []
        subdir_name = os.path.join(dir_name, polar)
        sample = random.sample(os.listdir(subdir_name), num)
        for file_name in sample:
            with open(os.path.join(subdir_name, file_name), "r") as file:
                reviews.append(file.read())
        return reviews

    def _create_imdb_dataset(self, dir_name: str,
                             sample_size: int) -> pd.DataFrame:
        size = int(sample_size / 2)
        pos_reviews = self._read_imdb_sample(dir_name, "pos", size)
        neg_reviews = self._read_imdb_sample(dir_name, "neg", size)
        labels = ["Positive"]*len(pos_reviews)+["Negative"]*len(neg_reviews)
        reviews = pd.DataFrame({"text": pos_reviews+neg_reviews,
                                "sentiment": labels})
        reviews = reviews.sample(frac=1,
                                 random_state=SEED).reset_index(drop=True)
        return reviews

    def _df2list(self, df: pd.DataFrame) -> List[Tuple]:
        return df.to_records(index=False)

    def _preprocess_text(self, data_set: pd.DataFrame) -> pd.DataFrame:
        # filter: keep only <= MAX_TEXT_LEN
        filtered = data_set[data_set.text.str.len() < MAX_TEXT_LEN]

        # remove @ and url
        pattern = r"(http://[^\"\s]+)|(www.[^\"\s]+)|(\@\w+)|(<\w+\s?/?>)"
        filtered.text = filtered.text.str.replace(pattern, "", regex=True)
        return filtered

    def save2csv(self):
        self._dataset.to_csv(f"csv_data/{self._data_name}.csv")


if __name__ == "__main__":
    random.seed(SEED)

    airline_path = f"{DATA_DIR}{TWITTER_DIR}{AIRLINE_FILE}"
    airline_data = DataLoader("airline", airline_path)
    print(airline_data.data[0])
    airline_data.save2csv()

    # twitter_path = f"{DATA_DIR}{TWITTER_DIR}{TWITTER_FILE}"
    # twitter_data = DataLoader("twitter", twitter_path, sample_size=10)
    # print(twitter_data.data[0])

    # imdb_path = f"{DATA_DIR}{IMDB_DIR}"
    # imdb_data = DataLoader("imdb", imdb_path, sample_size=10)
    # print(imdb_data.data[0])
