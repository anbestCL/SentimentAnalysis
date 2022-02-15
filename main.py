import argparse
import random

from SentimentClassifier import SentimentClassifier
from DataLoader import SEED, IMDB_DIR, TWITTER_FILE, DataLoader, \
                       DATA_DIR, TWITTER_DIR, AIRLINE_FILE
from ResultsProcesser import ResultsProcesser

BATCH_SIZE = 25

random.seed(SEED)
parser = argparse.ArgumentParser(description='Classify sentiment')
parser.add_argument('--data_name', type=str,
                    help='name of data: airline, twitter or imdb')
parser.add_argument('--sample_size', type=int, default=10,
                    help='size of sample, relevant for twitter and imdb')

args = parser.parse_args()

if args.data_name == "airline":
    airline_path = f"{DATA_DIR}{TWITTER_DIR}{AIRLINE_FILE}"
    data = DataLoader("airline", airline_path)

elif args.data_name == "twitter":
    twitter_path = f"{DATA_DIR}{TWITTER_DIR}{TWITTER_FILE}"
    data = DataLoader("twitter", twitter_path, sample_size=args.sample_size)

elif args.data_name == "imdb":
    imdb_path = f"{DATA_DIR}{IMDB_DIR}"
    data = DataLoader("imdb", imdb_path, sample_size=args.sample_size)

classifier = SentimentClassifier(data.data, args.data_name)
classifier.classify()

results = ResultsProcesser.fromdict(classifier.result_dict, args.data_name)
results.metrics()
