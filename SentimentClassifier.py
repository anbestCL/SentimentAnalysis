from typing import Iterator, List, Tuple, Dict
import pickle

import boto3

from comprehend_request import send_batch_request, send_single_request

from DataLoader import DataLoader, DATA_DIR, TWITTER_DIR, AIRLINE_FILE


class SentimentClassifier:

    def __init__(self, data: List[Tuple], data_name: str):
        restructured = list(zip(*data))
        self._data = restructured[0]
        self._labels = restructured[2] if len(restructured) == 3 else 1
        self._data_name = data_name

    def _batch(self, examples: List[str],
               batch_size: int) -> Iterator[List[str]]:
        for batch_start in range(0, len(examples), batch_size):
            batch_end = min(batch_start+batch_size, len(examples))
            yield examples[batch_start:batch_end]

    def classify(self) -> None:
        comprehend = boto3.client(service_name="comprehend")

        classification_results = []
        for batch in self._batch(self._data, BATCH_SIZE):
            batch_results = send_batch_request(comprehend, batch)
            classification_results.extend(batch_results)

        print("Classification complete")
        self.results = classification_results

        self._save()

    @staticmethod
    def classify_single_text(text: str) -> str:
        comprehend = boto3.client(service_name="comprehend")

        result = send_single_request(comprehend, text)
        print(result["Sentiment"])

    def _create_result_dict(self) -> Dict[int, List]:
        complete_dict = {}
        for index, (text, label, result_dict) in enumerate(
            zip(self._data, self._labels, self.results)
        ):
            complete_dict[index] = [text, label, result_dict]
        return complete_dict

    def _save(self) -> None:
        self.result_dict = self._create_result_dict()
        results_file_name = f"processed_data/comprehend_results_\
                            {self._data_name}.p"
        pickle.dump(self.result_dict, open(results_file_name, "wb"))


if __name__ == "__main__":
    BATCH_SIZE = 25

    airline_path = f"{DATA_DIR}{TWITTER_DIR}{AIRLINE_FILE}"
    airline_data = DataLoader("airline", airline_path)

    classifier = SentimentClassifier(airline_data.data, "airline")
    example = "I hate this so much, but I cannot stop."
    SentimentClassifier.classify_single_text(example)
