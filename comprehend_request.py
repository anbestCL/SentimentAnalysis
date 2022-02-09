from typing import List, Dict
from urllib.error import HTTPError
import boto3


def send_single_request(service: boto3.Session.client, text: str) -> Dict:
    response = service.detect_sentiment(Text=text, LanguageCode='en')
    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        return {key: response[key] for key in ["Sentiment", "SentimentScore"]}
    else:
        raise HTTPError(response)


def send_batch_request(service: boto3.Session.client, text_list: List[str]):
    response = service.batch_detect_sentiment(TextList=text_list,
                                              LanguageCode='en')
    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        return [{key: result[key] for key in ["Sentiment", "SentimentScore"]}
                for result in response["ResultList"]]
    else:
        raise HTTPError(response)


if __name__ == "__main__":
    comprehend = boto3.client(service_name='comprehend')
    text = "It is raining today in Seattle"

    # Single text
    response = send_single_request(comprehend, text)
    print(response["Sentiment"], "\n", response["SentimentScore"])

    # Text batch
    response = send_batch_request(comprehend, [text])
    print(response[0]["Sentiment"], "\n", response[0]["SentimentScore"])
