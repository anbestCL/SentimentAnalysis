from urllib.error import HTTPError
import boto3

def send_single_request(service, text):
    """Sends a request to the AWS Comprehend service to detect sentiment of a single text

    Args:
        service (boto3 client): AWS client, here AWS Comprehend
        text (String): Text used for sentiment detection

    Raises:
        HTTPError: if request unsuccesful

    Returns:
        Dict: containing sentiment and sentimentscore for input text
    """
    response = service.detect_sentiment(Text=text, LanguageCode='en')
    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        return {key:response[key] for key in ["Sentiment", "SentimentScore"] }
    else:
        raise HTTPError(response)

def send_batch_request(service, text_list):
    """Sends a request to AWS Comprehend API for a text batch (max. 25 documents)

    Args:
        service (boto3 client): AWS client, here AWS Comprehend
        text_list (List[String]): list of candidate texts

    Raises:
        HTTPError: if request unsuccessful

    Returns:
        List[Dict]: list of dicts containing sentiment and sentiment score for each text
    """
    response = service.batch_detect_sentiment(TextList=text_list, LanguageCode='en')
    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        return [{key:result[key] for key in ["Sentiment", "SentimentScore"]} for result in response["ResultList"]]
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


