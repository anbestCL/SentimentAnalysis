import boto3

comprehend = boto3.client(service_name='comprehend')
                
text = "It is raining today in Seattle"

response = comprehend.detect_sentiment(Text=text, LanguageCode='en')

print(response["Sentiment"], "\n", response["SentimentScore"])

