# Sentiment Analysis

## Goal

Show case for using [Amazon Comprehend API](https://docs.aws.amazon.com/comprehend/latest/dg/what-is.html) to identify the sentiment of a document. Case involves sending sample datasets of alternatively [IMDB movie reviews](https://ai.stanford.edu/~amaas/data/sentiment/) or [Twitter statements](https://www.kaggle.com/kazanova/sentiment140) or [Airline tweets](https://data.world/crowdflower/airline-twitter-sentiment) to API, receiving sentiment classification and analyzing sentiment classifications according to accuracy, precision, recall and f-score.

Using [```DetectSentiment```](https://docs.aws.amazon.com/comprehend/latest/dg/API_DetectSentiment.html) and [```BatchDetectSentiment```](https://docs.aws.amazon.com/comprehend/latest/dg/API_BatchDetectSentiment.html) from AWS.

## How to Use

Need to specify the data name (Imdb: ```imdb``` or Twitter: ```twitter```) and size of sample to get classification via AWS Comprehend.
```main.py --data_name imdb --sample_size 1000```

By default, text is preprocessed to remove twitter handles, url links and html signifiers.

This script runs the classification and automatically displays the metrics calculated based on the results, i.e. accuracy, precision, recall, fscore and display of confusion matrix.

## Notes

Amazon Comprehend classifies as ```POSITIIVE```, ```NEGATIVE```, ```NEUTRAL``` or ```MIXED```.

IMDB movie reviews consider only binary classification into positive and negative.

Twitter dataset considers positive, negative and neutral, however, it is classified according to the use of emoticons in the tweets. No longer up-to-date.

Airline dataset considers positive, negative and neutral tweets. It contains an airline column, so tweets can be analyzed separately for each airline.

### Comparison: AWS vs. Custom

| AWS  | Custom  |
|---|---|
| + easy to set up  | - longer set up, model training required  |
| + immediate classification result  |  + fast results once model trained |
| - no model details available | - model self-defined |
| - no customisation possible | + absolute customisation possible |
| - pricey | - cheap because private GPU can be used|
