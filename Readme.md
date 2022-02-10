# Sentiment Analysis

## Goal

Show case for using [Amazon Comprehend API](https://docs.aws.amazon.com/comprehend/latest/dg/what-is.html) to identify the sentiment of a document. Case involves sending sample datasets of alternatively [IMDB movie reviews](https://ai.stanford.edu/~amaas/data/sentiment/) or [Twitter statements](https://www.kaggle.com/kazanova/sentiment140) to API, receiving sentiment classification and analyzing sentiment classifications according to accuracy, precision, recall and f-score.

Using [```DetectSentiment```](https://docs.aws.amazon.com/comprehend/latest/dg/API_DetectSentiment.html) and [```BatchDetectSentiment```](https://docs.aws.amazon.com/comprehend/latest/dg/API_BatchDetectSentiment.html) from AWS.

## How to Use

Need to specify the data type (Imdb: ```imdb``` or Twitter: ```twitter```) and size of sample to get classification via AWS Comprehend. 
```classify.py --data_type imdb --sample_size 1000```

Visualize classification quality for respective data type:
```analyze_results.py --data_type imdb```

## Notes

Amazon Comprehend classifies as ```POSITIIVE```, ```NEGATIVE```, ```NEUTRAL``` or ```MIXED```. 

IMDB movie reviews consider only binary classification into positive and negative. 

Twitter dataset considers positive, negative and neutral, however, it is classified according to the use of emoticons in the tweets. No longer up-to-date.

### Comparison: Comprehend vs. custom

| AWS  | Custom  |
|---|---|
| + easy to set up  | - longer set up, model training required  |
| + immediate classification result  |  + fast results once model trained |
| + trained on ??? |     |
| - no customisation possible | + absolute customisation possible |
| - pricey | - cheap because private GPU can be used|

