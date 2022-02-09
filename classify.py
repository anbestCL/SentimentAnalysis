import os
import random

import boto3

from comprehend_request import send_batch_request

DIR_NAME = "aclImdb/train"
EX_COUNT = 10

def create_example_list(dir_name, polar, num):
    ex_list = []
    subdir_name = os.path.join(dir_name, polar)
    top_num = os.listdir(subdir_name)[:num]
    for file in top_num:
        ex_file = open(os.path.join(subdir_name, file), "r")
        ex_list.append((ex_file.read(), polar))
        ex_file.close()
    return ex_list

def create_dataset(dir_name, ex_count):
    examples_labeled = []
    for polar in ("pos", "neg"):
        ex_list = create_example_list(dir_name, polar, ex_count)
        examples_labeled.extend(ex_list)
    random.shuffle(examples_labeled)
    return examples_labeled


if __name__ == "__main__":
    comprehend = boto3.client(service_name='comprehend')

    examples_labeled = create_dataset(DIR_NAME, EX_COUNT)
    
    examples, labels = list(zip(*examples_labeled))
    classification_results = send_batch_request(comprehend, examples)


