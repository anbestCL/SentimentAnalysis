import os
import random
import pickle

import boto3

from comprehend_request import send_batch_request

DIR_NAME = "orig_data/aclImdb/train"
RESULTS_PATH_NAME = "processed_data/aclImdb/"
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

def save_results(dataset, results, path):
    complete_dict = {}
    for index, ((text,label), result_dict) in enumerate(zip(dataset, results)):
        complete_dict[index] = [text, label, result_dict]

    path_name = os.path.join(path, f"orig_w_results_{len(complete_dict.keys())}",)
    pickle.dump(complete_dict, open(path_name, "wb"))
        
def classify():
    comprehend = boto3.client(service_name='comprehend')

    examples_labeled = create_dataset(DIR_NAME, EX_COUNT)
    
    examples, _ = list(zip(*examples_labeled))
    classification_results = send_batch_request(comprehend, examples)

    save_results(examples_labeled, classification_results, RESULTS_PATH_NAME)

if __name__ == "__main__":
    classify()
