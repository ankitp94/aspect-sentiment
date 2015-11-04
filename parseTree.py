import nltk
import nltk.parse.stanford
import csv
import os
import pdb
from unidecode import unidecode
import json
import random

os.environ['STANFORD_PARSER'] = '/home/asus/Downloads/parser/stanford-parser-full-2015-04-20/'
os.environ['STANFORD_MODELS'] = '/home/asus/Downloads/parser/stanford-parser-full-2015-04-20/'
model_path =  '/home/asus/Downloads/parser/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'

#os.environ['STANFORD_PARSER'] = '../stanford-parser-full-2015-04-20'
#os.environ['STANFORD_MODELS'] = '../stanford-parser-full-2015-04-20'


def parse_sent(parser, sent_list):
    sentences = parser.parse_sents(sent_list)
    return sentences


def load(filename):
    input_file = open(filename, 'r')
    csv_reader = csv.reader(input_file)
    data_set = {}
    for line in csv_reader:
        try:
            review = unidecode(line[0].decode('utf-8')).lower()
            if len(review) > 150:
                raise Exception
            if not all(ord(c) < 128 for c in review): pdb.set_trace()
            data_set[review] = [{'label': int(line[1])}]
        except:
            print "error ==> " , line
            pass
    input_file.close()
    print "DOne loading"
    return data_set


def tokenize(data_set):
    index = 0
    tok_review_set = []
    review_set = []
    for review in data_set:
        data_set[review].append(index)
        index += 1
        tok_review_set.append(nltk.word_tokenize(review))
        review_set.append(review)
    return tok_review_set, review_set


def binarize(tree):
    nltk.tree.Tree.collapse_unary(tree, True, True)
    nltk.tree.Tree.chomsky_normal_form(tree)
    return tree


def dump_to_file(filename, dataset):
    with open(filename, 'w') as handle:
        for data in dataset:
            handle.write(json.dumps(data) + '\n')
    return

if __name__ == '__main__':
    parser = nltk.parse.stanford.StanfordParser(model_path=model_path)
    data_set = load('trial.csv')
    token_set, review_set = tokenize(data_set)
    max_number = len(data_set)

    print "parsing" , max_number
    forest = parse_sent(parser, token_set[:max_number])
    print "parsing done"
    sents = []
    for tree in forest:
        sents.append(list(tree)[0])
    sents = [binarize(sent) for sent in sents]
    data = []
    for parsed_tree, review in zip(sents, review_set[:max_number]):
        data.append({"tree": str(parsed_tree), "label": data_set[review][:-1]})
    train_sample = int(len(data_set) * 0.7)
    dev_sample = int(len(data_set) * 0.1)
    test_sample = len(data_set) -1 - train_sample - dev_sample
    data_idx = list(range(len(data_set) -1 ))
    random.seed(15)
    random.shuffle(data_idx)
    train_idx = data_idx[:train_sample]
    dev_idx = data_idx[train_sample:train_sample+dev_sample]
    test_idx = data_idx[train_sample+dev_sample:]
    train_data = [data[i] for i in train_idx]
    dev_data = [data[i] for i in dev_idx]
    test_data = [data[i] for i in test_idx]
    dump_to_file('./data/train.json', train_data)
    dump_to_file('./data/dev.json', dev_data)
    dump_to_file('./data/test.json', test_data)
