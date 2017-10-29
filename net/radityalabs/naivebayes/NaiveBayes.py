import os
import csv
import random

class NaiveBayes(object):
    def __init__(self):
        self.path = os.path.expanduser(
            "~/PycharmProjects/SentimentAnalysis_NaiveBayes/net/radityalabs/data/positive-negative-data/")

    def load_positive_document(self):
        document = []
        with open(self.path + "positive-data.csv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                document.append((row[0], "positive"))
        random.shuffle(document)
        return document[:10]

    def load_negative_document(self):
        document = []
        with open(self.path + "negative-data.csv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                document.append((row[0], "negative"))
        random.shuffle(document)
        return document[:10]

    # 1.
    def determine_training_and_testing(self):
        doc_positive = self.load_positive_document()
        print(doc_positive)

naive = NaiveBayes()
print(naive.load_positive_document())
