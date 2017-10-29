import os
import csv
import random
import nltk


class NaiveBayes(object):
    def __init__(self):
        self.path = os.path.expanduser(
            "~/PycharmProjects/SentimentAnalysis_NaiveBayes/net/radityalabs/data/positive-negative-data/")
        self.frequency_table = {}
        self.number_of_positive = 5
        self.number_of_negative = 3

        self.negative_prior = 0
        self.positive_prior = 0

    def load_positive_document(self):
        document = []
        with open(self.path + "positive-data.csv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                document.append((row[0], "positive"))
        random.shuffle(document)
        return document[:self.number_of_positive]

    def load_negative_document(self):
        document = []
        with open(self.path + "negative-data.csv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                document.append((row[0], "negative"))
        random.shuffle(document)
        return document[:self.number_of_negative]

    def run(self):
        self.determine_training_and_testing()

    # 1.
    def determine_training_and_testing(self):
        doc_positive = self.load_positive_document()
        doc_negative = self.load_negative_document()
        self.create_frequency_table(doc_positive, doc_negative)

    # 2.
    def create_frequency_table(self, doc_positive, doc_negative):
        for pos in doc_positive:
            tokens = nltk.word_tokenize(pos[0])
            for token in tokens:
                if token in self.frequency_table:
                    self.frequency_table[token] += 1
                else:
                    self.frequency_table[token] = 1
        for neg in doc_negative:
            tokens = nltk.word_tokenize(neg[0])
            for token in tokens:
                if token in self.frequency_table:
                    self.frequency_table[token] += 1
                else:
                    self.frequency_table[token] = 1

    # 3.
    def compute_the_prior(self):
        total = self.number_of_positive + self.number_of_negative
        self.positive_prior = self.number_of_positive / total
        self.negative_prior = self.number_of_negative / total

    # 4.
    '''
    equation for compute probability likelihood.
    p(w|c) = count(w,c) + 1 / count(c) + |V|
    
    where :
        p(w|c)  = conditional probability / likelihood
            w   = word attribute
            c   = class (negative or positive)
            
        count(w,c)     = total count of word attribute occurs in class C. 
                         (Look Step 2 - Frequency Table). 
                            
        + 1            = Laplacian Smoothing @see #https://en.wikipedia.org/wiki/Laplacian_smoothing
        
        count(c)       = total count of word attribute in particular class occurs in training set.
        
        |V|            = vocabulary, total count of DIFFERENT word attribute in training set.
    '''
    def compute_the_conditional_probability_or_likelihood(self):
        print("hello")


naive = NaiveBayes()
naive.run()
