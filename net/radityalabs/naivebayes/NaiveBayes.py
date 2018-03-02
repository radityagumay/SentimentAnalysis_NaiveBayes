import sys, unicodedata
import os
import csv
import random
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords


class NaiveBayes(object):
    def __init__(self):
        self.path = os.path.expanduser(
            "~/PycharmProjects/SentimentAnalysis_NaiveBayes/net/radityalabs/data/positive-negative-data/")
        self.frequency_table = {}
        self.number_of_positive = 15
        self.number_of_negative = 13

        self.negative_prior = 0
        self.positive_prior = 0

        self.preprocessing = Preprocessing()

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
            tokens = word_tokenize(pos[0])
            tokens = self.preprocessing.tokenize(tokens)
            for token in tokens:
                if token in self.frequency_table:
                    self.frequency_table[token] += 1
                else:
                    self.frequency_table[token] = {1, "-"}
        for neg in doc_negative:
            tokens = word_tokenize(neg[0])
            tokens = self.preprocessing.tokenize(tokens)
            for token in tokens:
                if token in self.frequency_table:
                    neg_dic = self.frequency_table.keys()

                    self.frequency_table[token] += 1
                else:
                    self.frequency_table[token] = {"-", 1}

    # 3.
    def compute_the_prior(self):
        total = self.number_of_positive + self.number_of_negative
        self.positive_prior = self.number_of_positive / total
        self.negative_prior = self.number_of_negative / total

    # 4.
    '''
    P(C) = or we could say,
         = prior probabiliy is probability before run test
         
    Posterior Probability = is P(C) * test evidence
    
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
        
        p(amazing|positive) = [frequency_table] + 1 / total count of word in class 'positive' + tokens
    '''

    def compute_the_conditional_probability_or_likelihood(self):
        p = self.frequency_table.keys()


class Preprocessing(object):
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.tablePunctuation = dict.fromkeys(
            i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

    def tokenize(self, tokens):
        n_tokens = []
        for token in tokens:
            if len(token) > 3:
                # we dont need stemmer for a moment
                # stem = self.stemmer.stem(token)
                punct = token.translate(self.tablePunctuation)
                if punct is not None:
                    stop = punct not in set(stopwords.words('english'))
                    if stop:
                        n_tokens.append(punct)
        return n_tokens


naive = NaiveBayes()
naive.load_positive_document()
