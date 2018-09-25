from constants import *
from preprocess import Preprocess
import os
import re
import csv
import pickle
from collections import defaultdict
import numpy as np

class Dictionary:
    word2idx = {}
    idx2word = []
    labels2idx = {}
    idx2labels = []

    pp_data = None # type: Preprocess
    max_labels = None

    oov = '<UNK>'
    pad = '<PAD>'

    def __init__(self, preprocessed_data, max_labels = 1000):
        self.pp_data = preprocessed_data # type: Preprocess
        self.max_labels = max_labels

        if self.dictionariesAreBuilt():
            self.loadDictionaries()
        else:
            self.generateDictionaries(data_folder + train_data_write_file, True)

    def dictionariesAreBuilt(self):
        return False
        return os.path.isfile(data_folder + idx2word_file) and os.path.isfile(data_folder + word2idx_file) \
               and os.path.isfile(data_folder + labels2idx_file) and os.path.isfile(data_folder + idx2labels_file)

    def generateDictionaries(self, processed_csv, save):
        self.idx2word.append(self.pad)
        self.word2idx[self.pad] = 0
        self.idx2word.append(self.oov)
        self.word2idx[self.oov] = 1

        with open(processed_csv, 'r') as csv_data:
            data = csv.reader(csv_data, delimiter=self.pp_data.csv_delimiter)
            answers = []

            for (_, _, question, answer) in data:
                words = question + ' ' + answer
                answers.append(answer)

                for word in re.split(r'[^\w]+', words):
                    lowercase_word = word.lower()

                    if lowercase_word not in self.word2idx:
                        index = len(self.idx2word)
                        self.idx2word.append(lowercase_word)
                        self.word2idx[lowercase_word] = index

        csv_data.close()

        labels = defaultdict(int)
        for answer in answers:
            labels[answer.lower()] += 1


        sorted_answers = sorted(labels, key=labels.get, reverse=True)

        if str(self.max_labels) == 'all':
            self.idx2labels = sorted_answers
        else:
            self.idx2labels = sorted_answers[0:self.max_labels - 1]

        self.idx2labels.append(self.oov) # append out of vocabulary word

        for i in range(len(self.idx2labels)):
            self.labels2idx[self.idx2labels[i]] = i

        if save:
            self.saveDictionaries()


    def saveDictionaries(self):
        fd = open(data_folder + idx2word_file, 'wb')
        pickle.dump(self.idx2word, fd)
        fd.close()

        fd = open(data_folder + word2idx_file, 'wb')
        pickle.dump(self.word2idx, fd)
        fd.close()

        fd = open(data_folder + labels2idx_file, 'wb')
        pickle.dump(self.labels2idx, fd)
        fd.close()

        fd = open(data_folder + idx2labels_file, 'wb')
        pickle.dump(self.idx2labels, fd)
        fd.close()

    def loadDictionaries(self):
        fd = open(data_folder + word2idx_file, 'rb')
        self.word2idx = pickle.load(fd)
        fd.close()

        fd = open(data_folder + idx2word_file, 'rb')
        self.idx2word = pickle.load(fd)
        fd.close()

        fd = open(data_folder + idx2labels_file, 'rb')
        self.idx2labels = pickle.load(fd)
        fd.close()

        fd = open(data_folder + labels2idx_file, 'rb')
        self.labels2idx = pickle.load(fd)
        fd.close()

    def getVocabSize(self):
        return len(self.idx2word)