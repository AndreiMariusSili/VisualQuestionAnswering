from constants import *
import json
import gzip
import os
import numpy as np
import h5py
from collections import Counter
import codecs

class Preprocess:
    q_data_train = None
    q_data_test = None
    q_data_val = None
    a_data_train = None
    a_data_test = None
    a_data_val = None

    train_data = dict()
    test_data = dict()
    val_data = dict()

    image_to_visual_feat_mapping = None
    img_features = None
    image_dimension = None

    csv_delimiter = '~'

    reader = codecs.getreader("utf-8")

    def __init__(self):
        with open(data_folder + json_file, 'r') as f:
            self.image_to_visual_feat_mapping = json.load(f)['VQA_imgid2id']

        with gzip.GzipFile(data_folder + q_train_data_file, 'r') as file:
            self.q_data_train = json.load(self.reader(file))

        with gzip.GzipFile(data_folder + a_train_data_file, 'r') as file:
            self.a_data_train = json.load(self.reader(file))

        with gzip.GzipFile(data_folder + q_test_data_file, 'r') as file:
            self.q_data_test = json.load(self.reader(file))

        with gzip.GzipFile(data_folder + a_test_data_file, 'r') as file:
            self.a_data_test = json.load(self.reader(file))

        with gzip.GzipFile(data_folder + q_val_data_file, 'r') as file:
            self.q_data_val = json.load(self.reader(file))

        with gzip.GzipFile(data_folder + a_val_data_fie, 'r') as file:
            self.a_data_val = json.load(self.reader(file))

        self.img_features = np.asarray(h5py.File(data_folder + h5_file)['img_features'])
        self.calculateImageDimension()


    def getFeatures(self, answers):
        # for now get most common answer
        return self.getMostCommonAnswer(answers)


    def getMostCommonAnswer(self, answers):
        most_common_answers = dict()
        for answer_dict in answers:
            if not answer_dict['answer'] in most_common_answers:
                most_common_answers[answer_dict['answer']] = 1
            else:
                most_common_answers[answer_dict['answer']] += 1

        return str(Counter(most_common_answers).most_common()[0][0])

    def preprocessData(self, q_data, a_data, write_file):
        save_variable = dict()
        writeFile = open(write_file, 'w')

        for index, question_info in enumerate(q_data['questions']):
            image_id = str(question_info['image_id'])
            question = question_info['question']
            question = self.format_question(question)
            question_id = str(a_data['annotations'][index]['question_id'])

            save_variable[question_id] = dict()
            save_variable[question_id]['question'] = question
            save_variable[question_id]['annotations'] = a_data['annotations'][index]
            save_variable[question_id]['image_features'] = self.img_features[self.image_to_visual_feat_mapping[str(image_id)]]

            features = self.getFeatures(a_data['annotations'][index]['answers'])
            training_example = question_id + self.csv_delimiter + image_id + self.csv_delimiter + str(question) + self.csv_delimiter + features
            writeFile.write(training_example + '\n')

        writeFile.close()
        return save_variable

    def preprocess(self):
        #if os.path.isfile(data_folder + train_data_write_file) and os.path.isfile(data_folder + test_data_write_file) and os.path.isfile(data_folder + val_data_write_file):
        #    return

        self.train_data = self.preprocessData(self.q_data_train, self.a_data_train, data_folder + train_data_write_file)
        self.test_data = self.preprocessData(self.q_data_test, self.a_data_test, data_folder + test_data_write_file)
        self.val_data = self.preprocessData(self.q_data_val, self.a_data_val, data_folder + val_data_write_file)

    def calculateImageDimension(self):
        result = len(self.img_features[0])
        self.image_dimension = result
        return result


    def format_question(self, question):
        question_len = len(question)
        question_list = list(question)

        if (question_list[question_len-1] == '?' and question_list[question_len-2] != ' '):
            question_list[question_len-1] = ' '
        question_list.append('?')

        return ''.join(question_list)
