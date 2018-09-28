from collections import Counter, defaultdict
from typing import Iterable, Dict, Union
from misc.constants import *
import pickle
import gzip
import json
import h5py
import csv
import os
import re


def run(max_labels: int):
    """Process the raw VQA data. Create csv files for training, evaluation, and testing that contain:
        - question id
        - image id
        - question string
        - answer string
        - serialized visual features
    Also create word2id, id2word, label2id, and id2label mappings.

    The raw data is required to be available in the under the following folder structure:
        -- ROOT
        ---- data
        ------ raw
        -------- gzip
        ---------- available at https://github.com/timbmg/NLP1-2017-VQA/tree/master/data
        -------- h5
        ---------- available at https://aashishv.stackstorage.com/s/MvvB4IQNk9QlydI
        -------- json
        ---------- also available at https://aashishv.stackstorage.com/s/MvvB4IQNk9QlydI

        Args:
            max_labels: how many labels to tokenize. Rest is treated as OOV.
    """
    if not os.path.isdir(GZIP_FOLDER) or not os.path.isdir(JSON_FOLDER) or not os.path.isdir(H5_FOLDER):
        raise BlockingIOError('All or part of the raw data is not available. Check docstring for details.')

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    print('Raw data exists. Processed data will be persisted to {}.'.format(os.path.abspath(PROCESSED_FOLDER)))

    print('Loading raw data...', end='')
    with gzip.GzipFile(os.path.join(GZIP_FOLDER, Q_TRAIN_DATA_FILE), 'r') as file:
        q_data_train = json.load(file)

    with gzip.GzipFile(os.path.join(GZIP_FOLDER, A_TRAIN_DATA_FILE), 'r') as file:
        a_data_train = json.load(file)

    with gzip.GzipFile(os.path.join(GZIP_FOLDER, Q_TEST_DATA_FILE), 'r') as file:
        q_data_test = json.load(file)

    with gzip.GzipFile(os.path.join(GZIP_FOLDER, A_TEST_DATA_FILE), 'r') as file:
        a_data_test = json.load(file)

    with gzip.GzipFile(os.path.join(GZIP_FOLDER + Q_VAL_DATA_FILE), 'r') as file:
        q_data_val = json.load(file)

    with gzip.GzipFile(os.path.join(GZIP_FOLDER, A_VAL_DATA_FILE), 'r') as file:
        a_data_val = json.load(file)
    print('Done.')

    print('Processing raw training data...', end='')
    __process_data(q_data_train, a_data_train, PROCESSED_FOLDER + PROCESSED_TRAIN_FILE)
    print('Done.')

    print('Processing raw validation data...', end='')
    __process_data(q_data_val, a_data_val, PROCESSED_FOLDER + PROCESSED_VAL_FILE)
    print('Done.')

    print('Processing raw test data...', end='')
    __process_data(q_data_test, a_data_test, PROCESSED_FOLDER + PROCESSED_TEST_FILE)
    print('Done.')

    print('Creating word and label mappings...', end='')
    __generate_dictionaries(PROCESSED_FOLDER + PROCESSED_TRAIN_FILE, max_labels)
    print('Done.')


def __most_common_answer(answers: Iterable[str]) -> str:
    """Find the most common item in an iterable of answers to a question.

    Args:
        answers: iterable of answers.

    Returns:
        the most common answer
    """
    most_common_answers = dict()
    for answer_dict in answers:
        if not answer_dict['answer'] in most_common_answers:
            most_common_answers[answer_dict['answer']] = 1
        else:
            most_common_answers[answer_dict['answer']] += 1

    return str(Counter(most_common_answers).most_common()[0][0])


def __format_question(question: str) -> str:
    """ Normalise question according to https://github.com/timbmg/NLP1-2017-VQA/blob/master/VQA%20Dataset%20Structure.ipynb

    Args:
        question: a question string.

    Returns:
        The normalised question string.
    """
    question_len = len(question)
    question_list = list(question)

    if question_list[question_len - 1] == '?' and question_list[question_len - 2] != ' ':
        question_list[question_len - 1] = ' '
    question_list.append('?')

    return ''.join(question_list)


def __process_data(q_data: Dict, a_data: Dict, path: str) -> None:
    """Create csv file containig:
        - question id
        - image id
        - question string
        - answer string
        - serialized visual features
    Persist all csv data to 'path'.
    Args:
        q_data: a dictionary of question data.
        a_data: a dictionary of answer data.
        path: where to store the processed data file.
    """
    with open(JSON_FOLDER + IMG2ID_FILE, 'r') as imgid2id_json:
        img2id = json.load(imgid2id_json)['VQA_imgid2id']
    with h5py.File(H5_FOLDER + VL_FEAT_FILE) as vl_feat_h5:
        vl_feat = vl_feat_h5['img_features']

        with open(path, 'w+') as pre_processed_file:
            fields = ['question_id', 'image_id', 'question', 'answer', 'visual_features']
            writer = csv.DictWriter(pre_processed_file, fields)
            writer.writeheader()

            for index, question_info in enumerate(q_data['questions']):
                question_id = str(a_data['annotations'][index]['question_id'])
                image_id = str(question_info['image_id'])
                question = question_info['question']
                question = __format_question(question)
                visual_feature = pickle.dumps(vl_feat[img2id[image_id]])

                row = {'question_id': question_id, 'image_id': image_id, 'question': question,
                       'answer': __most_common_answer(a_data['annotations'][index]['answers']),
                       'visual_features': visual_feature}

                writer.writerow(row)


def __generate_dictionaries(path_to_csv: str, max_labels: Union[str, int] = 1000) -> None:
    """Create word2id, id2word, label2id, and id2label mappings from a csv of processed data.

    Args:
        path_to_csv: where the processed data lies.
        max_labels: how many labels to map. Can be 'all' or an integer.
    """
    word2idx = dict()
    idx2word = list()
    labels2idx = dict()

    idx2word.append(BOS)
    word2idx[BOS] = 0
    idx2word.append(EOS)
    word2idx[EOS] = 1
    idx2word.append(PAD)
    word2idx[PAD] = 2
    idx2word.append(OOV)
    word2idx[OOV] = 3

    with open(path_to_csv, 'r') as csv_data:
        data = csv.reader(csv_data)
        answers = []

        for (_, _, question, answer, _) in data:
            words = question + ' ' + answer
            answers.append(answer)

            for word in re.split(r'[^\w]+', words):
                lowercase_word = word.lower()

                if lowercase_word not in word2idx:
                    index = len(idx2word)
                    idx2word.append(lowercase_word)
                    word2idx[lowercase_word] = index

    labels = defaultdict(int)
    for answer in answers:
        labels[answer.lower()] += 1

    sorted_answers = sorted(labels, key=labels.get, reverse=True)

    if str(max_labels) == 'all':
        idx2labels = sorted_answers
    else:
        idx2labels = sorted_answers[0:max_labels - 1]

    idx2labels.append(OOV)  # append out of vocabulary word

    for i in range(len(idx2labels)):
        labels2idx[idx2labels[i]] = i

    with open(PROCESSED_FOLDER + IDX2WORD_FILE, 'wb') as fd:
        pickle.dump(idx2word, fd)
    with open(PROCESSED_FOLDER + WORD2IDX_FILE, 'wb') as fd:
        pickle.dump(word2idx, fd)
    with open(PROCESSED_FOLDER + LABELS2IDX_FILE, 'wb') as fd:
        pickle.dump(labels2idx, fd)
    with open(PROCESSED_FOLDER + IDX2LABELS_FILE, 'wb') as fd:
        pickle.dump(idx2labels, fd)