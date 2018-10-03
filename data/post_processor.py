import pickle

from misc.constants import *

import pandas as pd

import os


def run():
    print('Creating binaries for full dataset...')
    _pickle_full_dataset()

    print('Creating binaries for OOV-answer-filtered dataset... old - new = filtered')
    _filter_oov_answers()

    print('Creating binaries for dummy dataset...')
    _pickle_dummy_dataset()


def _pickle_full_dataset():
    part_to_file = {
        'train': {
            'csv': PROCESSED_TRAIN_FILE,
            'gzip': FULL_TRAIN_SET_FILE
        },
        'val': {
            'csv': PROCESSED_VAL_FILE,
            'gzip': FULL_VAL_SET_FILE
        },
        'test': {
            'csv': PROCESSED_TEST_FILE,
            'gzip': FULL_TEST_SET_FILE
        }
    }

    for part in part_to_file:
        print('Loading {} set into DataFrame and storing as binary...'.format(part), end='')
        csv_file_path = os.path.join(PROCESSED_FOLDER, part_to_file[part]['csv'])
        data = pd.read_csv(csv_file_path)
        data['visual_features'] = data['visual_features'].apply(lambda x: pickle.loads(eval(x)))
        gzip_file_path = os.path.join(PROCESSED_FOLDER, part_to_file[part]['gzip'])
        data.to_pickle(gzip_file_path)
        print('Done')


def _filter_oov_answers():
    part_to_file = {
        'train': {
            'full': FULL_TRAIN_SET_FILE,
            'filtered': FILTERED_TRAIN_SET_FILE
        },
        'val': {
            'full': FULL_VAL_SET_FILE,
            'filtered': FILTERED_VAL_SET_FILE
        },
        'test': {
            'full': FULL_TEST_SET_FILE,
            'filtered': FILTERED_TEST_SET_FILE
        }
    }
    with open(os.path.join(PROCESSED_FOLDER, LABEL2IDX_FILE), 'rb') as f:
        labels2idx = pickle.load(f)

    for part in part_to_file:
        print('Loading full {} set binaries and filtering...'.format(part), end='')
        data = pd.read_pickle(os.path.join(PROCESSED_FOLDER, part_to_file[part]['full']))

        mask = []
        for answer in data['answer']:
            mask.append(bool(labels2idx.get(answer, None)))

        full_len = len(data)
        data = data[mask]
        filter_len = len(data)
        print('{} - {} = {}...'.format(full_len, filter_len, full_len - filter_len), end='')
        data.to_pickle(os.path.join(PROCESSED_FOLDER, part_to_file[part]['filtered']))
        print('Done')


def _pickle_dummy_dataset():
    print('Loading dummy set into DataFrame and storing as binary...', end='')
    data = pd.read_pickle(os.path.join(PROCESSED_FOLDER, FILTERED_TRAIN_SET_FILE))
    data = data[0: 128]
    data.to_pickle(os.path.join(PROCESSED_FOLDER, DUMMY_TRAIN_SET_FILE))
    print('Done')
