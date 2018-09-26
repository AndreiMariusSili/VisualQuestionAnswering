import csv
from typing import Union, Tuple
import torch.utils.data as data
from misc.constants import *
import pandas as pd
import pickle
import torch
import os
import ast
import linecache

RETURNED_ITEM = Union[Tuple[torch.Tensor, None, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class VQADataset(data.Dataset):
    _data: pd.DataFrame

    def __init__(self, part: str, img_feats: bool, in_memory: bool):
        assert part in ['train', 'test', 'val'], 'part must be one of "train", "test", "val"'
        self._part = part
        self._img_feats = img_feats
        self._in_memory = in_memory

        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        with open(os.path.join(PROCESSED_FOLDER, IDX2LABELS_FILE), 'rb') as f:
            self._idx2labels = pickle.load(f)
        with open(os.path.join(PROCESSED_FOLDER, LABELS2IDX_FILE), 'rb') as f:
            self._labels2idx = pickle.load(f)
        with open(os.path.join(PROCESSED_FOLDER, IDX2WORD_FILE), 'rb') as f:
            self._idx2word = pickle.load(f)
        with open(os.path.join(PROCESSED_FOLDER, WORD2IDX_FILE), 'rb') as f:
            self._word2idx = pickle.load(f)

        part_to_file = {
            'train': PROCESSED_TRAIN_FILE,
            'val': PROCESSED_VAL_FILE,
            'test': PROCESSED_TEST_FILE
        }

        csv_file_path = os.path.join(PROCESSED_FOLDER, part_to_file[part])
        if in_memory:
            self._data = pd.read_csv(csv_file_path)
        else:
            self._fields = ['question_id', 'image_id', 'question', 'answer', 'visual_features']
            self._csv_file_path = csv_file_path

    def __getitem__(self, item: int) -> RETURNED_ITEM:
        x_img = None

        if self._in_memory:
            row = self._data.iloc[item]
        else:
            row = linecache.getline(self._csv_file_path, item + 1)
            row = list(csv.reader([row]))[0]
            row = {self._fields[idx]: value for idx, value in
                   enumerate(row)}

        x_nlp = [self._word2idx[word.lower()] for word in row['question'].split()[:-1]]
        x_nlp = torch.tensor(x_nlp, device=self._device, dtype=torch.long)
        if self._img_feats:
            x_img = torch.tensor(pickle.loads(ast.literal_eval(row['visual_features'])), device=self._device, dtype=torch.float)
        t = [self._labels2idx[label.lower()] for label in row['answer'].split()]
        t = torch.tensor(t, device=self._device, dtype=torch.long)

        return x_nlp, x_img, t

    def __len__(self) -> int:
        return len(self._data)
