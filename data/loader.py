from typing import Union, Tuple, List, Dict
import torch.utils.data as data
from datetime import datetime
from misc.constants import *
import pandas as pd
import linecache
import pickle
import torch
import ast
import csv
import os

RETURNED_ITEM = Union[Tuple[List[int], None, List[int]], Tuple[List[int], torch.Tensor, List[int]]]
UNCOLLATED_BATCH = List[Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor]]
COLLATED_BATCH = Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor]


class VQADatasetInMemory(data.Dataset):
    """Loads the VQA Dataset in memory as a pandas DataFrame. Should be iterated over using the torch DataLoader
    utility. This class provides faster batch retrieval, but slower initialisation. Useful especially for GPU training.
    """

    _data: pd.DataFrame

    def __init__(self, part: str, retrieve_vl_feat: bool, fix_q_len: int, fix_a_len: int):
        """Initialise dataset. Reads the entire dataset and the word/label to id maps into memory.

        Args:
            part: the dataset to load. Choice of: train, test, or val
            retrieve_vl_feat: Whether to retrieve visual_features or not.
            fix_q_len: Maximum question length.
            fix_a_len: Maximum answer length.
        """
        self._fix_a_len = fix_a_len
        self._fix_q_len = fix_q_len

        self._question_max_length = fix_q_len if fix_q_len is not None else 0
        self._answer_max_length = fix_a_len if fix_a_len is not None else 0

        start = datetime.now()

        assert part in ['train', 'test', 'val'], 'part must be one of "train", "test", "val"'
        self._part = part
        self._img_feats = retrieve_vl_feat

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
        self._data = pd.read_csv(csv_file_path)
        self._data['visual_features'] = self._data['visual_features'].apply(lambda x: pickle.loads(ast.literal_eval(x)))

        print('InMemoryLoader | Initialised dataset in {}'.format(datetime.now() - start))

    def __getitem__(self, item: int) -> RETURNED_ITEM:
        """Retrieves an item from the dataset at the index location provided by 'item'. Takes care of normalizing
        question strings and saves statistics for latter processing.

        Args:
            item: the index of the row to be retrieved.

        Returns:
            A tuple of tensors: natural language features, image features, targets. The image features may be None
            if those were not requested.
        """
        x_img = None
        row = self._data.iloc[item]

        question = (BOS + ' ' + row['question'].strip('?!.') + ' ' + EOS).split()
        x_nl = [self._word2idx.get(word.lower(), self._word2idx[OOV]) for word in question]
        if self._fix_q_len is None and len(x_nl) > self._question_max_length:
            self._question_max_length = len(x_nl)

        answer = row['answer'].strip('?!.').split()
        t = [self._labels2idx.get(label.lower(), self._labels2idx[OOV]) for label in answer]
        if self._fix_a_len is None and len(t) > self._answer_max_length:
            self._answer_max_length = len(t)

        if self._img_feats:
            x_img = torch.tensor(row['visual_features'], dtype=torch.float)

        return x_nl, x_img, t

    def collate_fn(self, batch: UNCOLLATED_BATCH) -> COLLATED_BATCH:
        """Process a list of data points (natural language features, image features, targets) into a torch Tensor. The
        sequences are padded at the end to have the same length. Each element in the tuple is stacked in its own Tensor.

        Args:
            batch: a list of tuples ((natural language features, image features, targets)).

        Returns:
            a tuple of tensors: (natural language features, image features, targets)
        """
        x_nl_batch = []
        x_img_batch = []
        t_batch = []

        for (question, visual_features, target) in batch:
            q_len_diff = self._question_max_length - len(question)
            if q_len_diff >= 0:
                required_padding = [self._word2idx[PAD]] * q_len_diff
                question.extend(required_padding)
            else:
                question = question[:q_len_diff - 1]
                question.append(self._word2idx[EOS])
            x_nl_batch.append(torch.tensor(question, dtype=torch.long))

            a_len_diff = self._answer_max_length - len(target)
            if a_len_diff >= 0:
                required_padding = [self._word2idx[PAD]] * a_len_diff
                target.extend(required_padding)
            else:
                target = target[:a_len_diff]
            t_batch.append(torch.tensor(target, dtype=torch.long))

            x_img_batch.append(visual_features)

        if self._img_feats:
            return torch.stack(x_nl_batch, dim=0), torch.stack(x_img_batch, dim=0), torch.stack(t_batch, dim=0)
        else:
            return torch.stack(x_nl_batch, dim=0), None, torch.stack(t_batch, dim=0)

    def __len__(self) -> int:
        return len(self._data)


class VQADatasetOnDisk(data.Dataset):
    """Initialises a linecache reader to retrieve rows from the csv dataset. All pre-processing is done at batch
    retrieval time. Should be iterated over using the torch DataLoader utility. This class provides slower batch
    retrieval, but much faster initialisation. Useful especially for experimenting on CPU.
    """

    def __init__(self, part: str, retrieve_vl_feat: bool, fix_q_len: int, fix_a_len: int):
        """Initialise dataset. Only determines the size of the csv file and stores the word/label to id maps into memory.

        Args:
            part: the dataset to load. Choice of: train, test, or val
            retrieve_vl_feat: Whether to retrieve visual_features or not.
        """
        start = datetime.now()

        assert part in ['train', 'test', 'val'], 'part must be one of "train", "test", "val"'
        self._part = part
        self._img_feats = retrieve_vl_feat

        self._fix_a_len = fix_a_len
        self._fix_q_len = fix_q_len

        self._question_max_length = fix_q_len if fix_q_len is not None else 0
        self._answer_max_length = fix_a_len if fix_a_len is not None else 0

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

        self._csv_file_path = os.path.join(PROCESSED_FOLDER, part_to_file[part])
        self._size = sum(1 for _ in open(self._csv_file_path, 'r'))
        self._fields = {'question_id': 0, 'image_id': 1, 'question': 2,
                        'answer': 3, 'visual_features': 4}

        print('OnDiskLoader | Initialised dataset in {}'.format(datetime.now() - start))

    def __getitem__(self, item: int) -> RETURNED_ITEM:
        """Retrieves an item from the dataset by reading and parsing the line of the file specified by 'item'.
        Takes care of normalizing question strings and saves statistics for latter processing.

        Args:
            item: the index of the row to be retrieved.

        Returns:
            A tuple of tensors: natural language features, image features, targets. The image features may be None
            if those were not requested.
        """
        item += 1  # add 1 to prevent reading the headings.
        x_img = None

        raw_row = linecache.getline(self._csv_file_path, item + 1)
        row = list(csv.reader([raw_row]))[0]

        question = (BOS + ' ' + row[self._fields['question']].strip('?!.') + ' ' + EOS).split()
        x_nl = [self._word2idx.get(word.lower(), self._word2idx[OOV]) for word in question]
        if self._fix_q_len is None and len(x_nl) > self._question_max_length:
            self._question_max_length = len(x_nl)

        answer = row[self._fields['answer']].strip('?!.').split()
        t = [self._labels2idx.get(label.lower(), self._labels2idx[OOV]) for label in answer]
        if self._fix_a_len is None and len(t) > self._answer_max_length:
            self._answer_max_length = len(t)

        if self._img_feats:
            visual_features = pickle.loads(ast.literal_eval(row[self._fields['visual_features']]))
            x_img = torch.tensor(visual_features, dtype=torch.float)

        return x_nl, x_img, t

    def collate_fn(self, batch: UNCOLLATED_BATCH) -> COLLATED_BATCH:
        """Process a list of data points (natural language features, image features, targets) into a torch Tensor. The
        sequences are padded at the end to have the same length. Each element in the tuple is stacked in its own Tensor.

        Args:
            batch: a list of tuples ((natural language features, image features, targets)).

        Returns:
            a tuple of tensors: (natural language features, image features, targets)
        """
        x_nl_batch = []
        x_img_batch = []
        t_batch = []

        for (question, visual_features, target) in batch:
            q_len_diff = self._question_max_length - len(question)
            if q_len_diff >= 0:
                required_padding = [self._word2idx[PAD]] * q_len_diff
                question.extend(required_padding)
            else:
                question = question[:q_len_diff-1]
                question.append(self._word2idx[EOS])
            x_nl_batch.append(torch.tensor(question, dtype=torch.long))

            a_len_diff = self._answer_max_length - len(target)
            if a_len_diff >= 0:
                required_padding = [self._word2idx[PAD]] * a_len_diff
                target.extend(required_padding)
            else:
                target = target[:a_len_diff]
            t_batch.append(torch.tensor(target, dtype=torch.long))

            x_img_batch.append(visual_features)

        if self._img_feats:
            return torch.stack(x_nl_batch, dim=0), torch.stack(x_img_batch, dim=0), torch.stack(t_batch, dim=0)
        else:
            return torch.stack(x_nl_batch, dim=0), None, torch.stack(t_batch, dim=0)

    def __len__(self) -> int:
        return self._size - 1  # -1 because I am adding 1 to the index in __getitem__


class VQALoader(object):
    def __init__(self, part: str, retrieve_visual_features: bool, store_in_memory: bool, batch_size,
                 num_workers=os.cpu_count(), fix_q_len: int = None, fix_a_len: int = None):
        """Initialises a DataLoader object with the VQA dataset

        Args:
            part: what data to load: train, test, val.
            retrieve_visual_features: whether to retrieve visual features.
            store_in_memory: whether to store the dataset in memory.
            batch_size: what batch size to use.
        """
        if store_in_memory:
            self._dataset = VQADatasetInMemory(part, retrieve_visual_features, fix_q_len=fix_q_len,
                                               fix_a_len=fix_a_len)
            self._loader = data.DataLoader(self._dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=self._dataset.collate_fn,
                                           num_workers=num_workers)
        else:
            self._dataset = VQADatasetOnDisk(part, retrieve_visual_features, fix_q_len=fix_q_len, fix_a_len=fix_a_len)
            self._loader = data.DataLoader(self._dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=self._dataset.collate_fn,
                                           num_workers=num_workers)

    def get(self):
        return self._loader

    @property
    def word2idx(self) -> Dict[str, int]:
        return self._dataset._word2idx

    @property
    def label2idx(self) -> Dict[str, int]:
        return self._dataset._labels2idx

    @property
    def idx2word(self) -> List[str]:
        return self._dataset._idx2word

    @property
    def idx2label(self) -> List[str]:
        return self._dataset._idx2labels
