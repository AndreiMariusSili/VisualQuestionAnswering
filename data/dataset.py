from typing import Union, Tuple, List, Dict
import torch.utils.data as data
from datetime import datetime
from misc.constants import *
import pandas as pd
import numpy as np
import pickle
import torch
import os

RETURNED_ITEM = Union[Tuple[List[int], None, List[int]], Tuple[List[int], np.ndarray, List[int]]]
UNCOLLATED_BATCH = List[Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor]]
COLLATED_BATCH = Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor]


class VQADataset(data.Dataset):
    """Loads the VQA Dataset in memory as a pandas DataFrame. Should be iterated over using the torch DataLoader
        utility.
    """
    _part: str
    _img_feats: bool
    _fix_q_len: int
    _fix_a_len: int
    _dummy: bool

    _data: pd.DataFrame
    _idx2word: List[str]
    _idx2label: List[str]
    _label2idx: Dict[str, int]

    @property
    def word2idx(self) -> Dict[str, int]:
        return self._word2idx

    @property
    def label2idx(self) -> Dict[str, int]:
        return self._label2idx

    @property
    def idx2word(self) -> List[str]:
        return self._idx2word

    @property
    def idx2label(self) -> List[str]:
        return self._idx2label

    @property
    def vocab_size(self) -> int:
        return len(self.idx2word)

    @property
    def output_size(self)-> int:
        return len(self.idx2label)

    def __init__(self, part: str, img_feats: bool, fix_q_len: int = None):
        """Initialise dataset. Reads the entire dataset and the word/label to id maps into memory.

        Args:
            part: Choice between ('train', 'val', 'test', 'dummy')
            img_feats: Whether to retrieve image features or not.
            fix_q_len: Maximum question length.
        """
        assert part in ['train', 'val', 'test', 'dummy']

        start = datetime.now()

        self._part = part
        self._img_feats = img_feats
        self._fix_q_len = fix_q_len

        self._question_max_length = fix_q_len if fix_q_len is not None else 0

        with open(os.path.join(PROCESSED_FOLDER, IDX2LABEL_FILE), 'rb') as f:
            self._idx2label = pickle.load(f)
        with open(os.path.join(PROCESSED_FOLDER, LABEL2IDX_FILE), 'rb') as f:
            self._label2idx = pickle.load(f)
        with open(os.path.join(PROCESSED_FOLDER, IDX2WORD_FILE), 'rb') as f:
            self._idx2word = pickle.load(f)
        with open(os.path.join(PROCESSED_FOLDER, WORD2IDX_FILE), 'rb') as f:
            self._word2idx = pickle.load(f)

        part_to_file = {
            'train': FILTERED_TRAIN_SET_FILE,
            'val': FILTERED_VAL_SET_FILE,
            'test': FILTERED_TEST_SET_FILE,
            'dummy': DUMMY_TRAIN_SET_FILE
        }
        self._data = pd.read_pickle(os.path.join(PROCESSED_FOLDER, part_to_file[part]))

        print('VQADataset initialised in {}'.format(datetime.now() - start))

    def convert_answer_to_string(self, answer: List[int]):
        """ Converts a list of labels to its string representation.
        Args:
            answer: A list of integer labels.
        Returns:
            The string representation of the answer.
        """

        return " ".join([self.idx2label[idx] for idx in answer])

    def convert_question_to_string(self, question: List[int]):
        """ Converts a list of words to its string representation.
        Args:
            question: A list of integer words.
        Returns:
            The string representation of the answer.
        """

        return " ".join([self.idx2word[idx] for idx in question])

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

        answer = row['answer'].strip('?!.')
        t = [self._label2idx.get(answer.lower(), self._label2idx[OOV])]

        if self._img_feats:
            x_img = row['visual_features']

        return x_nl, x_img, t

    def __len__(self) -> int:
        return len(self._data)

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

        q_len_max = self._fix_q_len
        if self._fix_q_len is None:
            q_len_max = max([len(q) for q, _, _ in batch])

        for (question, visual_features, target) in batch:
            q_len_diff = q_len_max - len(question)
            if q_len_diff >= 0:
                required_padding = [self._word2idx[PAD]] * q_len_diff
                question.extend(required_padding)
            else:
                question = question[:q_len_diff - 1]
                question.append(self._word2idx[EOS])
            x_nl_batch.append(torch.tensor(question, dtype=torch.long))

            t_batch.append(torch.tensor(target, dtype=torch.long))

            if self._img_feats:
                x_img_batch.append(torch.tensor(visual_features, dtype=torch.float))

        if self._img_feats:
            return torch.stack(x_nl_batch, dim=0), torch.stack(x_img_batch, dim=0), torch.stack(t_batch, dim=0)
        else:
            return torch.stack(x_nl_batch, dim=0), None, torch.stack(t_batch, dim=0)
