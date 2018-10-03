import os
from datetime import datetime, timedelta
from data import processor, post_processor
from data.dataset import VQADataset
import torch.utils.data as data
import argparse


def test():
    train_dataset = VQADataset('train', False, 20)
    train_loader = data.DataLoader(train_dataset, 64, True,
                                   num_workers=os.cpu_count(), collate_fn=train_dataset.collate_fn)
    start = datetime.now()
    time = 0
    step = 0
    for step, (_, _, _) in enumerate(train_loader):
        time += (datetime.now() - start).microseconds
        start = datetime.now()
    print("TrainDataset | No Visual | Fixed Length | Steps: {} Average Time: {}"
          .format(step + 1, timedelta(microseconds=time / (step + 1))))

    train_dataset = VQADataset('train', True)
    train_loader = data.DataLoader(train_dataset, 64, True,
                                   num_workers=os.cpu_count(), collate_fn=train_dataset.collate_fn)
    start = datetime.now()
    time = 0
    step = 0
    for step, (_, _, _) in enumerate(train_loader):
        time += (datetime.now() - start).microseconds
        start = datetime.now()
    print("TrainDataset | With Visual | Variable Length | Steps: {} Average Time: {}"
          .format(step + 1, timedelta(microseconds=time / (step + 1))))

    train_dataset = VQADataset('train', True, 20)
    train_loader = data.DataLoader(train_dataset, 64, True,
                                   num_workers=os.cpu_count(), collate_fn=train_dataset.collate_fn)
    start = datetime.now()
    time = 0
    step = 0
    for step, (_, _, _) in enumerate(train_loader):
        time += (datetime.now() - start).microseconds
        start = datetime.now()
    print("TrainDataset | With Visual | Fixed Length | Steps: {} Average Time: {}"
          .format(step + 1, timedelta(microseconds=time / (step + 1))))

    train_dataset = VQADataset('train', False)
    train_loader = data.DataLoader(train_dataset, 64, True,
                                   num_workers=os.cpu_count(), collate_fn=train_dataset.collate_fn)
    start = datetime.now()
    time = 0
    step = 0
    for step, (_, _, _) in enumerate(train_loader):
        time += (datetime.now() - start).microseconds
        start = datetime.now()
    print("TrainDataset | No Visual | Variable Length | Steps: {} Average Time: {}"
          .format(step + 1, timedelta(microseconds=time / (step + 1))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', required=True, type=str, choices=['process', 'post_process', 'test', 'magic'],
                        help='Which action to perform.')
    parser.add_argument('-l', '--labels', type=int, default=1000,
                        help='How many target labels to tokenize(rest is treated as OOV)')
    args, _ = parser.parse_known_args()
    if args.task == 'magic':
        processor.run(args.labels)
        post_processor.run()
        test()
    if args.task == 'process':
        processor.run(args.labels)
    elif args.task == 'post_process':
        post_processor.run()
    elif args.task == 'test':
        test()
