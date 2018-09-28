from datetime import datetime, timedelta
from data import processor
import argparse

from data.loader import VQALoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', required=True, type=str, choices=['process', 'test_loader'],
                        help='Which action to perform.')
    parser.add_argument('-l', '--labels', type=int, default=1000,
                        help='How many target labels to tokenize(rest is treated as OOV)')
    args, _ = parser.parse_known_args()

    if args.task == 'process':
        processor.run(args.labels)
    if args.task == 'test_loader':
        loader = VQALoader('train', False, True, 64).get()
        start = datetime.now()
        time = 0
        for step, (x_nlp, x_img, t) in enumerate(loader):
            time += (datetime.now() - start).microseconds
            start = datetime.now()
        print("InMemoryLoaderNoVisual | Steps: {} Average Time: {}".format(step + 1, timedelta(microseconds=time / (step + 1))))

        loader = VQALoader('train', True, True, 64, fix_q_len=2, fix_a_len=1).get()
        start = datetime.now()
        time = 0
        for step, (x_nlp, x_img, t) in enumerate(loader):
            time += (datetime.now() - start).microseconds
            start = datetime.now()
        print("InMemoryLoaderWithVisual | Steps: {} Average Time: {}".format(step + 1, timedelta(microseconds=time / (step + 1))))

        loader = VQALoader('train', False, False, 64).get()
        start = datetime.now()
        time = 0
        for step, (x_nlp, x_img, t) in enumerate(loader):
            time += (datetime.now() - start).microseconds
            start = datetime.now()
        print("OnDiskLoaderNoVisual | Steps: {} Average Time: {}".format(step + 1, timedelta(microseconds=time / (step + 1))))

        loader = VQALoader('train', True, False, 64, fix_q_len=2, fix_a_len=1).get()
        start = datetime.now()
        time = 0
        for step, (x_nlp, x_img, t) in enumerate(loader):
            time += (datetime.now() - start).microseconds
            start = datetime.now()
        print("OnDiskLoaderWithVisual | Steps: {} Average Time: {}".format(step + 1, timedelta(microseconds=time / (step + 1))))
