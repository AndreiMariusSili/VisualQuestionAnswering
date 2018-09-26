from data import processor
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', required=True, type=str, choices=['process'], help='Which action to perform.')
    args, _ = parser.parse_known_args()

    if args.task == 'process':
        processor.run()
