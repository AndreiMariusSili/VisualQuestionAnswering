import argparse
import random
import itertools
import numpy as np

from types import SimpleNamespace
from data.dataset import VQADataset
from main import init_train_save

# constants
RANDOM_SEARCH = 'random'
GRID_SEARCH = 'grid'

LSTM = 'lstm'
BOW = 'bow'

RANDOM_ITERATIONS = 60


def _is_model_better(acc, max_acc):
    return np.mean(acc[-100:]) >= max_acc


def _get_model_name(config_model, config_trainer):
    if config_model.model_type == 'lstm':
        visual_features_location_str = '{}{}{}'.format(
            'i' if 'lstm_input' in config_model.visual_features_location else 'x',
            'c' if 'lstm_context' in config_model.visual_features_location else 'x',
            'o' if 'lstm_output' in config_model.visual_features_location else 'x'
        )

        if not config_model.visual_model:
            visual_features_location_str = 'xxx'

        model_name = '{}_{}_{}_{}drop_{}hidd_{}batch_{}'.format(
            config_model.model_type,
            'pre-embed' if config_model.use_pretrained_embeddings else 'rnd-embed',
            visual_features_location_str,
            str(config_model.lstm_dropout).replace('.', ''),
            config_model.hidden_units,
            config_model.batch_sizes,
            str(config_trainer.lr).replace('.', '')
        )
    else:
        visual_model = 'ooo' if config_model.visual_model else 'xxx'  # to have the same convention as lstm
        model_name = '{}_{}_{}_{}batch_{}'.format(
            config_model.model_type,
            'pre-embed' if config_model.use_pretrained_embeddings else 'rnd-embed',
            visual_model,
            config_model.batch_sizes,
            str(config_trainer.lr).replace('.', '')
        )

    return model_name


def _init_train_save(parameters: dict):
    # migrate parameters from Dictionary() to Namespace()
    config_model = SimpleNamespace()
    config_model.model_type = parameters.get('model_type')
    config_model.embedding_size = parameters.get('embedding_size')
    config_model.hidden_units = parameters.get('lstm_hidden_units')
    config_model.number_stacked_lstms = parameters.get('number_stacked_lstms')
    config_model.visual_model = parameters.get('visual_model')
    config_model.visual_features_location = parameters.get('visual_features_location')
    config_model.use_pretrained_embeddings = parameters.get('use_pretrained_embeddings')
    config_model.img_features_len = parameters.get('img_features_len')
    config_model.lstm_dropout = parameters.get('lstm_dropout')
    config_model.question_max_len = parameters.get('question_max_len')
    config_model.full_size_visual_features = parameters.get('full_size_visual_features')
    config_model.batch_sizes = parameters.get('batch_sizes')

    config_trainer = SimpleNamespace()
    config_trainer.save = parameters.get('save')
    config_trainer.verbose = parameters.get('verbose')
    config_trainer.epochs = parameters.get('epochs')
    config_trainer.lr = parameters.get('lr')

    config_model.model_name = _get_model_name(config_model, config_trainer)

    train_data = VQADataset("train", True, fix_q_len=parameters['question_max_len'])
    val_data = VQADataset("val", True, fix_q_len=parameters['question_max_len'])

    print('\nStart training with parameters: \nconfig_model:{}\nconfig_trainer:{}\n'.format(config_model, config_trainer))

    return init_train_save(config_model, config_trainer, train_data, val_data)


def get_random_parameters(parameter_space: dict) -> dict:
    random_parameters = {}

    for key, value in parameter_space.items():
        random_parameters[key] = random.choice(value)

    return random_parameters


def grid_search_bow(parameter_space: dict) -> (object, list, list, list, str, dict):
    # prepare variables for saving the best model
    best_model = (None, None, None, None, None, None)
    max_acc = 0

    # remove unused parameters
    parameter_space.pop('lstm_hidden_units', None)
    parameter_space.pop('number_stacked_lstms', None)
    parameter_space.pop('lstm_dropout', None)
    parameter_space.pop('visual_features_location', None)
    parameter_space.pop('full_size_visual_features', None)

    # run the grid search
    keys = list(parameter_space.keys())
    values = list(parameter_space.values())
    setups = list(itertools.product(*values))
    parameter_dictionary_list = [dict(zip(keys, setup)) for setup in setups]

    for parameters in parameter_dictionary_list:
        # init/train/save and get accuracies and models
        try:
            model, loss_valid, acc_valid, loss_train, model_name = _init_train_save(parameters)
        except Exception as e:
            exception_str_ = 'Model type {} encountered an exception for parameters:\n\t{}\n{}'
            print(exception_str_.format(parameters['model_type'], parameters, e), flush=True)
            continue

        # print results
        results_str_ = 'Model name: {}\tMax accuracy: {:.4f}, Final accuracy: {:.4f}'
        print(results_str_.format(model_name, max(*acc_valid), acc_valid[-1]), flush=True)

        # update best model
        if _is_model_better(acc_valid, max_acc):
            max_acc = acc_valid[-1]
            best_model = (model, loss_valid, acc_valid, loss_train, model_name, parameters)

    return best_model


def grid_search_lstm(parameter_space: dict) -> (object, list, list, list, str, dict):
    # prepare variables for saving the best model
    best_model = (None, None, None, None, None, None)
    max_acc = 0

    if parameter_space['full_size_visual_features'][0]:
        parameter_space.pop('lstm_hidden_units', None)
        # should also remove 'lstm_context' from list of list, but life is too short.

    # run the grid search
    keys = list(parameter_space.keys())
    values = list(parameter_space.values())
    setups = list(itertools.product(*values))
    parameter_dictionary_list = [dict(zip(keys, setup)) for setup in setups]

    for parameters in parameter_dictionary_list:
        # init/train/save and get accuracies and models
        try:
            model, loss_valid, acc_valid, loss_train, model_name = _init_train_save(parameters)
        except Exception as e:
            exception_str_ = 'Model type {} encountered an exception for parameters:\n\t{}\n{}'
            print(exception_str_.format(parameters['model_type'], parameters, e), flush=True)
            continue

        # print results
        results_str_ = 'Model name: {}\tMax accuracy: {:.4f}, Final accuracy: {:.4f}'
        print(results_str_.format(model_name, max(*acc_valid), acc_valid[-1]), flush=True)

        # update best model
        if _is_model_better(acc_valid, max_acc):
            max_acc = acc_valid[-1]
            best_model = (model, loss_valid, acc_valid, loss_train, model_name, parameters)

    return best_model


def random_search(parameter_space: dict, search_iterations: int) -> (object, list, list, list, str, dict):
    # prepare variables for saving the best model
    best_model = (None, None, None, None, None, None)
    max_acc = 0

    for iteration in range(0, search_iterations):
        # get random parameters
        parameters = get_random_parameters(parameter_space)

        # init/train/save and get accuracies and models
        try:
            model, loss_valid, acc_valid, loss_train, model_name = _init_train_save(parameters)
        except Exception as e:
            exception_str_ = 'Model type {} encountered an exception for parameters:\n\t{}\n{}'
            print(exception_str_.format(parameters['model_type'], parameters, e), flush=True)
            continue

        # print results
        results_str_ = 'Model name: {}\tMax accuracy: {:.4f}, Final accuracy: {:.4f}'
        print(results_str_.format(model_name, max(*acc_valid), acc_valid[-1]), flush=True)

        # update best model
        if _is_model_better(acc_valid, max_acc):
            max_acc = acc_valid[-1]
            best_model = (model, loss_valid, acc_valid, loss_train, model_name, parameters)

    return best_model


def search_hyperparameters_bow(parameter_space: dict, search_type: str, search_iterations: int) -> (object, list, list, list, str, dict):
    if search_type == RANDOM_SEARCH:
        return random_search(parameter_space, search_iterations)
    elif search_type == GRID_SEARCH:
        return grid_search_bow(parameter_space)
    else:
        raise Exception('Argument search_type has an invalid value: {}'.format(search_type))


def search_hyperparameters_lstm(parameter_space: dict, search_type: str, search_iterations: int) -> (object, list, list, list, str, dict):
    if search_type == RANDOM_SEARCH:
        return random_search(parameter_space, search_iterations)
    elif search_type == GRID_SEARCH:
        return grid_search_lstm(parameter_space)
    else:
        raise Exception('Argument search_type has an invalid value: {}'.format(search_type))


def search_hyperparameters(parameter_space: dict, args) -> (object, list, list, list, str, dict):
    if args.model == BOW:
        return search_hyperparameters_bow(parameter_space, args.search_type, args.search_iterations)
    elif args.model == LSTM:
        return search_hyperparameters_lstm(parameter_space, args.search_type, args.search_iterations)
    else:
        raise Exception('Argument args.model has invalid value: {}'.format(args.model))


def main():
    # read command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-type', type=str, default=RANDOM_SEARCH, choices=[RANDOM_SEARCH, GRID_SEARCH])
    parser.add_argument('--search-iterations', type=int, default=RANDOM_ITERATIONS)
    parser.add_argument('--model', type=str, default=BOW, choices=[LSTM, BOW])
    args = parser.parse_args()

    # setup parameter space
    parameter_space = {
        'model_type': [args.model],  # bow, lstm
        'save': [True],
        'verbose': [False],
        'full_size_visual_features': [False],
        'img_features_len': [2048],
        'question_max_len': [20],  # bow
        'embedding_size': [300],
        'number_stacked_lstms': [2],
        'epochs': [100],
        'batch_sizes': [128, 256],
        'lr': [1e-2, 1e-3, 1e-4, 1e-5],
        'lstm_hidden_units': [512, 1024, 2048],
        'visual_model': [False, True],
        'use_pretrained_embeddings': [True, False],
        'lstm_dropout': [0.0, 0.3, 0.5],
        'visual_features_location': [['lstm_input'], ['lstm_context'], ['lstm_output'], ['lstm_context', 'lstm_output']]
    }
    # 'visual_features_location' can be list of any combinations of ['lstm_context', 'lstm_output', 'lstm_input']

    # search optimal hyper-parameters
    model, loss_valid, acc_valid, loss_train, model_name, parameters = search_hyperparameters(parameter_space, args)


if __name__ == "__main__":
    main()
