import argparse
import random
import itertools
import numpy as np
from types import SimpleNamespace
from data.loader import VQALoader
from main import init_train_save


# constants
RANDOM_SEARCH = 'random'
GRID_SEARCH = 'grid'

LSTM = 'lstm'
BOW = 'bow'

RANDOM_ITERATIONS = 60


def _is_model_better(acc, max_acc):
    return np.mean(acc[-100:]) >= max_acc


def _init_train_save(parameters: dict):
    # migrate parameters from Dictionary() to Namespace()
    config_model = SimpleNamespace()
    config_model.model_type = parameters['model_type']
    config_model.embedding_size = parameters['embedding_dims']
    config_model.hidden_units = parameters['lstm_hidden_units']
    config_model.number_stacked_lstms = parameters['number_stacked_lstms']
    config_model.visual_model = parameters['visual_model']
    config_model.visual_features_location = ['lstm_context', 'lstm_output']  # ['lstm_context', 'lstm_output', 'lstm_input']
    config_model.use_pretrained_embeddings = parameters['pre_trained_embedding']
    config_model.embedding_size = parameters['embedding_dims']
    config_model.image_features = parameters['image_features']

    config_trainer = SimpleNamespace()
    config_trainer.save = True
    config_trainer.verbose = True
    config_trainer.epochs = parameters['nr_epoch']
    config_trainer.lr = parameters['learning_rate']

    train_data = VQALoader("train", True, True, parameters['batch_sizes'], fix_q_len=parameters['max_question_lens'], fix_a_len=parameters['max_answers'])
    val_data = VQALoader("val", True, True, parameters['batch_sizes'], num_workers=0, fix_q_len=parameters['max_question_lens'], fix_a_len=parameters['max_answers'])

    return init_train_save(config_model, config_trainer, train_data, val_data)


def get_random_parameters(parameter_space: dict) -> dict:
    random_parameters = {}

    for key, value in parameter_space.items():
        random_parameters[key] = random.choice(value)

    return random_parameters


def grid_search_bow(parameter_space: dict) -> (object, list, list, list, str, dict):
    # prepare variables for saving the best model
    best_model = None
    max_acc = 0

    # remove unused parameters
    parameter_space.pop('lstm_hidden_units', None)
    parameter_space.pop('number_stacked_lstms', None)
    parameter_space.pop('add_mlp', None)
    parameter_space.pop('mlp_hidden_units', None)
    parameter_space.pop('dropouts', None)
    parameter_space.pop('attention', None)

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
    best_model = None
    max_acc = 0

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
    best_model = None
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
    parser.add_argument('--model', type=str, default=LSTM, choices=[LSTM, BOW])
    args = parser.parse_args()

    # setup parameter space
    parameter_space = {
        'nr_epoch': [5, 8, 10],
        'batch_sizes': [32, 64, 128],
        'learning_rate': [1e-2, 1e-3, 1e-4, 1e-5],
        'image_features': [2048],
        'max_question_lens': [10, 15, 20, 30],
        'max_answers': [500, 1000, 2000, 4000, 'all'],
        'embedding_dims': [200, 300, 400, 600],
        'lstm_hidden_units': [256, 512, 104],
        'number_stacked_lstms': [0, 1, 2],
        'visual_model': [False, True],
        'pre_trained_embedding': [True, False],
        'model_type': [args.model],
    }

    # search optimal hyper-parameters
    model, loss_valid, acc_valid, loss_train, model_name, parameters = search_hyperparameters(parameter_space, args)


if __name__ == "__main__":
    main()
