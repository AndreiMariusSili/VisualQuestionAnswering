import argparse
import random
import itertools

from Trainer import Trainer

# constants
RANDOM_SEARCH = 'random'
GRID_SEARCH = 'grid'

LSTM = 'lstm'
BOW = 'bow'

RANDOM_ITERATIONS = 60


def _init_train_save(parameters: dict):
    return Trainer.init_train_save(embedding_size=parameters['embedding_dims'],
                                   epochs=parameters['nr_epoch'],
                                   lr=parameters['learning_rate'],
                                   question_maxlen=parameters['max_question_lens'],
                                   visual_model=parameters['visual_model'],
                                   hidden_units=parameters['lstm_hidden_units'],
                                   dropout=parameters['dropouts'],
                                   number_stacked_lstms=parameters['number_stacked_lstms'],
                                   adding_mlp=parameters['add_mlp'],
                                   number_mlp_units=parameters['mlp_hidden_units'],
                                   save=True,
                                   modelname=None,
                                   verbose=True,
                                   model_type=parameters['model_type'],
                                   image_features=None,
                                   max_a_len=parameters['max_answers'],
                                   use_pretrained_embeddings=parameters['pre_trained_embedding'],
                                   batch_size=parameters['batch_sizes'],
                                   attention=parameters['attention'],
                                   )


def get_random_parameters(parameter_space: dict) -> dict:
    random_parameters = {}

    for key, value in parameter_space.items():
        random_parameters[key] = random.choice(value)

    return random_parameters


def grid_search_bow(parameter_space: dict) -> (object, list, list, str, dict):

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
        model, loss, acc, model_name = _init_train_save(parameters)

        # print results
        results_str_ = 'Model name: {}\n\tMax accuracy: {0:.4f}, Final accuracy: {0:.4f}'
        print(results_str_.format(model_name, max(*acc), acc[-1]), flush=True)

        # update best model
        if acc[-1] >= max_acc:
            max_acc = acc[-1]
            best_model = (model, loss, acc, model_name, parameters)

    return best_model


def grid_search_lstm(parameter_space: dict) -> (object, list, list, str, dict):

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
        model, loss, acc, model_name = _init_train_save(parameters)

        # print results
        results_str_ = 'Model name: {}\n\tMax accuracy: {0:.4f}, Final accuracy: {0:.4f}'
        print(results_str_.format(model_name, max(*acc), acc[-1]), flush=True)

        # update best model
        if acc[-1] >= max_acc:
            max_acc = acc[-1]
            best_model = (model, loss, acc, model_name, parameters)

    return best_model


def random_search(parameter_space: dict, search_iterations: int, model_type: str) -> (object, list, list, str, dict):
    # prepare variables for saving the best model
    best_model = None
    max_acc = 0

    for iteration in range(0, search_iterations):
        # get random parameters
        parameters = get_random_parameters(parameter_space)

        # add further necessary parameters to list
        parameters['model_type'] = model_type

        # init/train/save and get accuracies and models
        model, loss, acc, model_name = _init_train_save(parameters)

        # print results
        results_str_ = 'Model name: {}\n\tMax accuracy: {0:.4f}, Final accuracy: {0:.4f}'
        print(results_str_.format(model_name, max(*acc), acc[-1]), flush=True)

        # update best model
        if acc[-1] >= max_acc:
            max_acc = acc[-1]
            best_model = (model, loss, acc, model_name, parameters)

    return best_model


def search_hyperparameters_bow(parameter_space: dict, search_type: str, search_iterations: int) -> (object, list, list, str, dict):

    if search_type == RANDOM_SEARCH:
        random_search(parameter_space, search_iterations, BOW)
    elif search_type == GRID_SEARCH:
        grid_search_bow(parameter_space)
    else:
        raise Exception('Argument search_type has an invalid value: {}'.format(search_type))


def search_hyperparameters_lstm(parameter_space: dict, search_type: str, search_iterations: int) -> (object, list, list, str, dict):

    if search_type == RANDOM_SEARCH:
        return random_search(parameter_space, search_iterations, LSTM)
    elif search_type == GRID_SEARCH:
        return grid_search_lstm(parameter_space)
    else:
        raise Exception('Argument search_type has an invalid value: {}'.format(search_type))


def search_hyperparameters(parameter_space: dict, args) -> (object, list, list, str, dict):
    if args.model == BOW:
        return search_hyperparameters_bow(parameter_space, args.search_type, args.search_iterations)
    elif args.model == LSTM:
        return search_hyperparameters_lstm(parameter_space, args.search_type, args.search_iterations)
    else:
        raise Exception('Argument args.model has invalid value: {}'.format(args.model))


def main():
    # read command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-type", type=str, default=RANDOM_SEARCH, choices=[RANDOM_SEARCH, GRID_SEARCH])
    parser.add_argument("--search-iterations", type=int, default=RANDOM_ITERATIONS)
    parser.add_argument("--model", type=str, default=LSTM, choices=[LSTM, BOW])
    args = parser.parse_args()

    # setup parameter space
    parameter_space = {
        'nr_epoch': [5, 8, 10],
        'batch_sizes': [32, 64, 128],
        'learning_rate': [1e-2, 1e-3, 1e-4, 1e-5],
        'max_question_lens': [10, 15, 20, 30],
        'max_answers': [500, 1000, 2000, 4000, 'all'],
        'embedding_dims': [200, 300, 400, 600],
        'lstm_hidden_units': [256, 512, 104],
        'number_stacked_lstms': [0, 1, 2],
        'add_mlp': [False, True],
        'mlp_hidden_units': [512, 1024, 2048],
        'dropouts': [0.3, 0.4, 0.5],
        'visual_model': [False, True],
        'attention': [False, True],
        'pre_trained_embedding': [True, False],
    }

    # search optimal hyper-parameters
    model, loss, acc, model_name, parameters = search_hyperparameters(parameter_space, args)


if __name__ == "__main__":
    main()
