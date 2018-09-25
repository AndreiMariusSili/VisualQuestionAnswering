import argparse
from bow import BOW
from lstm import LSTM
from preprocess import Preprocess
from dictionary import Dictionary
from constants import *

def bow_hyperparameter_search(embedding_dims, max_question_lens, max_answers):
    helper = Preprocess()
    helper.preprocess()
    dictionary = Dictionary(helper, 'all')

    for embedding_dim in embedding_dims:
        print('Building model BOW-all-answers' + '-embedding_dim=' + str(embedding_dim))
        bow = BOW(dictionary, 15, embedding_dim, visual_model=True)
        model = bow.train(verbose=2)
        acc = bow.evaluate(model, test_data_file=val_data_write_file)
        print('Model evaluated, acc=' + str(acc))

def lstm_hyperparameter_search(number_hidden_units, dropouts, number_stacked_lstms, add_mlp, mlp_hidden_units):
    helper = Preprocess()
    helper.preprocess()

    best_embedding_length_bow = 300 #tbd
    best_max_anwers_bow = 500 #tbd
    best_question_maxlen_bow = 15 #Tbd

    dictionary = Dictionary(helper, best_max_anwers_bow)

    for number_h_units in number_hidden_units:
        for dropout in dropouts:
            for number_stacked_lstm in number_stacked_lstms:
                for adding_mlp in add_mlp:
                    lstm = None
                    if adding_mlp:
                        for number_mlp_hidden_units in mlp_hidden_units:
                            print('Building model LSTM-lstm_h_units=' + str(number_h_units) + '-dropout=' + str(dropout) + '-nr_stacked_lstm=' + str(number_stacked_lstm) + 'mlp_hidden_units=' + str(number_mlp_hidden_units))
                            lstm = LSTM(dictionary, question_maxlen=best_question_maxlen_bow, embedding_vector_length=best_embedding_length_bow, lstm_hidden_units=number_h_units, dropout=dropout, recurrent_dropout=dropout, number_stacked_lstms=number_stacked_lstm, adding_mlp=adding_mlp, number_mlp_units=number_mlp_hidden_units)
                            model = lstm.train(verbose=2, train_data_file=train_data_write_file)
                            acc = lstm.evaluate(model, test_data_file=val_data_write_file)
                            print('Model evaluated, acc=' + str(acc))
                    else:
                        print('Building model LSTM-lstm_h_units=' + str(number_h_units) + '-dropout=' + str(dropout) + '-nr_stacked_lstm=' + str(number_stacked_lstm))
                        lstm = LSTM(dictionary, question_maxlen=best_question_maxlen_bow, embedding_vector_length=best_embedding_length_bow, lstm_hidden_units=number_h_units, dropout=dropout, recurrent_dropout=dropout, number_stacked_lstms=number_stacked_lstm, adding_mlp=0)
                        model = lstm.train(verbose=2, train_data_file=train_data_write_file)
                        acc = lstm.evaluate(model, test_data_file=val_data_write_file)
                        print('Model evaluated, acc=' + str(acc))


def training_hyperparameter_search(nr_epoch, batch_size):
    best_embedding_length_bow = 300 #tbd
    best_max_anwers_bow = 500 #tbd
    best_question_maxlen_bow = 15 #Tbd

    helper = Preprocess()
    helper.preprocess()
    dictionary = Dictionary(helper, best_max_anwers_bow)

    for epoch in nr_epoch:
        for batch_size in batch_size:
            bow = BOW(dictionary, best_question_maxlen_bow, best_embedding_length_bow, visual_model=True)
            model = bow.train(verbose=2, train_data_file=train_data_write_file, epochs=epoch, batch_size=batch_size)
            acc = bow.evaluate(model, test_data_file=val_data_write_file)
            print('Model evaluated, acc=' + str(acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter_type_search", type=str, default="bow")

    args = parser.parse_args()

    max_question_len = [10, 15, 20, 30] # analyze all the questions
    embedding_dim = [200, 300, 400 , 600]
    max_answers = [500, 1000, 2000, 4000, 'all']

    number_hidden_units = [256, 512, 104]
    dropout = [0.3, 0.4, 0.5]
    number_stacked_lstms = [0, 1, 2]
    mlp_hidden_units = [512, 1024, 2048]
    add_mlp = [0, 1]

    nr_epoch = [5, 8, 10]
    batch_size = [32, 64, 128]

    if args.parameter_type_search == 'bow':
        bow_hyperparameter_search(embedding_dim, max_question_len, max_answers)
    elif args.parameter_type_search == 'lstm':
        lstm_hyperparameter_search(number_hidden_units, dropout, number_stacked_lstms, add_mlp, mlp_hidden_units)
    elif args.parameter_type_search == 'train':
        training_hyperparameter_search(nr_epoch, batch_size)