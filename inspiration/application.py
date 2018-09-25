from preprocess import Preprocess
from dictionary import Dictionary
from lstm import LSTM
from keras.models import load_model
from constants import *
from utils import *
from bow import BOW
import argparse
import numpy as np
from rnn import RNN


def train_and_evaluate(args):
    helper = Preprocess()
    helper.preprocess()

    dictionary = Dictionary(helper, args.max_answers)
    lstm = LSTM(dictionary, question_maxlen=args.max_question_len, embedding_vector_length=args.embedding_length,
                visual_model=args.visual_model, lstm_hidden_units=args.number_hidden_units, dropout=args.dropout,
                recurrent_dropout=args.r_dropout, number_stacked_lstms=args.number_stacked_lstms, adding_mlp=1, number_mlp_units=args.number_mlp_unts)
    rnn = RNN(dictionary, question_maxlen=args.max_question_len, embedding_vector_length=args.embedding_length,
                visual_model=args.visual_model, rnn_hidden_units=args.number_hidden_units, dropout=args.dropout,
                recurrent_dropout=args.r_dropout)
    bow = BOW(dictionary, question_maxlen=args.max_question_len, embedding_vector_length=args.embedding_length,
              visual_model=args.visual_model)

    if (not args.model_name):
        print('Building model BOW-max_answers=' + str(args.max_answers) + '-embedding_dim=' + str(args.embedding_length) + '-max_question_len=' + str(args.max_question_len))
        model = lstm.train(save=True, epochs=args.nr_epochs, batch_size=args.batch_size)
    else:
        model = load_model(model_folder + args.model_name)
    lstm.evaluate(model)

def plot_statistics(statistics):
    return

def visualize_errors(args):
    results_bow = np.load(hyper_parameter_folder + final_model_folder +  'FINAL-MODEL-BOW-acc=0.384199864966 bow-question_maxlen=15-embedd_length=300-visual_model=True.npy')
    results_lstm = np.load('mata', 'incur')

    only_bow = []
    only_lstm = []
    both_correct = []
    both_wrong = []

    for result, index in range(results_lstm):
        result_lstm = results_lstm[index]
        result_bow = results_bow[index]

        if str(result_lstm['answer']) == str(result_lstm['prediction']) and str(result_bow['answer']) != str(result_bow['prediction']): #same question
            to_push = dict()
            to_push['img_id'] = result_lstm['image_id']
            to_push['predicted_lstm'] = result_lstm['prediction']
            to_push['predicted_bow'] = result_bow['[prediction']
            only_lstm.append(to_push)

        if str(result_lstm['answer']) != str(result_lstm['prediction']) and str(result_bow['answer']) == str(result_bow['prediction']): #same question
            to_push = dict()
            to_push['img_id'] = result_lstm['image_id']
            to_push['predicted_lstm'] = result_lstm['prediction']
            to_push['predicted_bow'] = result_bow['[prediction']
            only_bow.append(to_push)

        if str(result_lstm['answer']) == str(result_lstm['prediction']) and str(result_bow['answer']) == str(result_bow['prediction']):  # same question
            to_push = dict()
            to_push['img_id'] = result_lstm['image_id']
            to_push['predicted_lstm'] = result_lstm['prediction']
            to_push['predicted_bow'] = result_bow['[prediction']
            both_correct.append(to_push)


        if str(result_lstm['answer']) != str(result_lstm['prediction']) and str(result_bow['answer']) != str(result_bow['prediction']):  # same question
            to_push = dict()
            to_push['img_id'] = result_lstm['image_id']
            to_push['predicted_lstm'] = result_lstm['prediction']
            to_push['predicted_bow'] = result_bow['[prediction']
            both_wrong.append(to_push)

    print(only_lstm)
    print(only_bow)
    print(both_correct)
    print(both_wrong)
    exit(1)
    #statistics = get_statistics(results)
    print(statistics)
    exit(1)
    plot_statistics(statistics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_question_len" , help="maximum of words of a question", default=15, type=int)
    parser.add_argument("--embedding_length", help="embedding layer output dimension", default=300, type=int)
    parser.add_argument("--max_answers", help="max labels stored for c", default=1000, type=int)
    parser.add_argument("--nr_epochs", help="number of epochs used for training", default=10, type=int)
    parser.add_argument("--batch_size", help="batch size used for training", default=64, type=int)
    parser.add_argument("--model_name", help="name of model to load in /models", default='', type=str)
    parser.add_argument("--number_hidden_units", help="number of hidden units in lstm/rnn", default=256, type=int)
    parser.add_argument("--number_mlp_unts", help="number_mlp_units", default=512, type=int)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--r_dropout", type=float, default=0.2)
    parser.add_argument("--visual_model", type=bool, default=True)
    parser.add_argument("--only_analyze", type=bool, default=False)
    parser.add_argument("--number_stacked_lstms", type=int, default=0)
    parser.add_argument("--model_type", type=str, default="bow")

    args = parser.parse_args()

    if args.only_analyze:
        visualize_errors(args)
    else:
        train_and_evaluate(args)
