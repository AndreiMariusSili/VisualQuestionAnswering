from constants import *
from utils import *
from dictionary import Dictionary
from keras.models import Sequential


class ModelBase:
    question_maxlen = None
    top_words = None
    embedding_vector_length = None

    dictionary = dict()
    visual_model = None
    model_name = None
    model_type = None

    training_history = None

    def __init__(self, dictionary : Dictionary, question_maxlen = 20, embedding_vector_length = 300, visual_model=True):
        self.dictionary = dictionary
        self.question_maxlen = question_maxlen
        self.embedding_vector_length = embedding_vector_length
        self.visual_model = visual_model
        self.top_words = len(self.dictionary.word2idx) + 1

    def build_visual_model(self, X, Y):
        raise NotImplementedError

    def build_language_model(self, X, Y):
        raise NotImplementedError

    def get_model(self, X, Y) -> Sequential:
        if self.visual_model:
            return self.build_visual_model(X,Y)
        else:
            return self.build_language_model(X,Y)

    def train(self, train_data_file=train_data_write_file, save=False, save_name='', epochs=10, batch_size=64, verbose=1):
        save_name = save_name if save_name else self.model_name

        X, X_features, Y, _, _ = prepare_data(data_folder + train_data_file, self.dictionary, self.question_maxlen)

        model = self.get_model(X, Y)

        if self.visual_model:
            history = model.fit([X, X_features], Y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        else:
            history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose)

        self.training_history = history

        #np.save(hyper_parameter_folder + final_model_folder + 'FINAL-LSTM-HISTORY-TRAINING-ACTUAL-VISUAL', history)

        if save:
            self.save_model(model, save_name)

        return model

    def evaluate(self, model : Sequential, test_data_file=test_data_write_file, visualize_results=True):
        X, X_features, Y, answers, X_question_ids = prepare_data(data_folder + test_data_file, self.dictionary, self.question_maxlen)

        if self.visual_model:
            scores, acc = model.evaluate([X, X_features], Y)
        else:
            scores, acc = model.evaluate(X, Y)

        predictions = None
        if self.visual_model:
            predictions = model.predict([X, X_features])
        else:
            predictions = model.predict(X)

        if visualize_results:
            analyse_results(X.tolist(), predictions, answers, X_question_ids, model, self.dictionary, acc, self.model_name, self.model_type)

        return acc

    def save_model(self, model : Sequential, model_name):
        model.save(model_folder + model_name, overwrite=True)