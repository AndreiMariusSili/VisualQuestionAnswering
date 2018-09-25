from keras.models import Sequential
from keras.layers import Dense, Reshape, Merge, Dropout, Input
from keras.layers import LSTM as keras_lstm
from keras.layers.embeddings import Embedding
from dictionary import Dictionary
from constants import *
from model_base import ModelBase

class LSTM(ModelBase):
    lstm_hidden_units = None
    dropout = None
    recurrent_dropout = None
    number_stacked_lstms = None
    mlp_hidden_units = None
    adding_mlp = None

    def __init__(self, dictionary : Dictionary, question_maxlen=20, embedding_vector_length=300, visual_model=True, lstm_hidden_units = 512, dropout = 0.2, recurrent_dropout = 0.2, number_stacked_lstms = 0, adding_mlp = 0, number_mlp_units = 1024):
        super(LSTM, self).__init__(dictionary, question_maxlen, embedding_vector_length, visual_model)
        self.lstm_hidden_units = lstm_hidden_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.number_stacked_lstms = number_stacked_lstms
        self.model_name = "lstm-q_len=" + str(question_maxlen) + "-embedd_len=" + str(embedding_vector_length) + "-h_units=" \
                        + str(lstm_hidden_units) +"-dropo=" + str(dropout) + "-r_dr=" + str(recurrent_dropout)  + \
                          "-visual=" + str(visual_model) + "-stacked=" + str(number_stacked_lstms) + "-mlp_units=" + str(number_mlp_units)
        self.model_type = 'lstm'
        self.adding_mlp = adding_mlp
        self.mlp_hidden_units = number_mlp_units

    def build_visual_model(self, X, Y):
        image_model = Sequential()
        image_dimension = self.dictionary.pp_data.calculateImageDimension()
        image_model.add(Reshape((image_dimension,), input_shape = (image_dimension,)))

        language_model = Sequential()
        language_model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.question_maxlen))

        for i in range(self.number_stacked_lstms - 1):
            language_model.add(keras_lstm(self.lstm_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True))

        language_model.add(keras_lstm(self.lstm_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
            return_sequences=False))

        if self.adding_mlp:
            language_model.add(Dense(self.mlp_hidden_units, init='uniform', activation='tanh'))
            language_model.add(Dropout(self.dropout))
       
        model = Sequential()
        model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))
        model.add(Dense(len(self.dictionary.labels2idx), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        return model

    def build_language_model(self, X, Y):
        language_model = Sequential()
        language_model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.question_maxlen))

        for i in range(self.number_stacked_lstms - 1):
            language_model.add(
                keras_lstm(self.lstm_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                           return_sequences=True))

        language_model.add(
            keras_lstm(self.lstm_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                       return_sequences=False))

        if self.adding_mlp:
            language_model.add(Dense(self.mlp_hidden_units, init='uniform', activation='tanh'))
            language_model.add(Dropout(self.dropout))

        language_model.add(Dense(len(self.dictionary.labels2idx), activation='softmax'))
        language_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(language_model.summary())

        return language_model