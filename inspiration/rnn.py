from keras.models import Sequential
from keras.layers import Dense, Reshape, Merge, Dropout, Input, SimpleRNN
from keras.layers.embeddings import Embedding
from dictionary import Dictionary
from constants import *
from model_base import ModelBase

class RNN(ModelBase):
    rnn_hiden_units = None
    deeper_lstm = None
    dropout = None
    recurrent_dropout = None

    def __init__(self, dictionary : Dictionary, question_maxlen=20, embedding_vector_length=300, visual_model=True, rnn_hidden_units = 512, dropout = 0.2, recurrent_dropout = 0.2, deeper_lstm = False):
        super(RNN, self).__init__(dictionary, question_maxlen, embedding_vector_length, visual_model)
        self.rnn_hiden_units = rnn_hidden_units
        self.deeper_lstm = deeper_lstm
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.model_name = "rnn-question_maxlen=" + str(question_maxlen) + "-embedd_length=" + str(embedding_vector_length) + "-lstm_hidden_units=" \
                        + str(rnn_hidden_units) +"-dropout=" + str(dropout) + "-recurrent_dropout=" + str(recurrent_dropout) + "-deeper_lstm=" + str(deeper_lstm) + \
                          "-visual_model=" + str(visual_model)
        self.model_type = 'rnn'


    def build_visual_model(self, X, Y):
        image_model = Sequential()
        image_dimension = self.dictionary.pp_data.calculateImageDimension()
        image_model.add(Reshape((image_dimension,), input_shape = (image_dimension,)))

        language_model = Sequential()
        language_model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.question_maxlen))
        language_model.add(SimpleRNN(self.rnn_hiden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)) # discover these parameters?
        model = Sequential()
        model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))
        model.add(Dense(self.dictionary.max_labels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        return model

    def build_language_model(self, X, Y):
        model = Sequential()
        model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.question_maxlen))
        model.add(SimpleRNN(self.rnn_hiden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout))
        model.add(Dense(self.dictionary.max_labels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        return model