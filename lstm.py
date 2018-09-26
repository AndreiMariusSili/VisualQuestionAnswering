import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM Model PyTorch implementation
    """

    def __init__(self, dictionary_size, output_size, embedding_size, hidden_size, question_len=25,
                 img_feature_size=2048, number_stacked_lstms=0, visual_model=False,  attention=False):
        super(LSTM, self).__init__()

        self.attention = attention
        self.visual_model = visual_model

        self.embedding = nn.Embedding(dictionary_size, embedding_size)

        if visual_model:
            lstm_input_size = embedding_size * question_len + img_feature_size
        else:
            lstm_input_size = embedding_size * question_len
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=number_stacked_lstms)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, sentence, image_features=None):
        hidden = self.embedding(sentence).view(sentence.shape[0], -1)
        if self.visual_model:
            hidden = torch.cat((hidden, image_features), dim=1)
        lstm_out, _ = self.lstm(hidden)
        return self.linear(lstm_out)
