import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class LSTM(torch.nn.Module):
    """
    LSTM Model PyTorch implementation
    """

    def __init__(self, vocab_size, output_size, embedding_size, hidden_size, img_feature_size=2048,
                 number_stacked_lstms=1, visual_model=False, pretrained_embeddings=None, embedding_trainable=True,
                 visual_features_location=None, dropout=0, full_size_visual_features=False):
        """

        :param vocab_size:
        :param output_size:
        :param embedding_size:
        :param hidden_size:
        :param img_feature_size:
        :param number_stacked_lstms:
        :param visual_model:
        :param visual_features_location: list[str];
            'lstm_context', 'lstm_output', 'lstm_input'
        """
        super(LSTM, self).__init__()

        for value in visual_features_location:
            assert value in ['lstm_context', 'lstm_output', 'lstm_input']

        self.visual_features_location = visual_features_location
        self.visual_model = visual_model
        self.hidden_size = hidden_size if full_size_visual_features is False else img_feature_size
        self.full_size_visual_features = full_size_visual_features
        self.number_stacked_lstms = number_stacked_lstms

        self.embedding = torch.torch.nn.Embedding(vocab_size, embedding_size)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if not embedding_trainable:
                self.embedding.weight.requires_grad = False

        if self.visual_model and 'lstm_input' in self.visual_features_location:
            self.lstm = torch.nn.LSTM(input_size=embedding_size + img_feature_size, hidden_size=self.hidden_size,
                                      num_layers=number_stacked_lstms,
                                      batch_first=True, dropout=dropout)
        else:
            self.lstm = torch.nn.LSTM(input_size=embedding_size, hidden_size=self.hidden_size,
                                      num_layers=number_stacked_lstms,
                                      batch_first=True, dropout=dropout)
        if self.visual_model and 'lstm_context' in self.visual_features_location and not full_size_visual_features:
            self.visual_features_to_layer_features = torch.nn.Linear(img_feature_size, img_feature_size)
            self.layer_features_to_hidden = torch.nn.Linear(img_feature_size, self.hidden_size)
        if self.visual_model and 'lstm_output' in self.visual_features_location:
            self.lstm_to_linear = torch.nn.Linear(self.hidden_size + img_feature_size, self.hidden_size + img_feature_size)
            self.output_linear = torch.nn.Linear(self.hidden_size + img_feature_size, output_size)
        else:
            self.output_linear = torch.nn.Linear(self.hidden_size, img_feature_size)

    def forward(self, sentence, image_features=None):
        input = self.embedding(sentence)

        if self.visual_model and 'lstm_input' in self.visual_features_location:
            # concat image to each input
            input = torch.cat((input, image_features.unsqueeze(1).repeat(1, input.shape[1], 1)), 2)

        if self.visual_model and self.full_size_visual_features:
            c_0 = image_features.unsqueeze(0).repeat(self.number_stacked_lstms, 1, 1)
        elif self.visual_model and 'lstm_context' in self.visual_features_location:
            layer_features = self.visual_features_to_layer_features(image_features)
            c_0 = self.layer_features_to_hidden(layer_features).unsqueeze(0).repeat(self.number_stacked_lstms, 1, 1)
        else:
            c_0 = torch.zeros(self.number_stacked_lstms, sentence.shape[0], self.hidden_size, device=DEVICE)

        h_0 = torch.zeros(self.number_stacked_lstms, sentence.shape[0], self.hidden_size, device=DEVICE)

        lstm_out, _ = self.lstm(input, (h_0, c_0))
        if self.visual_model and 'lstm_output' in self.visual_features_location:
            # concatenate features to the last state of the lstm cell
            last_lstm_state = lstm_out[:, -1]
            lstm_state_image_features = torch.cat((last_lstm_state, image_features), 1)
            lstm_to_linear = self.lstm_to_linear(lstm_state_image_features)
            output = self.output_linear(lstm_to_linear)
        else:
            # only take the state of the last lstm cell to predict the output
            last_lstm_state = lstm_out[:, -1]
            output = self.output_linear(last_lstm_state)
        return output
