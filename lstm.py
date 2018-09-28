import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class LSTM(nn.Module):
    """
    LSTM Model PyTorch implementation
    """

    def __init__(self, vocab_size, output_size, embedding_size, hidden_size,
                 img_feature_size=2048, number_stacked_lstms=1, visual_model=False, attention=False):
        super(LSTM, self).__init__()

        self.attention = attention
        self.visual_model = visual_model
        self.hidden_size = hidden_size
        self.number_stacked_lstms = number_stacked_lstms

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.layer_features_to_hidden = nn.Linear(img_feature_size, hidden_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=number_stacked_lstms, batch_first=True)

        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, sentence, image_features=None):
        if self.visual_model:
            c_0 = self.layer_features_to_hidden(image_features).unsqueeze(0)#repeat(self.number_stacked_lstms)
        else:
            c_0 = torch.zeros(self.number_stacked_lstms, sentence.shape[0], self.hidden_size, device=DEVICE)

        h_0 = torch.zeros(self.number_stacked_lstms, sentence.shape[0], self.hidden_size, device=DEVICE)

        input = self.embedding(sentence)
        lstm_out, _ = self.lstm(input, (h_0, c_0))
        return self.output_linear(lstm_out[:,-1])
