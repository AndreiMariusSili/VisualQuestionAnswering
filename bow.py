import torch
import torch.nn as nn


class BOW(nn.Module):
    """
    BOW Model PyTorch implementation
    """

    def __init__(self, dictionary_size, output_size, embedding_size, question_len=25,
                 img_feature_size=2048, visual_model=False):
        super(BOW, self).__init__()

        self.visual_model = visual_model

        self.embedding = nn.Embedding(dictionary_size, embedding_size)

        if visual_model:
            linear_input_size = embedding_size * question_len + img_feature_size
        else:
            linear_input_size = embedding_size * question_len
        self.linear = nn.Linear(linear_input_size, output_size)

    def forward(self, sentence, image_features=None):
        hidden = self.embedding(sentence).view(sentence.shape[0], -1)
        if self.visual_model:
            hidden = torch.cat((hidden, image_features), dim=1)
        return self.linear(hidden)
