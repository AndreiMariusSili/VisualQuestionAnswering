import argparse

import pickle
from types import SimpleNamespace

from bow import BOW
from data.loader import VQALoader
from lstm import LSTM
from misc.constants import EMBEDDING_FOLDER, EMBEDDING_FILE
from serializer import Serializer
from trainer import Trainer

QUESTION_MAX_LEN = 20
ANSWER_MAX_LEN = 1
IMG_FEATURES_LEN = 2048
BATCH_SIZE = 32
MODELS_PATH = './models/'


def get_embeddings(embedding_size):
    assert embedding_size in [50, 100, 200, 300]
    embedding_file = EMBEDDING_FOLDER + EMBEDDING_FILE.format(embedding_size)
    with open(embedding_file, 'rb') as f:
        pretrained_embeddings = pickle.load(f)
    return pretrained_embeddings


def init_train_save(config_model, config_trainer, train_data, val_data):
    serializer = Serializer(MODELS_PATH)

    vocab_size = len(train_data.word2idx)
    output_size = len(train_data.label2idx)

    if config_model.use_pretrained_embeddings:
        pretrained_embeddings = get_embeddings(config_model.embedding_size)
    else:
        pretrained_embeddings = None

    if config_model.model_type == 'lstm':
        model = LSTM(vocab_size=vocab_size, output_size=output_size, embedding_size=config_model.embedding_size,
                     hidden_size=config_model.hidden_units, img_feature_size=config_model.img_features_len,
                     number_stacked_lstms=config_model.number_stacked_lstms,
                     visual_model=config_model.visual_model, visual_features_location=config_model.visual_features_location,
                     pretrained_embeddings=pretrained_embeddings)
    elif config_model.model_type == 'bow':
        model = BOW(vocab_size=vocab_size, output_size=output_size, embedding_size=config_model.embedding_size,
                    question_len=config_model.question_max_len, img_feature_size=config_model.img_features_len, visual_model=config_model.visual_model,
                    pretrained_embeddings=pretrained_embeddings)
    else:
        raise ValueError("No such model")

    trainer = Trainer(train_data, val_data, serializer)

    #MODELNAME is set HERE
    return trainer.train(model, config_trainer.epochs, config_trainer.lr, config_trainer.verbose, config_trainer.save,
                  modelname=config_model.model_type, config_trainer=config_trainer, config_model=config_model)

def load_evaluate(modelname):
    serializer = Serializer(MODELS_PATH)
    model, config_model, config_trainer, aux = serializer.load(modelname)

    train_data = VQALoader("train", True, True, BATCH_SIZE, fix_q_len=QUESTION_MAX_LEN, fix_a_len=ANSWER_MAX_LEN)
    val_data = VQALoader("val", True, True, BATCH_SIZE, 0, fix_q_len=QUESTION_MAX_LEN, fix_a_len=ANSWER_MAX_LEN)

    trainer = Trainer(train_data, val_data, serializer)
    model = model.to(trainer.device)
    from torch.nn import CrossEntropyLoss
    trainer.validate(model, criterion=CrossEntropyLoss().to(trainer.device), verbose=True)


if __name__ == "__main__":
    # I use this just so I can do config.property. If I just define config as dict, I need to do config["property"]
    config_model = SimpleNamespace()
    config_trainer = SimpleNamespace()

    config_model.model_type = 'lstm'
    config_model.embedding_size = 300
    config_model.hidden_units = 256
    config_model.number_stacked_lstms = 1
    config_model.visual_model=True
    config_model.visual_features_location = ['lstm_context', 'lstm_output']  # ['lstm_context', 'lstm_output', 'lstm_input']
    config_model.use_pretrained_embeddings = True
    config_model.embedding_size = 300
    config_model.img_features_len = IMG_FEATURES_LEN
    config_model.question_max_len = QUESTION_MAX_LEN

    config_trainer.save=True
    config_trainer.verbose = True
    config_trainer.epochs = 2
    config_trainer.lr = 0.001

    train_data = VQALoader("train", True, True, BATCH_SIZE, fix_q_len=QUESTION_MAX_LEN, fix_a_len=ANSWER_MAX_LEN)
    val_data = VQALoader("val", True, True, BATCH_SIZE, num_workers=0, fix_q_len=QUESTION_MAX_LEN, fix_a_len=ANSWER_MAX_LEN)

    init_train_save(config_model, config_trainer, train_data, val_data)

    #load_evaluate('lstm')