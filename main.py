import pickle
from types import SimpleNamespace
from bow import BOW
from data.dataset import VQADataset
from lstm import LSTM
from misc.constants import *
from serializer import Serializer
from trainer import Trainer
from torch.nn import CrossEntropyLoss


def get_embeddings(embedding_size):
    assert embedding_size in [50, 100, 200, 300]
    embedding_file = EMBEDDING_FOLDER + EMBEDDING_FILE.format(embedding_size)
    with open(embedding_file, 'rb') as f:
        pretrained_embeddings = pickle.load(f)
    return pretrained_embeddings


def init_train_save(cfg_model, cfg_trainer, train_data, val_data):
    serializer = Serializer()

    vocab_size = len(train_data.word2idx)
    output_size = len(train_data.label2idx)

    if cfg_model.use_pretrained_embeddings:
        pretrained_embeddings = get_embeddings(cfg_model.embedding_size)
    else:
        pretrained_embeddings = None

    if cfg_model.model_type == 'lstm':
        model = LSTM(vocab_size=vocab_size, output_size=output_size, embedding_size=cfg_model.embedding_size,
                     hidden_size=cfg_model.hidden_units, img_feature_size=cfg_model.img_features_len,
                     number_stacked_lstms=cfg_model.number_stacked_lstms,
                     visual_model=cfg_model.visual_model, visual_features_location=cfg_model.visual_features_location,
                     pretrained_embeddings=pretrained_embeddings, dropout=cfg_model.lstm_dropout,
                     full_size_visual_features=cfg_model.full_size_visual_features)
    elif cfg_model.model_type == 'bow':
        model = BOW(vocab_size=vocab_size, output_size=output_size, embedding_size=cfg_model.embedding_size,
                    question_len=cfg_model.question_max_len, img_feature_size=cfg_model.img_features_len,
                    visual_model=cfg_model.visual_model,
                    pretrained_embeddings=pretrained_embeddings)
    else:
        raise ValueError("No such model")

    trainer = Trainer(train_data, val_data, serializer)

    # MODEL NAME is set HERE
    return trainer.train(model, cfg_trainer.epochs, cfg_trainer.lr, cfg_trainer.verbose, cfg_trainer.save,
                         model_name=cfg_model.model_name, config_trainer=cfg_trainer, config_model=cfg_model)


def load_evaluate(model_name, train_data, val_data):
    serializer = Serializer()
    model, cfg_model, cfg_trainer, aux = serializer.load(model_name)

    trainer = Trainer(train_data, val_data, serializer)
    model = model.to(trainer.device)
    trainer.validate(model, criterion=CrossEntropyLoss().to(trainer.device), verbose=True)


if __name__ == "__main__":
    # I use this just so I can do config.property. If I just define config as dict, I need to do config["property"]
    config_model = SimpleNamespace()
    config_trainer = SimpleNamespace()

    config_model.model_type = 'lstm'
    config_model.use_pretrained_embeddings = True
    config_model.embedding_size = 300
    config_model.hidden_units = 256
    config_model.number_stacked_lstms = 2
    config_model.visual_model = True
    config_model.visual_features_location = ['lstm_context',
                                             'lstm_output']  # ['lstm_context', 'lstm_output', 'lstm_input']
    config_model.full_size_visual_features = False  # ignores 'lstm_context' from list above
    config_model.img_features_len = IMG_FEATURES_LEN
    config_model.question_max_len = QUESTION_MAX_LEN
    config_model.lstm_dropout = 0.5

    config_trainer.save = True
    config_trainer.verbose = False
    config_trainer.epochs = 100
    config_trainer.lr = 0.001

    config_model.model_name = 'demo_name_{}'.format(config_model.model_type)

    config_trainer.data = "dummy"  # ["dummy", "full"]
    if config_trainer.data == "dummy":
        train_set = VQADataset("dummy", True, QUESTION_MAX_LEN)
        val_set = VQADataset("dummy", True, QUESTION_MAX_LEN)
    elif config_trainer.data == "full":
        train_set = VQADataset("train", True, QUESTION_MAX_LEN)
        val_set = VQADataset("val", True, QUESTION_MAX_LEN)
        # test_set = VQADataset("test", True, QUESTION_MAX_LEN)
    else:
        raise ValueError('Unknown data flag.')

    action = "train"  # ["train", "eval", "test"]
    if action == "train":
        init_train_save(config_model, config_trainer, train_set, val_set)
    elif action == "eval":
        load_evaluate('lstm', train_set, val_set)
    elif action == "test":
        pass
    else:
        raise ValueError('Unknown action flag.')
