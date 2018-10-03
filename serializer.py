from misc.constants import *
import pickle
import torch
import os


# TODO: change in order to save and load just from the parameters
class Serializer:
    def __init__(self):
        pass

    def save(self, model, config_train, config_model, model_name, aux=None):
        path = os.path.join(MODELS_FOLDER, model_name)
        suffix = ''
        count = 0
        while os.path.exists(path + suffix):
            count += 1
            suffix = str(count)
        model_name = model_name + suffix
        path = path + suffix + '/'
        os.mkdir(path)

        # save model
        torch.save(model, path + 'model')
        # save parameters as objects
        with open(path + 'config_model.pkl', 'wb+') as fd:
            pickle.dump(config_model, fd)
        with open(path + 'config_train.pkl', 'wb+') as fd:
            pickle.dump(config_train, fd)
        if aux:
            with open(path + 'aux.pkl', 'wb+') as fd:
                pickle.dump(aux, fd)
        # save parameters as text
        with open(path + 'config_model.txt', 'w+') as fd:
            fd.write(str(config_model))
        with open(path + 'config_train.txt', 'w+') as fd:
            fd.write(str(config_train))
        if aux:
            with open(path + 'aux.txt', 'w+') as fd:
                fd.write(str(aux))

        return model_name

    def load(self, model_name):
        path = os.path.join(MODELS_FOLDER, model_name)

        # load model
        model = torch.load(os.path.join(path, 'model'))

        # load other parameters
        with open(os.path.join(path, 'config_model.pkl'), 'rb') as fd:
            config_model = pickle.load(fd)
        with open(os.path.join(path, 'config_train.pkl'), 'rb') as fd:
            config_train = pickle.load(fd)
        if os.path.isfile(path + 'aux.pkl'):
            with open(os.path.join(path, 'aux.pkl'), 'rb') as fd:
                aux = pickle.load(fd)
        else:
            aux = None
        return model, config_model, config_train, aux
