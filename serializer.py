import os

import pickle
import torch


#TODO change in order to save and load just from the parameters
class Serializer:
    def __init__(self, models_path):
        self.models_path = models_path

    def save(self, model, config_train, config_model, modelname, aux=None):
        path = self.models_path + modelname
        suffix = ''
        count = 0
        while os.path.exists(path+suffix):
            count += 1
            suffix = str(count)
        modelname = modelname + suffix
        path = path+suffix+'/'
        os.mkdir(path)

        #save model
        torch.save(model, path+'model')
        #save parameters as objects
        with open(path+'config_model.pkl', 'wb+') as fd:
            pickle.dump(config_model, fd)
        with open(path + 'config_train.pkl', 'wb+') as fd:
            pickle.dump(config_train, fd)
        if aux:
            with open(path + 'aux.pkl', 'wb+') as fd:
                pickle.dump(aux, fd)
        #save parameters as text
        with open(path + 'config_model.txt', 'w+') as fd:
            fd.write(str(config_model))
        with open(path + 'config_train.txt', 'w+') as fd:
            fd.write(str(config_train))
        if aux:
            with open(path + 'aux.txt', 'w+') as fd:
                fd.write(str(aux))

        return modelname

    def load(self, modelname):
        path = self.models_path + '/' + modelname + '/'

        #load model
        model = torch.load(path + 'model')

        # load other parameters
        with open(path + 'config_model.pkl', 'rb') as fd:
            config_model = pickle.load(fd)
        with open(path + 'config_train.pkl', 'rb') as fd:
            config_train = pickle.load(fd)
        if os.path.isfile(path + 'aux.pkl'):
            with open(path + 'aux.pkl', 'rb') as fd:
                aux = pickle.load(fd)
        else:
            aux = None
        return model, config_model, config_train, aux