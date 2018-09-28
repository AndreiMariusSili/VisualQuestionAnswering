import torch
from torch.autograd import Variable
import torch.nn.functional as F

from bow import BOW
from data.loader import VQALoader
from lstm import LSTM

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Trainer:

    @staticmethod
    def init_train_save(embedding_size=300, epochs=10, lr=0.001, question_maxlen=20, visual_model=True, hidden_units = 256, dropout = 0.2, number_stacked_lstms=1, adding_mlp = 0, number_mlp_units = 1024,
                        save=False, modelname='model', verbose=True, model_type=None, image_features=None, max_a_len=1):
        """
        return: model, lossv, accv, modelname
        """
        train_data = VQALoader("train", True, True, 32, fix_q_len=question_maxlen, fix_a_len=max_a_len)
        val_data = VQALoader("val", True, True, 32, 0, fix_q_len=question_maxlen, fix_a_len=max_a_len)

        vocab_size = len(train_data.word2idx)
        output_size = len(train_data.label2idx)

        if model_type == 'lstm':
            model = LSTM(vocab_size=vocab_size, output_size=output_size, embedding_size=embedding_size,
                         hidden_size=hidden_units, img_feature_size=2048, number_stacked_lstms=number_stacked_lstms,
                         visual_model=visual_model, attention="PULA")
        elif model_type == 'rnn':
            model = RNN(...)
        elif model_type == 'bow':
            model = BOW(vocab_size=vocab_size, output_size=output_size, embedding_size=embedding_size,
                        question_len=question_maxlen, img_feature_size=2048, visual_model=True)
        else:
            raise ValueError("dafuq man?")

        print(model)

        model.to(DEVICE)

        train_loader = train_data.get()
        validation_loader = val_data.get()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        #model_description = {}

        loss_valid, acc_valid, loss_train = [], [], []

        for epoch in range(1, epochs + 1):
            Trainer.train(model=model, train_loader=train_loader, optimizer=optimizer,
                          epoch=epoch, loss_vector=loss_train, verbose=verbose)
            Trainer.validate(model, validation_loader, loss_valid, acc_valid, verbose=verbose)

        # model_description["lossv"] = [value for value in lossv]
        # model_description["accv"] = [value for value in accv]

        model.cpu()

        if save:
            modelname = Trainer.save(model, 'final-epoch:'+str(len(acc_valid))+'-acc:' + str(acc_valid[-1]) + '-'+modelname)

        return model, loss_valid, acc_valid, modelname

    @staticmethod
    def validate(model, validation_loader, loss_vector, accuracy_vector, verbose=True):
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        val_loss, correct = torch.tensor([0], dtype=torch.float, device=DEVICE), 0
        for batch_idx, (questions, image_fetures, target) in enumerate(validation_loader):
            with torch.no_grad():
                questions, image_fetures, target = Variable(questions.to(DEVICE)), Variable(image_fetures.to(DEVICE)), Variable(target.to(DEVICE))
                output = model(questions, image_fetures)
                loss = criterion(output, target[:, 0])
                val_loss = torch.add(val_loss, loss)

                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target[:, 0].data).sum().cpu()

        val_loss /= len(validation_loader)
        loss_vector.append(val_loss)

        accuracy = float(100. * correct) / len(validation_loader.dataset)

        accuracy_vector.append(accuracy)

        if verbose:
            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                val_loss.item(), correct.cpu(), len(validation_loader.dataset), accuracy))

    @staticmethod
    def train(model, train_loader, optimizer, epoch, loss_vector, log_interval=100, verbose=True):
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        for batch_idx, (questions, image_fetures, target) in enumerate(train_loader):
            questions, image_fetures, target = Variable(questions.to(DEVICE)), Variable(image_fetures.to(DEVICE)), Variable(target.to(DEVICE))
            optimizer.zero_grad()
            output = model(questions, image_fetures)

            loss = criterion(output, target[:,0])
            loss.backward()
            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(questions), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        loss_vector.append(loss)

    @staticmethod
    def save(model, path):
        torch.save(model, path)
        return

    @staticmethod
    def load(path):
        model = torch.load(path)
        return model


Trainer.init_train_save(model_type='bow')