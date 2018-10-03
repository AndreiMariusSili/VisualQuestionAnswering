import torch
from torch.autograd import Variable
from torch.utils import data

# TODO: save on highest accuracy
from torch.utils.data import DataLoader

from data.dataset import VQADataset
from misc.constants import *
from serializer import Serializer


class Trainer(object):
    serializer: Serializer
    train_loader: DataLoader
    valid_data: VQADataset
    train_data: VQADataset

    def __init__(self, train_data: VQADataset, valid_data: VQADataset, serializer: Serializer):
        self.train_data = train_data
        self.valid_data = valid_data
        self.train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=NUM_WORKERS, collate_fn=train_data.collate_fn)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.serializer = serializer

    def train(self, model, epochs, lr=0.001, verbose=True, save=False, model_name=None, config_trainer=None,
              config_model=None):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        loss_valid, acc_valid, loss_train = [], [], []

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        model = model.to(self.device)

        for epoch in range(0, epochs):
            epoch_loss_train = self._train_epoch(model=model, optimizer=optimizer,
                                                 criterion=criterion,
                                                 epoch=epoch, verbose=verbose)
            loss_train.extend(epoch_loss_train)

            epoch_loss_valid, epoch_acc_valid = self.validate(model=model, criterion=criterion, verbose=verbose)
            loss_valid.extend(epoch_loss_valid)
            acc_valid.extend(epoch_acc_valid)

        model.cpu()

        if save:
            model_name = self.serializer.save(model, config_trainer, config_model, model_name,
                                              aux=[['acc_valid'], acc_valid, ['loss_valid'], loss_valid])
            if verbose:
                print("saved model " + model_name)
        else:
            model_name = None

        return model, loss_valid, acc_valid, loss_train, model_name

    def validate(self, model, criterion, verbose=True):
        model.eval()
        val_loss, correct = torch.tensor([0], dtype=torch.float, device=self.device), 0
        loss_vector, accuracy_vector = [], []
        with torch.no_grad():
            for idx in range(len(self.valid_data)):
                question, image_features, answer = self.valid_data[idx]
                question = torch.tensor(question, dtype=torch.long, device=self.device).unsqueeze(0)  # batch dim
                image_features = torch.tensor(image_features, dtype=torch.float, device=self.device).unsqueeze(0)
                answer = torch.tensor(answer, dtype=torch.long, device=self.device)
                output = model(question, image_features)
                loss = criterion(output, answer)
                val_loss = torch.add(val_loss, loss)

                pred = output.data.argmax(1)  # get the index of the max log-probability
                correct += pred.eq(answer).sum().cpu()

                if verbose and idx < 10:
                    print(self.valid_data.convert_question_to_string(question.squeeze().numpy().tolist()), end=" ")
                    print(("Truth", self.valid_data.convert_answer_to_string([answer.item()])), end=" ")
                    print(("Prediction", self.valid_data.convert_answer_to_string([pred.item()])))

        val_loss /= len(self.valid_data)
        loss_vector.append(val_loss.item())
        accuracy = float(100. * correct) / len(self.valid_data)
        accuracy_vector.append(accuracy)

        # Print accuracy
        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            val_loss.item(), correct.cpu(), len(self.valid_data), accuracy))

        return loss_vector, accuracy_vector

    def _train_epoch(self, model, optimizer, criterion, epoch, log_interval=100, verbose=True):
        model.train()
        loss_vector = []
        for batch_idx, (questions, image_features, target) in enumerate(self.train_loader):
            questions, image_features, target = Variable(questions.to(self.device)), Variable(
                image_features.to(self.device)), Variable(target.to(self.device))
            optimizer.zero_grad()
            output = model(questions, image_features)

            loss = criterion(output, target[:, 0])
            loss.backward()
            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch + 1,
                                                                               (batch_idx + 1) * len(questions),
                                                                               len(self.train_data),
                                                                               100 * batch_idx / len(
                                                                                   self.train_loader),
                                                                               loss.item()))

            loss_vector.append(loss.item())

        return loss_vector
