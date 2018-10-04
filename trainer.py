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
        self.valid_loader = data.DataLoader(valid_data, batch_size=1, shuffle=True,
                                            num_workers=0, collate_fn=valid_data.collate_fn)

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
        correct = 0
        loss_vector, accuracy_vector = [], []
        loss_sum = 0
        with torch.no_grad():
            #for idx in range(len(self.valid_data)):
                # question, image_features, answer = self.valid_data[idx]
            for batch_idx, (question, image_features, answer) in enumerate(self.valid_loader):
                question = torch.tensor(question, dtype=torch.long, device=self.device)  # batch dim
                image_features = torch.tensor(image_features, dtype=torch.float, device=self.device)
                answer = torch.tensor(answer, dtype=torch.long, device=self.device).reshape(answer.shape[0])
                output = model(question, image_features)
                loss = criterion(output, answer)
                loss_sum += loss.item()
                pred = output.data.argmax(1)  # get the index of the max log-probability
                correct += pred.eq(answer).sum().cpu()

                if verbose and batch_idx < 10:
                    print(self.valid_data.convert_question_to_string(question.squeeze().cpu().numpy().tolist()),
                          end=" ")
                    print(("Truth", self.valid_data.convert_answer_to_string([answer.cpu().item()])), end=" ")
                    print(("Prediction", self.valid_data.convert_answer_to_string([pred.cpu().item()])))

        avg_loss = loss_sum / len(self.valid_data)
        loss_vector.append(avg_loss)
        accuracy = float(100. * correct) / len(self.valid_data)
        accuracy_vector.append(accuracy)

        # Print accuracy
        print('Validation set: [Average loss: {:.4f}]\t[Accuracy: {}/{}\t({:.2f}%)]'.format(
            avg_loss, correct.cpu(), len(self.valid_data), accuracy))

        return loss_vector, accuracy_vector

    def _train_epoch(self, model, optimizer, criterion, epoch, log_interval=100, verbose=True):
        model.train()
        loss_vector = []
        for batch_idx, (questions, image_features, target) in enumerate(self.train_loader):
            questions, image_features, target = Variable(questions.to(self.device)), Variable(
                image_features.to(self.device)), Variable(target.to(self.device))
            optimizer.zero_grad()
            output = model(questions, image_features)

            loss = criterion(output, target.squeeze())
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                preds = output.argmax(-1)
                correct = torch.sum(preds == target.squeeze()).cpu().item()
                acc = float(correct) / BATCH_SIZE
                print('Train Epoch: {} [{}/{}\t({:.0f}%)]\t[Loss: {:.6f}]\t[Acc: {}/{}\t({:.2f}%)]'
                      .format(epoch + 1, (batch_idx + 1) * len(questions), len(self.train_data),
                              100 * (batch_idx + 1) / len(self.train_loader), loss.item(), correct, BATCH_SIZE,
                              100 * acc))

            loss_vector.append(loss.item())

        return loss_vector
