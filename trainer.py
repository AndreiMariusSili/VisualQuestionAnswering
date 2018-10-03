import torch
from torch.autograd import Variable

#TODO save on highest accuracy

class Trainer:

    def __init__(self, train_data, valid_data, serializer):
        self.train_data = train_data
        self.valid_data = valid_data
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.serializer = serializer

    def train(self, model, epochs, lr=0.001, verbose=True, save=False, modelname=None, config_trainer=None, config_model=None):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        loss_valid, acc_valid, loss_train = [], [], []

        #train criterion ignores <unk>
        criterion_train = torch.nn.CrossEntropyLoss(ignore_index=len(self.train_data.idx2label)-1).to(self.device)
        criterion_val = torch.nn.CrossEntropyLoss().to(self.device)

        model = model.to(self.device)

        for epoch in range(0, epochs):
            epoch_loss_train = self._train_epoch(model=model, optimizer=optimizer,
                                                criterion=criterion_train,
                                                epoch=epoch, verbose=verbose)
            loss_train.extend(epoch_loss_train)

            epoch_loss_valid, epoch_acc_valid = self.validate(model=model, criterion=criterion_val, verbose=verbose)
            loss_valid.extend(epoch_loss_valid)
            acc_valid.extend(epoch_acc_valid)

        model.cpu()

        if save:
            #'final-epoch:' + str(len(acc_valid)) + '-acc:' + str(acc_valid[-1]) + '-' + modelname
            modelname = self.serializer.save(model, config_trainer, config_model, modelname, aux=[['acc_valid'],acc_valid, ['loss_valid'], loss_valid])
            if verbose:
                print("saved model "+modelname)
        else:
            modelname = None

        return model, loss_valid, acc_valid, loss_train, modelname

    def validate(self, model, criterion, verbose=True):
        model.eval()
        validation_loader = self.valid_data.get()
        val_loss, correct = torch.tensor([0], dtype=torch.float, device=self.device), 0
        loss_vector, accuracy_vector = [], []
        for batch_idx, (questions, image_features, target) in enumerate(validation_loader):
            with torch.no_grad():
                questions, image_features, target = Variable(questions.to(self.device)), Variable(image_features.to(self.device)), Variable(target.to(self.device))
                output = model(questions, image_features)
                loss = criterion(output, target[:, 0])
                val_loss = torch.add(val_loss, loss)

                pred = output.data.argmax(1)  # get the index of the max log-probability
                correct += pred.eq(target[:, 0]).sum().cpu()

        val_loss /= len(validation_loader)
        loss_vector.append(val_loss.item())

        accuracy = float(100. * correct) / len(validation_loader.dataset)

        accuracy_vector.append(accuracy)

        if verbose:
            #Print some samples
            for idx_in_batch in range(10):
                print("\nquestion:")
                print([self.valid_data.idx2word[word] for word in questions[idx_in_batch]])
                print("truth:")
                print([self.valid_data.idx2label[target[idx_in_batch, 0]]])
                print("answer:")
                print([self.valid_data.idx2label[pred[idx_in_batch]]])

            # Print accuracy
            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                val_loss.item(), correct.cpu(), len(validation_loader.dataset), accuracy))

        return  loss_vector, accuracy_vector

    def _train_epoch(self, model, optimizer, criterion, epoch, log_interval=100, verbose=True):
        model.train()
        train_loader = self.train_data.get()
        loss_vector = []
        for batch_idx, (questions, image_features, target) in enumerate(train_loader):
            questions, image_features, target = Variable(questions.to(self.device)), Variable(image_features.to(self.device)), Variable(target.to(self.device))
            optimizer.zero_grad()
            output = model(questions, image_features)

            loss = criterion(output, target[:,0])
            loss.backward()
            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(questions), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

            loss_vector.append(loss.item())

        return loss_vector