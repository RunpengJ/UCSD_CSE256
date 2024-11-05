import torch
import torch.nn as nn
import torch.nn.functional as F
from feed_forward import FeedForward


class Classifier(nn.Module):
    def __init__(self, d_model, d_hidden, d_out):
        super().__init__()

        self.ff = FeedForward(d_model=d_model, d_ff=d_hidden, d_out=d_out, activate_fn="relu")


    def forward(self, x):
        out = torch.mean(x, dim=1)
        out = self.ff(out)

        return out
    

class SpeechClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier


    def forward(self, x):
        out, att_maps = self.encoder(x)
        out = self.classifier(out)

        return out, att_maps
    

def eval_classifier(data_loader, model, loss_fn, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_train_loss = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)

            logits, _ = model(X)
            loss = loss_fn(logits, Y)
            _, predicted = torch.max(logits, 1)
            
            total_train_loss += loss.item()
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)

        accuracy = (100 * total_correct / total_samples)
        train_loss = total_train_loss / total_samples
        return accuracy, train_loss


def train_classifier(data_loader, model, loss_fn, optimizer, device):
    model.train()
    total_correct = 0
    total_samples = 0
    total_train_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)

        logits, _ = model(X)
        loss = loss_fn(logits, Y)
        _, predicted = torch.max(logits, 1)

        total_train_loss += loss.item()
        total_correct += (predicted == Y).sum().item()
        total_samples += Y.size(0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = (100 * total_correct / total_samples)
    train_loss = total_train_loss / total_samples
    return accuracy, train_loss


def experiment_classifier(n_epoch, train_loader, test_loader, model, loss_fn, optimizer, device):
    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []

    model = model.to(device)

    for epoch in range(n_epoch):
        train_accuracy, train_loss = train_classifier(train_loader, model, loss_fn, optimizer, device)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_classifier(test_loader, model, loss_fn, device)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)

        print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}, train loss {train_loss:.3f}, dev loss {test_loss:.3f}')

    return all_train_accuracy, all_test_accuracy, all_train_loss, all_test_loss