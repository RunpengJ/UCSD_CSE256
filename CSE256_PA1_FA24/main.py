# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, DAN


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        if not isinstance(model, DAN):
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        if not isinstance(model, DAN):
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)


        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy, all_train_loss, all_test_loss


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    # Parse the command-line arguments
    args = parser.parse_args()

    if args.model == "BOW":
        # Load dataset
        start_time = time.time()
        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy, nn2_train_loss, nn2_test_loss = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy, nn3_train_loss, nn3_test_loss = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        # Start the timer to measure load time
        start_time = time.time()

        # Load pretrained embeddings
        word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")

        # Load training and development datasets
        train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)

        # Create data loaders with the custom collate function
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=train_data.collate_fn)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False, collate_fn=dev_data.collate_fn)

        print(f"Data loaded in : {time.time() - start_time} seconds")

        # Train and evaluate pretrained DAN
        start_time = time.time()
        print("Pretrained DAN :")
        pre_DAN_train_accuracy, pre_DAN_test_accuracy, pre_DAN_train_loss, pre_DAN_test_loss = experiment(DAN(embed_size=300, hidden_size=100, word_embed=word_embeddings, frozen=True), train_loader, test_loader)
        print(f"Finished training in : {time.time() - start_time} seconds")

        ###### Randomly initialized embeddings ######
        print("DAN")
        start_time = time.time()
        DAN_train_accuracy, DAN_test_accuracy, rand_DAN_train_loss, rand_DAN_test_loss = experiment(DAN(embed_size=300, hidden_size=100, vocab_size=word_embeddings.word_indexer.__len__()), train_loader, test_loader)
        print(f"Finished training in : {time.time() - start_time} seconds")

        # Plot the training and testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(pre_DAN_train_accuracy, label='Pretrained DAN Training Acc')
        plt.plot(pre_DAN_test_accuracy, label='Pretrained DAN Testing Acc')
        plt.plot(DAN_train_accuracy, label='DAN Training Acc')
        plt.plot(DAN_test_accuracy, label='DAN Testing Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy for training and testing')
        plt.legend()
        plt.grid()

        # Save the accuracy figure
        dan_accuracy_file = 'dan_accuracy.png'
        plt.savefig(dan_accuracy_file)
        print(f"DAN accuracy plot saved as {dan_accuracy_file}\n\n")

        # Plot the training and testing loss
        plt.figure(figsize=(8, 6))
        plt.plot(pre_DAN_train_loss, label='Pretrained DAN Training Loss')
        plt.plot(pre_DAN_test_loss, label='Pretrained DAN Testing Loss')
        plt.plot(rand_DAN_train_loss, label='DAN Training Loss')
        plt.plot(rand_DAN_test_loss, label='DAN Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss for training and testing')
        plt.legend()
        plt.grid()

        # Save the loss figure
        dan_loss_file = 'dan_loss.png'
        plt.savefig(dan_loss_file)
        print(f"DAN loss plot saved as {dan_loss_file}\n\n")

if __name__ == "__main__":
    main()