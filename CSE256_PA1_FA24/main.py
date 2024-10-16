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
from byte_pair_encoding import Byte_Pair_Encoding


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer, device):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

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
def eval_epoch(data_loader, model, loss_fn, device):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
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
def experiment(model, train_loader, test_loader, device):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []

    model = model.to(device)

    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, device)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
        nn2_train_accuracy, nn2_test_accuracy, nn2_train_loss, nn2_test_loss = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader, device)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy, nn3_train_loss, nn3_test_loss = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader, device)

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
        training_accuracy_file = 'results/bow_train_accuracy.png'
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
        testing_accuracy_file = 'results/bow_dev_accuracy.png'
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
        print("Pretrained DAN :")
        start_time = time.time()
        pretrain_train_accuracy, pretrain_test_accuracy, pretrain_train_loss, pretrain_test_loss = experiment(
            DAN(embed_size=300, hidden_size=100, word_embed=word_embeddings, frozen=True), train_loader, test_loader, device)
        print(f"Finished training in : {time.time() - start_time} seconds")

        # Randomly initialized embeddings 
        print("DAN")
        start_time = time.time()
        rand_init_train_accuracy, rand_init_test_accuracy, rand_init_train_loss, rand_init_test_loss = experiment(
            DAN(embed_size=50, hidden_size=100, vocab_size=word_embeddings.word_indexer.__len__()), train_loader, test_loader, device)
        print(f"Finished training in : {time.time() - start_time} seconds")

        # Plot the accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(pretrain_train_accuracy, label='Pretrained DAN Training')
        plt.plot(pretrain_test_accuracy, label='Pretrained DAN Testing')
        plt.plot(rand_init_train_accuracy, label='DAN Training')
        plt.plot(rand_init_test_accuracy, label='DAN Testing')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy for training and testing')
        plt.legend()
        plt.grid()

        # Save the accuracy figure
        dan_accuracy_file = 'results/dan_accuracy.png'
        plt.savefig(dan_accuracy_file)
        print(f"Accuracy plot saved as {dan_accuracy_file}\n\n")

        # Plot the loss
        plt.figure(figsize=(8, 6))
        plt.plot(pretrain_train_loss, label='Pretrained DAN Training')
        plt.plot(pretrain_test_loss, label='Pretrained DAN Testing')
        plt.plot(rand_init_train_loss, label='DAN Training')
        plt.plot(rand_init_test_loss, label='DAN Testing')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss for training and testing')
        plt.legend()
        plt.grid()

        # Save the loss figure
        dan_loss_file = 'results/dan_loss.png'
        plt.savefig(dan_loss_file)
        print(f"Loss plot saved as {dan_loss_file}\n\n")

    elif args.model == "SUBWORDDAN":
        K = 10000

        print('##### Loading dataset #####')
        start_time = time.time()
        train_data = Byte_Pair_Encoding("data/train.txt", K)
        test_data = Byte_Pair_Encoding("data/dev.txt", K, indexer=train_data.indexer, merge_ops=train_data.merge_ops)
        print(f"##### Finish BPE in {time.time() - start_time} secounds ##### ")
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=train_data.collate_fn)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=True, collate_fn=test_data.collate_fn)
        
        # Train model
        print("##### Training SUBWORDDAN #####")
        start_time = time.time()
        subword_train_accuracy, subword_test_accuracy, subword_train_loss, subword_test_loss = experiment(
            DAN(embed_size=300, hidden_size=100, vocab_size=K), train_loader, test_loader, device)
        print(f"Finished training in : {time.time() - start_time} seconds")

        # Plot the training and testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(subword_train_accuracy, label='Training')
        plt.plot(subword_test_accuracy, label='Testing')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy for training and testing')
        plt.legend()
        plt.grid()

        # Save the accuracy figure
        subworddan_accuracy_file = 'results/subworddan_accuracy.png'
        plt.savefig(subworddan_accuracy_file)
        print(f"Subword DAN accuracy plot saved as {subworddan_accuracy_file}\n\n")

        # Plot the training and testing loss
        plt.figure(figsize=(8, 6))
        plt.plot(subword_train_loss, label='Training')
        plt.plot(subword_test_loss, label='Testing')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss for training and testing')
        plt.legend()
        plt.grid()

        # Save the accuracy figure
        subworddan_loss_file = 'results/subworddan_loss.png'
        plt.savefig(subworddan_loss_file)
        print(f"Subword DAN loss plot saved as {subworddan_loss_file}\n\n")


if __name__ == "__main__":
    main()
