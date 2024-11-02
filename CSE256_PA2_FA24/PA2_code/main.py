import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import matplotlib.pyplot as plt
import argparse


from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerEncoder
from classifier import Classifier,SpeechClassifier
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


""" Hyperparameters for Language decoder. """
eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


""" Classifier training hyperparameters. """
n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels


def eval_classifier(data_loader, model, loss_fn):
    """ Compute the accuracy of the classifier on the data in data_loader."""
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


def train_classifier(data_loader, model, loss_fn, optimizer):
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


def experiment_classifier(n_epoch, train_loader, test_loader, model, loss_fn, optimizer):
    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []

    model = model.to(device)

    for epoch in range(n_epoch):
        train_accuracy, train_loss = train_classifier(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_classifier(test_loader, model, loss_fn)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)

        print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}, train loss {train_loss:.3f}, dev loss {test_loss:.3f}')

    return all_train_accuracy, all_test_accuracy, all_train_loss, all_test_loss


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def plot_metrics(train_metric, test_metric, metric_name, model_name):
    """Helper function to plot and save training metrics"""
    plt.figure(figsize=(8, 6))
    plt.plot(train_metric, label='train')
    plt.plot(test_metric, label='test')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} for training and testing ({model_name})')
    plt.legend()
    plt.grid()
    
    # Save the figure
    filename = f'../results/{model_name}_{metric_name.lower()}.png'
    plt.savefig(filename)
    print(f"\n\n{metric_name} plot saved as {filename}")
    plt.close()  # Close the figure to free memory


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='part1 for classification task, part2 for language model')
    # Parse the command-line arguments
    args = parser.parse_args()

    if args.model not in ["part1", "part2", "part3"]:
        raise ValueError(f"Invalid model argument: '{args.model}'. Valid options are: 'part1', 'part2', 'part3'")

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    if args.model == "part1":
        # Load dataset
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=False)

        # Create a classifier
        vocab_size = len(train_CLS_dataset.tokenizer.itos)
        encoder = TransformerEncoder(seq_lenth=block_size, vocab_size=vocab_size, d_model=n_embd, d_ff=4*n_embd, num_layers=n_layer, num_heads=n_head)
        classifier = Classifier(d_model=n_embd, d_hidden=n_hidden, d_out=n_output)
        speech_classifier = SpeechClassifier(encoder, classifier)

        # Sanity check
        # encoder = encoder.to("cpu")
        ultil = Utilities(tokenizer, encoder)
        sentence = texts[0]
        ultil.sanity_check(sentence, block_size)

        # Run experiment
        print(f"Device: {device}")
        
        optimizer = torch.optim.Adam(speech_classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        train_acc, test_acc, train_loss, test_loss = experiment_classifier(epochs_CLS, train_CLS_loader, test_CLS_loader, speech_classifier, loss_fn=criterion, optimizer=optimizer)


        # Plot the results
        plot_metrics(train_acc, test_acc, "Accuracy", "part1")
        plot_metrics(train_loss, test_loss, "Loss", "part1")


    elif args.model == "part2":
        print("Building part2 ...")

        # inputfile = "speechesdataset/train_LM.txt"
        # with open(inputfile, 'r', encoding='utf-8') as f:
        #     lmtrainText = f.read()
        # train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        # train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)


        # # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        # for i, (xb, yb) in enumerate(train_LM_loader):
        #     if i >= max_iters:
        #         break
        #     xb, yb = xb.to(device), yb.to(device)
        #     # LM training code here

    elif args.model == "part3":
        print("Building part3 ...")



    



if __name__ == "__main__":
    main()
