import torch
from torch import nn
from d2l import torch as d2l

from deepl.utils import train_ch3, predict_ch3

if __name__ == "__main__":
    # Define network structure parameters
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs))

    params = [W1, b1, W2, b2]

    # ReLU activation function
    def relu(X):
        return torch.max(X, torch.zeros_like(X))

    # Define neural network
    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)
        return H @ W2 + b2

    # Cross-entropy loss
    loss = nn.CrossEntropyLoss(reduction='none')

    # Load Fashion-MNIST dataset
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # Train model
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

    # Predict with trained model
    predict_ch3(net, test_iter)
