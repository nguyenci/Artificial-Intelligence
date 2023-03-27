import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# HW6
# Author: Cinthya Nguyen
# Class: CS540 SP23


def get_data_loader(training=True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if training:
        train_set = datasets.FashionMNIST('./ data', train=True, download=True,
                                          transform=custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    else:
        test_set = datasets.FashionMNIST('./ data', train=False, transform=custom_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size=64)

    return loader


def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 128),
        nn.ReLU(), nn.Linear(128, 64),
        nn.ReLU(), nn.Linear(64, 10)
    )

    return model


def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader - the train DataLoader produced by the first function
        criterion - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """

    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    batch_size = 64
    for epoch in range(T):
        avg_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if i % 60000 == 32:  # last batch
                avg_loss += loss.item() * 32
            else:
                avg_loss += loss.item() * batch_size

        print(f'Train Epoch: {epoch} Accuracy: {correct}/{total}('
              f'{100 * (correct/total):.2f}%) '
              f'Loss: {avg_loss/total:.3f}')


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    INPUT: 
        model - the trained model produced by the previous function
        test_loader - the test DataLoader
        criterion - cropy-entropy

    RETURNS:
        None
    """

    model.eval()

    avg_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loss = criterion(outputs, labels)
            avg_loss += loss

        if show_loss:
            print(f'Average loss: {avg_loss/len(test_loader):.4f}')

        print(f'Accuracy: {100 * (correct/total)}%')

    # model.train()


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images - a tensor test image set of shape Nx1x28x28
        index - specific index  i of the image to be tested: 0 <= i <= N - 1

    RETURNS:
        None
    """

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                   'Sneaker', 'Bag', 'Ankle Boot']

    image = test_images[index, 0, :, :]
    model.eval()
    prob = F.softmax(model(image.reshape([1, 1, 28, 28])), dim=1)
    # print(prob)
    results = torch.argsort(prob, dim=1, descending=True)[0]  # sort in descending order

    for i in range(3):
        num_class = results[i]
        label = class_names[num_class]
        print(f'{label}: {100 * prob[0][num_class]:.2f}%')


if __name__ == '__main__':
    """
    Main.
    """
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    # print(type(train_loader))
    # print(train_loader.dataset)
    test_loader = get_data_loader(training=False)
    model = build_model()
    # print(model)
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, False)
    test_images, test_labels = next(iter(test_loader))
    predict_label(model, test_images, 1)
