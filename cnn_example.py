import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import numpy as np
from client import Client

class MNISTDataset(Dataset):
    def __init__(self, image_file, label_file, transform):
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        label = self.labels[idx]

        return image, label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class CNN():

  def __init__(self, X, y):
    self.X = X
    self.y = y
    self.device = torch.device("cpu")
    self.model = Net().to(self.device)
    self.log_interval = 10
    self.optimizer = optim.Adadelta(self.model.parameters(), lr=1.0)
    self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.8)
    self.epoch = 0

  def train_local(self, num_iterations):
      self.model.train()
      for batch_idx, (data, target) in enumerate(self.X):
          data, target = data.to(self.device), target.to(self.device)
          self.optimizer.zero_grad()
          output = self.model(data)
          loss = F.nll_loss(output, target)
          loss.backward()
          self.optimizer.step()
          if batch_idx % self.log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  self.epoch, batch_idx * len(data), len(self.X.dataset),
                  100. * batch_idx / len(self.X), loss.item()))
          
          if num_iterations != -1 and batch_idx >= num_iterations:
             break
          
      self.epoch += 1

  def test(self):
      self.model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, target in self.y:
              data, target = data.to(self.device), target.to(self.device)
              output = self.model(data)
              test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()

      test_loss /= len(self.y.dataset)

      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(self.y.dataset),
        100. * correct / len(self.y.dataset)))
        

  def get_weights(self):
    state_dict = self.model.state_dict()
    def tensor_to_numpy(tensor):
        return tensor.cpu().numpy()

    json_compatible_state_dict = {key: tensor_to_numpy(value) for key, value in state_dict.items()}
    return json_compatible_state_dict


  def update_weights(self, weights):
    def list_to_tensor(lst):
      return torch.tensor(lst)

    state_dict = {key: list_to_tensor(value) for key, value in weights.items()}
    self.model.load_state_dict(state_dict)


def main():
    server_host = "127.0.0.1" #192.168.192.231"
    server_port = 5019

    train_kwargs = {'batch_size': 16}
    test_kwargs = {'batch_size': 16}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    

    dataset1 = MNISTDataset("../mnist_train/mnist-images-part-1.npy", "../mnist_train/mnist-labels-part-1.npy", transform)
    dataset2 = MNISTDataset("../mnist_test/test-images.npy", "../mnist_test/test-labels.npy", transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = CNN(train_loader, test_loader)
    client = Client(
       server_host, server_port, model, train_loader, test_loader
    )
    client.run()
    client.wait()
    client.model.test()


if __name__ == '__main__':
    main()