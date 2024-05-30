import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

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

  def train(self):
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
        

  def model_to_json(self):
    state_dict = self.model.state_dict()
    def tensor_to_list(tensor):
        return tensor.cpu().numpy().tolist()

    json_compatible_state_dict = {key: tensor_to_list(value) for key, value in state_dict.items()}
    return json_compatible_state_dict

  def json_to_model(self, weights):
    def list_to_tensor(lst):
      return torch.tensor(lst)

    state_dict = {key: list_to_tensor(value) for key, value in weights.items()}
    self.model.load_state_dict(state_dict)

def main():
    
    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 64}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = CNN(train_loader, test_loader)
    model.train()
    model.json_to_model(model.model_to_json())
    model.test()

if __name__ == '__main__':
    main()