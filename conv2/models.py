import torch
import torch.nn as nn
import torch.nn.functional as F

from alsh_conv import AlshConv2d, SRPTable

class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    

class AlshCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = AlshConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, is_first_layer=True, is_last_layer=False, num_hashes=5, num_tables=8, max_bits=16, hash_table=SRPTable)
        self.conv2 = None # dynamically set input channels for this
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        self.conv2 = nn.Conv2d(in_channels=x.size(1), out_channels=64, kernel_size=3, stride=1)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

        

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.Hardshrink(lambd=0.3),
            # nn.Softshrink(),
            # nn.ReLU(),
            # nn.Tanh(),
            # nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.Hardshrink(lambd=0.3),
            # nn.Softshrink(),
            # nn.ReLU(),
            # nn.Tanh(),
            # nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(3872, num_classes), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
