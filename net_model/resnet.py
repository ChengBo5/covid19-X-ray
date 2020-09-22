import torchvision
from torch import nn

class Resnet18(nn.Module):

    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet_net = torchvision.models.resnet18(num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = 'Resnet18'

    def forward(self, x):
        logits = self.resnet_net(x)
        return logits

class Resnet50(nn.Module):

    def __init__(self):
        super(Resnet50, self).__init__()
        self.resnet_net = torchvision.models.resnet50(num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = 'Resnet50'

    def forward(self, x):
        logits = self.resnet_net(x)
        return logits

class Resnet101(nn.Module):

    def __init__(self):
        super(Resnet152, self).__init__()
        self.resnet_net = torchvision.models.resnet101(num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = 'Resnet101'

    def forward(self, x):
        logits = self.resnet_net(x)
        return logits

class Resnet152(nn.Module):

    def __init__(self):
        super(Resnet152, self).__init__()
        self.resnet_net = torchvision.models.resnet152(num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = 'Resnet152'

    def forward(self, x):
        logits = self.resnet_net(x)
        return logits
