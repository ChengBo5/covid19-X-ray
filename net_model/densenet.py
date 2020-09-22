import torchvision
from torch import nn


class DenseNetModel(nn.Module):

    def __init__(self):
        super(DenseNetModel, self).__init__()
        self.dense_net = torchvision.models.DenseNet(num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = 'DenseNetModel'

    def forward(self, x):
        logits = self.dense_net(x)
        return logits

class DenseNetModel201(nn.Module):

    def __init__(self):
        super(DenseNetModel201, self).__init__()
        self.dense_net = torchvision.models.densenet201(num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = 'DenseNetMode201'

    def forward(self, x):
        logits = self.dense_net(x)
        return logits

# %CheXNet pretrain
class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
        self.model_name = 'DenseNetModel121'

    def forward(self, x):
        x = self.densenet121(x)
        return x