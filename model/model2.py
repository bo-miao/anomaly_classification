import torchvision.models as models
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class ResNetEnsemble(nn.Module):
    def __init__(self):
        super(ResNetEnsemble, self).__init__()
        model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-1])
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(512*3, 5)

    def forward(self, x):
        b, c, h, w = x.shape
        o1 = []
        for i in range(c // 3):
            o1.append(self.model(x[:, i * 3: (i + 1) * 3]).view(b, -1))

        x = torch.cat(o1, dim=1)
        x = self.fc(x)

        return x


class ResNetSingle(nn.Module):
    def __init__(self):
        super(ResNetSingle, self).__init__()
        model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-1])
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(2048, 4)

    def forward(self, x):
        b,c,h,w = x.shape
        x = self.model(x)
        x = self.fc(x.view(b, -1))

        return x


if __name__=='__main__':
    #model = torchvision.models.resnet50()
    from torch.cuda.amp import GradScaler

    model = ResNetSingle()
    a = torch.rand(2,3,256,256)
    print(model)
    with autocast():
        m = model(a)
        print(m.shape)


