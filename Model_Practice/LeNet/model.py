import torch.nn as nn

class LeNet_Max(nn.Module):
    def __init__(self):
        super(LeNet_Max, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(6, 16, 5),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2))
        self.classification = nn.Sequential(nn.Flatten(),
                                            nn.Linear(400, 120),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(120, 84),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(84, 10))

    def forward(self, x):
        x = self.feature(x)
        return self.classification(x)


class LeNet_Avg(nn.Module):
    def __init__(self):
        super(LeNet_Avg, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
                                     nn.ReLU(inplace=True),
                                     nn.AvgPool2d(2),
                                     nn.Conv2d(6, 16, 5),
                                     nn.ReLU(inplace=True),
                                     nn.AvgPool2d(2))
        self.classification = nn.Sequential(nn.Flatten(),
                                            nn.Linear(400, 120),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(120, 84),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(84, 10))

    def forward(self, x):
        x = self.feature(x)
        return self.classification(x)