import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, class_num=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(3, 96, 11, stride=4, padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(3, 2),
                                     nn.Conv2d(96, 256, 5, stride=1, padding=(2, 2)),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(3, 2),
                                     nn.Conv2d(256, 384, 3, stride=1, padding=(1, 1)),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(384, 384, 3, stride=1, padding=(1, 1)),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(384, 256, 3, stride=1, padding=(1, 1)),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(3, stride=2, padding=0)
                                     )

        self.classifier = nn.Sequential(nn.Flatten(),
                                            nn.Dropout(0.5),
                                            nn.Linear(6*6*256, 2048),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Linear(2048, 2048),
                                            nn.ReLU(inplace=True),
                                            # nn.Dropout(0.5),
                                            nn.Linear(2048, class_num))
        #实际上这里的初始化是不需要的，因为pytorch会自动进行初始化，且初始化方法和这个一样
        # if init_weights:
        #     self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x


    # def _initialize_weights(self):
    #     for m in self.modules():
    #         nn.init.kaiming_normal_()


class AlexNet_half(nn.Module):
    def __init__(self, class_num=1000):
        super(AlexNet_half, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(3, 48, 11, stride=4, padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(3, 2),
                                     nn.Conv2d(48, 128, 5, stride=1, padding=(2, 2)),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(3, 2),
                                     nn.Conv2d(128, 192, 3, stride=1, padding=(1, 1)),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(192, 192, 3, stride=1, padding=(1, 1)),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(192, 128, 3, stride=1, padding=(1, 1)),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(3, stride=2, padding=0)
                                     )

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.5),
                                        nn.Linear(6 * 6 * 128, 2048),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(inplace=True),
                                        # nn.Dropout(0.5),
                                        nn.Linear(2048, class_num))

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
