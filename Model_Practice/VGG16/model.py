import torch
import torch.nn as nn

cfgs = {
    'vgg11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def make_feature(cfg:list):
    layers = []
    in_channel = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2)]
        else:
            layers += [nn.Conv2d(in_channel, v, 3, padding=1), nn.ReLU(inplace=True)]
            in_channel = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, feature, class_num = 1000, init_weights=False):
        super(VGG, self).__init__()
        self.feature = feature
        self.clssifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, class_num)
        )
        if init_weights:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.clssifier(x)
        return x


def vgg(model_name='vgg16', **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print('Warning: model number {} not in cfgs dict!'.format(model_name))
        exit(-1)
    model = VGG(make_feature(cfg), **kwargs)
    return model



















