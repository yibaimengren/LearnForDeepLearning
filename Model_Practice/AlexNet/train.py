import json

import torch
import torchvision.datasets
from torch.utils.data import DataLoader
import model
from torch import nn
from torch import optim
from torchvision import transforms

# trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])
data_transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'val':transforms.Compose([transforms.Resize((224, 224)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

train_dataset = torchvision.datasets.ImageFolder('dataset/train', transform=data_transform['train'])
test_dataset = torchvision.datasets.ImageFolder('dataset/test', transform=data_transform['val'])

#获取索引对应类别
flower_list = train_dataset.class_to_idx
class_dict = dict((val, key)for key, val in flower_list.items())
#写入json文件
json_str = json.dumps(class_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

#num_workers读取数据的线程数 在window系统下只能设为0
train_loader = DataLoader(train_dataset, 32, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, 32, shuffle=True, num_workers=8)

epoch = 300
lr = 0.0002
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

net = model.AlexNet_half(5).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=lr)

best_accuracy = 0.0
for epoch_index in range(epoch):
    print(f'epoch:{epoch_index}')
    net.train()  #使用train()来开启dropout，使用eval()关闭dropout
    for step, data in enumerate(train_loader):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'\r训练进度[{step}/{len(train_loader)}]',end="")

    with torch.no_grad():
        net.eval()
        correct_sum = 0
        for step, data in enumerate(test_loader):
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            correct = (outputs.argmax(1) == targets).sum()
            correct_sum += correct
        accuracy = correct_sum/len(test_dataset)
        if best_accuracy < accuracy:
            best_accuracy = accuracy
        print(f'accuracy:{accuracy}')

print(f'best_accuracy:{best_accuracy}')
